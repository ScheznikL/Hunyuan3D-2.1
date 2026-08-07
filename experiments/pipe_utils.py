import sys
import os
from peft import PeftModel
import torch
import yaml
from PIL import Image

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipelineInAlighnedSpace
from hy3dshape.utils import instantiate_from_config


def setup_pipeline_with_custom_weights(ckpt_path, 
                                       config_path,
                                       model_path = 'tencent/Hunyuan3D-2mini', 
                                       default_subfolder='hunyuan3d-dit-v2-mini',
                                       use_peft = True):
    
    # shape
     #'tencent/Hunyuan3D-2.1'
    pipeline = Hunyuan3DDiTFlowMatchingPipelineInAlighnedSpace.from_pretrained(
        model_path,
        subfolder = default_subfolder,
        )
    
    # ----------------
    config = yaml.safe_load(open(config_path, 'r'))
    print("[INFO] initializing from this config:")
    print(config)
    
    custom_model = instantiate_from_config(config['model']['params']['denoiser_cfg'])

    sd = torch.load(ckpt_path, weights_only = False, map_location='cpu')# 
    if 'state_dict' in sd:
        sd = sd['state_dict']
    
    ###OLD version
    ###sd = {k.replace('model.', '').replace('_forward_module.', ''): v for k, v in sd.items()}
    
    # Замість стандартного очищення, витягуємо саме EMA копію
    use_ema_weights = True # поставте True, щоб використати EMA

    if use_ema_weights:
        # 1. Беремо тільки ті ключі, що належать до EMA
        # 2. Перетворюємо 'model_ema.s_name' назад у нормальні імена 'model.name'
        # Примітка: LitEma замінює крапки на '_____', це треба повернути назад
        ema_sd = {}
        for k, v in sd.items():
            if 'model_ema' in k:
                # Видаляємо префікс EMA та повертаємо крапки замість '_____'
                new_key = k.replace('model_ema.', '')
                # Якщо ваш клас LitEma використовував заміну крапок:
                new_key = new_key.replace('_____', '.') 
                ema_sd[new_key] = v
        sd = ema_sd
    else:
        # Звичайне завантаження тренувальних ваг
        sd = {k.replace('model.', '').replace('_forward_module.', ''): v for k, v in sd.items()}


    # Завантажуємо ваги в модель
    msg = custom_model.load_state_dict(sd, strict=False)
    custom_model = custom_model.to('cuda').half()
    print(f"Model Load Status: {msg}")
    if any("lora_" in key for key in msg.unexpected_keys):
        print("[INFO] LoRA keys detected. Adapting model structure...")
        pipeline.model = adapt_missing_keys(custom_model=custom_model, ckpt_path= ckpt_path)
    else:
        # 4. Переводимо в потрібний формат і замінюємо модель у пайплайні
        custom_model = custom_model.cuda().half()
        
        # Якщо custom_model використовує Peft всередині:
        if use_peft:
            if hasattr(custom_model, "merge_and_unload"):
                custom_model = custom_model.merge_and_unload()
                print("LoRA weights merged into base model.")  
                
        pipeline.model = custom_model
    
    
    return pipeline

def adapt_missing_keys(custom_model, ckpt_path):
    from peft import get_peft_model, LoraConfig


    # 2. Обов'язково ініціалізуєте LoRA-структуру (параметри мають бути як при тренуванні!)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["img_attn.qkv", "img_attn.proj", "txt_attn.qkv", "txt_attn.proj"], 
        init_lora_weights=False
    )
    custom_model = get_peft_model(custom_model, lora_config)

    # 3. Тепер завантажуєте ваги - ключі lora_A/B тепер знайдуть свої місця
    sd = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in sd:
        sd = sd['state_dict']

    # Очищення префіксів для PeftModel
    clean_sd = {k.replace('model.', 'base_model.model.'): v for k, v in sd.items()}
    custom_model.load_state_dict(clean_sd, strict=False)
    custom_model = custom_model.merge_and_unload()
    
    return custom_model


def perf_merge(pipeline, ckpt_path):
    from peft import PeftModel

    # 1. Завантажуєте базу
    base_model = pipeline.model 

    # 2. Накладаєте адаптер з чекпоїнта
    model = PeftModel.from_pretrained(base_model, ckpt_path)

    # 3. Зливаєте ваги фізично (W_new = W_base + A*B)
    model = model.merge_and_unload() 

    return model