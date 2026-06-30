import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration


class MT5TextEncoder(nn.Module):
    def __init__(
        self,
        model_name="google/mt5-large",
        out_dim=1536,
        max_length=128,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name).eval()

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.max_length = max_length
        self.text_dim = self.encoder.config.d_model  # 4096 for mt5-large

        self.adapter = TextCondAdapter(
            in_dim=self.text_dim,
            out_dim=out_dim,
        )

    @torch.no_grad()
    def forward(self, texts):
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.encoder.device)

        out = self.encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )

        return self.adapter(
            emb=out.last_hidden_state,
            mask=tokens.attention_mask,
        )

    def unconditional_embedding(self, batch_size, **kwargs):
        device = next(self.encoder.parameters()).device
        dtype = next(self.encoder.parameters()).dtype

        emb = torch.zeros(
            batch_size,
            self.max_length,
            self.text_dim,
            device=device,
            dtype=dtype,
        )
        mask = torch.zeros(
            batch_size,
            self.max_length,
            device=device,
            dtype=torch.bool,
        )

        return self.adapter(emb=emb, mask=mask)

# class TextCondAdapter(nn.Module):
#     def __init__(self):
#         self.proj = nn.Linear(4096, 1536)
#         self.norm = nn.LayerNorm(1536)

class TextCondAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, emb, mask):
        emb = self.proj(emb)          # [B, L, D_model]
        return {
            "emb": emb,
            "mask": mask
        }
        #return self.proj(x)
    
class MT5Embedder(nn.Module):
    available_models = ["t5-v1_1-xxl"]

    def __init__(
        self,
        model_dir="t5-v1_1-xxl",
        model_kwargs=None,
        torch_dtype=None,
        use_tokenizer_only=False,
        conditional_generation=False,
        max_length=128,
    ):
        super().__init__()
        self.device = "cpu"
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.max_length = max_length
        if model_kwargs is None:
            model_kwargs = {
                # "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }
        model_kwargs["device_map"] = {"shared": self.device, "encoder": self.device}
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if use_tokenizer_only:
            return
        if conditional_generation:
            self.model = None
            self.generation_model = T5ForConditionalGeneration.from_pretrained(
                model_dir
            )
            return
        self.model = (
            T5EncoderModel.from_pretrained(model_dir, **model_kwargs)
            .eval()
            .to(self.torch_dtype)
        )

    def get_tokens_and_mask(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        tokens = text_tokens_and_mask["input_ids"][0]
        mask = text_tokens_and_mask["attention_mask"][0]
        
        # tokens = torch.tensor(tokens).clone().detach()
        # mask = torch.tensor(mask, dtype=torch.bool).clone().detach()
        return tokens, mask

    def get_text_embeddings(self, texts, attention_mask=True, layer_index=-1):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=text_tokens_and_mask["input_ids"].to(self.device),
                attention_mask=(
                    text_tokens_and_mask["attention_mask"].to(self.device)
                    if attention_mask
                    else None
                ),
                output_hidden_states=True,
            )
            text_encoder_embs = outputs["hidden_states"][layer_index].detach()

        return text_encoder_embs, text_tokens_and_mask["attention_mask"].to(self.device)

    @torch.no_grad()
    def __call__(self, tokens, attention_mask, layer_index=-1):
        with torch.amp.autocast('cuda'):
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        z = outputs.hidden_states[layer_index].detach()
        return z

    def general(self, text: str):
        # input_ids = input_ids = torch.tensor([list(text.encode("utf-8"))]) + num_special_tokens
        input_ids = self.tokenizer(text, max_length=128).input_ids
        print(input_ids)
        outputs = self.generation_model(input_ids)
        return outputs

# class ConditionTokenRouter:
#     def __call__(self, image_tokens, text_tokens):
#         # concatenate along sequence dimension
#         tokens = torch.cat([text_tokens, image_tokens], dim=1)
#         token_types = {
#             "text_len": text_tokens.shape[1],
#             "image_len": image_tokens.shape[1],
#         }
#         return tokens, token_types
