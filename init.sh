source /etc/profile.d/conda.sh
conda create -y -p /dcs/large/u5745134/hunyuanenv python=3.10
conda activate /dcs/large/u5745134/hunyuanenv

#pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

cd hy3dgen/texgen/custom_rasterizer
python setup.py install

cd ../differentiable_renderer
python setup.py install