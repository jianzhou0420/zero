
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

# DGL
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu116.html

pip install -e .



# pip install torch_scatter --find-links=https://data.pyg.org/whl/torch-1.13.1+cu116