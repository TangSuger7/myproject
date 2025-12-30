# conda create -n uni python=3.8 -y
# eval "$(conda shell.bash hook)"
# conda activate uni

# pip install /home/notebook/code/personal/80410839/sources/JYF/torch-1.13.0+cu117-cp38-cp38-linux_x86_64.whl /home/notebook/code/personal/80410839/sources/JYF/torchvision-0.14.0+cu117-cp38-cp38-linux_x86_64.whl
# pip install six opencv-python-headless
# # conda install torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
# pip install -r requirements.txt

# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/
# conda config --set show_channel_urls yes

# conda create -n gs2 python=3.10 -y
# eval "$(conda shell.bash hook)"
# conda activate gs2

# pip install /home/notebook/code/personal/80410839/sources/JYF/triton-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# pip install /home/notebook/code/personal/80410839/sources/JYF/torch-2.3.1+cu121-cp310-cp310-linux_x86_64.whl /home/notebook/code/personal/80410839/sources/JYF/torchvision-0.18.1+cu121-cp310-cp310-linux_x86_64.whl /home/notebook/code/personal/80410839/sources/JYF/torchaudio-2.3.1+cu121-cp310-cp310-linux_x86_64.whl

# pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# # pip install setuptools==58.0.0 wheel==0.36.2
# pip install optuna
# pip install optuna-dashboard
# pip install plotly
# pip install optuna-integration[pytorch_distributed]
# pip install /home/notebook/code/personal/80410839/sources/JYF/xformers-0.0.27-cp310-cp310-manylinux2014_x86_64.whl
# pip install opencv-python==4.7.0.72 opencv-contrib-python==4.7.0.72 numpy==1.24.4 hydra-core==1.3.2 tqdm==4.66.1 iopath==0.1.10 pillow==9.4.0 matplotlib==3.9.1 jupyter==1.0.0 eva-decord==0.6.1 Flask==3.0.3 Flask-Cors==5.0.0 av==13.0.0 dataclasses-json==0.6.7 eva-decord==0.6.1 gunicorn==23.0.0 imagesize==1.4.1 pycocotools==2.0.8 strawberry-graphql==0.243.0 submitit==1.5.1 black==24.2.0 usort==1.0.2 ufmt==2.0.0b2  fvcore==0.1.5.post20221221  pandas==2.2.2 scikit-image==0.24.0 tensorboard==2.17.0  tensordict==0.4.0 supervision transformers addict yapf timm 
# pip install opencv-python-headless easydict

# conda deactivate

cp /home/notebook/code/personal/80410839/Uni/LAB/pytorch_distributed.py /home/oppoer/.local/lib/python3.10/site-packages/optuna_integration/pytorch_distributed/pytorch_distributed.py 
cp /home/notebook/code/personal/80410839/Uni/__init__.py /home/oppoer/.local/lib/python3.10/site-packages/torch/__init__.py
cd /home/notebook/code/personal/80410839/Uni/FIND_FINAL && sh run.sh