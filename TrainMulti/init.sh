cd /home/notebook/data/group/00group/users/myjyf/MultiSeg/lab3829

sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
sudo apt-get update && sudo apt-get install libgl1



sudo pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
sudo pip install numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
sudo pip install scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
sudo pip install opencv_python-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
sudo pip install opencv-python-headless
sudo pip install hydra-core==1.3.2
sudo pip install pysodmetrics==1.4.2
sudo pip install einops
sudo pip install -U huggingface-hub
sudo pip install accelerate

sudo pip install timm

sudo pip install kornia
sudo sh train.sh