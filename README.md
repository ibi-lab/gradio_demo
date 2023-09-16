# gradio_demo

## Python Version: 3.10.12
## CUDA Version: 11.7

## Install
cd gradio_demo  
poetry install  
poetry run python -m spacy download en_core_web_sm  
git clone https://github.com/gaomingqi/Track-Anything.git  
cp Track-Anything_modify/track_anything.py ./Track-Anything/  
cp Track-Anything_modify/base_tracker.py ./Track-Anything/tracker/  
cp Track-Anything_modify/network.py ./Track-Anything/tracker/model/  

## Download Checkpoints
mkdir checkpoints  
cd checkpoints  
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth  
wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth  
cd ..  

## Execution
poetry run python gradio_sam_demo.py  
