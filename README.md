# gradio_demo

## Python Version: 3.10.12

## Install
poetry install
git clone https://github.com/gaomingqi/Track-Anything.git
cp Track-Anything_modify/track_anything.py ./Track-Anything/
cp Track-Anything_modify/base_tracker.py ./Track-Anything/tracker/
cp Track-Anything_modify/network.py ./Track-Anything/tracker/model/

mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth

## Execution
poetry run python gradio_sam_demo.py
