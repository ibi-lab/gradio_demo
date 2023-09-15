# ライブラリインポート
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'Track-Anything/')
from track_anything import TrackingAnything

# SAMの初期設定
sam_vit_h = ["vit_h", "checkpoints/sam_vit_h_4b8939.pth"]
sam_vit_l = ["vit_l", "checkpoints/sam_vit_l_0b3195.pth"]
model_type = sam_vit_l
device = "cuda"
sam = sam_model_registry[model_type[0]](checkpoint=model_type[1])
sam.to(device=device)
predictor = SamPredictor(sam)

# XMemの初期設定
xmem_checkpoint ="./checkpoints/XMem-s012.pth"
model = TrackingAnything(None, xmem_checkpoint, None, None)
model.xmem.clear_memory()

cap = cv2.VideoCapture('test.mp4')

ret, frame = cap.read()
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
predictor.set_image(image) # 画像をembeddingにする

input_point = np.array([[335, 260]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

masks, logits, painted_images = model.generator(images=frames, template_mask=masks[0])
painted_images = np.array(painted_images)
