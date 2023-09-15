from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'Track-Anything/')
from track_anything import TrackingAnything

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread('test.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image) # 画像をembeddingにする

input_point = np.array([[335, 260]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

cap = cv2.VideoCapture('test.mp4')
# test.mp4のfps
fps = cap.get(cv2.CAP_PROP_FPS)
# test.mp4の総フレーム数
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(fps, frame_count)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
# frames = np.array(frames)
print(len(frames))
    

xmem_checkpoint ="./checkpoints/XMem-s012.pth"
model = TrackingAnything(None, xmem_checkpoint, None, None)
model.xmem.clear_memory()
masks, logits, painted_images = model.generator(images=frames, template_mask=masks[0])
print(type(masks))
print(len(masks))
print(masks[0].shape)
painted_images = np.array(painted_images)
# painted_imagesを描画
# 動画のフレームレートとサイズを設定
fps = 30
width, height = painted_images[0].shape[1], painted_images[0].shape[0]

# 動画ファイルを書き込むための設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# 動画ファイルにフレームを書き込む
for i in range(len(painted_images)):
    out.write(painted_images[i])
    
out.release()

# print(masks.shape)