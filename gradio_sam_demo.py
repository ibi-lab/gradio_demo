import gradio as gr
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torch
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

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return gr.update(maximum=total_frames, visible=True, value=1)

def extract_frame(frame_slider, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_slider -= 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # template_frame.update(value=frame, visible=True) # これは動かない inputsにオブジェクト入れても
    # return gr.update(value=frame, visible=True) # これは動く
    return gr.Image.update(value=frame, visible=True) # これも動く
    # return frame # これも動く ただしvisibleなど、他の属性を更新できないから、update使った方がよさげ

def sam_example(template_frame, evt:gr.SelectData):
    point = [evt.index[0], evt.index[1]]
    input_point = np.array([point])
    input_label = np.array([1])
    predictor.set_image(template_frame) # 画像をembeddingにする
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    sam_frame = template_frame.copy()
    mask = np.zeros_like(template_frame)
    mask[masks[0]] = [255, 0, 0]
    alpha = 0.5
    sam_frame = cv2.addWeighted(sam_frame, 1, mask, alpha, 0)
    return gr.update(value=sam_frame, visible=True)
    

# Gradio UIのセットアップ
with gr.Blocks() as iface:
    # ビデオを選択するためのUI
    video = gr.inputs.Video(type="mp4", label="Input video")
    # フレームを選択するスライダー
    frame_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Track start frame", interactive=True, visible=False)
    # 任意のフレームの画像を表示する
    template_frame = gr.Image(type="numpy",interactive=True, visible=False, image_mode="RGB")
    # SAMで処理した画像を表示する
    sam_frame = gr.Image(type="numpy",interactive=False, visible=False, image_mode="RGB")
    
    # process buttonが押されたら、get_frame_countにvideoを入力し、get_frame_count内でframe_sliderの
    # maximum、visible、valueを更新する。(outputsは出力であって、入力してないのに、関数内でframe_sliderを更新するのは直感に反するきがする)
    video.upload(
        fn=get_frame_count,
        inputs=video,
        outputs=[frame_slider],
    )
    
    frame_slider.release(
        fn=extract_frame,
        inputs=[frame_slider, video],
        outputs=[template_frame],
    )
    
    template_frame.select(
        fn=sam_example,
        inputs=[template_frame],
        outputs=[sam_frame]
    )
    
    
iface.launch()