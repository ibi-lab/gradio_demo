import gradio as gr
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import torch
import torchvision
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "Track-Anything/")
from track_anything import TrackingAnything
from tqdm import tqdm
import spacy
from PIL import Image
import mmcv
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
import os
import time

from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration

# SAMの初期設定
sam_vit_h = ["vit_h", "checkpoints/sam_vit_h_4b8939.pth"]
sam_vit_l = ["vit_l", "checkpoints/sam_vit_l_0b3195.pth"]
model_type = sam_vit_l
device = "cuda"
sam = sam_model_registry[model_type[0]](checkpoint=model_type[1])
sam.to(device=device)
predictor = SamPredictor(sam)

# セマンティック(ade20k, coco, blip, clip)の初期設定
rank = "cuda"
oneformer_ade20k_processor = OneFormerProcessor.from_pretrained(
    "shi-labs/oneformer_ade20k_swin_tiny"
)
oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained(
    "shi-labs/oneformer_ade20k_swin_tiny"
).to(rank)
oneformer_coco_processor = OneFormerProcessor.from_pretrained(
    "shi-labs/oneformer_coco_swin_large"
)
oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained(
    "shi-labs/oneformer_coco_swin_large"
).to(rank)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(rank)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(rank)
clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd16")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd16").to(
    rank
)
clipseg_processor.image_processor.do_resize = False

# spacyの初期設定
nlp = spacy.load("en_core_web_sm")


# cocoとade20kの関数
def oneformer_coco_segmentation(
    image, oneformer_coco_processor, oneformer_coco_model, rank
):
    inputs = oneformer_coco_processor(
        images=image, task_inputs=["semantic"], return_tensors="pt"
    ).to(rank)
    outputs = oneformer_coco_model(**inputs)
    predicted_semantic_map = (
        oneformer_coco_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
    )
    return predicted_semantic_map


def oneformer_ade20k_segmentation(
    image, oneformer_ade20k_processor, oneformer_ade20k_model, rank
):
    inputs = oneformer_ade20k_processor(
        images=image, task_inputs=["semantic"], return_tensors="pt"
    ).to(rank)
    outputs = oneformer_ade20k_model(**inputs)
    predicted_semantic_map = (
        oneformer_ade20k_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
    )
    return predicted_semantic_map


# XMemの初期設定
xmem_checkpoint = "./checkpoints/XMem-s012.pth"
model = TrackingAnything(None, xmem_checkpoint, None, None)
model.xmem.clear_memory()


# blipのcaptionから名詞句を抽出する関数
def get_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        noun_phrases.append(chunk.text)
    return noun_phrases


def open_vocabulary_classification_blip(raw_image, blip_processor, blip_model, rank):
    # unconditional image captioning
    captioning_inputs = blip_processor(raw_image, return_tensors="pt").to(rank)
    out = blip_model.generate(**captioning_inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    ov_class_list = get_noun_phrases(caption)
    return ov_class_list


# CLIPの関数
def clip_classification(image, class_list, top_k, clip_processor, clip_model, rank):
    inputs = clip_processor(
        text=class_list, images=image, return_tensors="pt", padding=True
    ).to(rank)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    if top_k == 1:
        class_name = class_list[probs.argmax().item()]
        return class_name
    else:
        top_k_indices = probs.topk(top_k, dim=1).indices[0]
        top_k_class_names = [class_list[index] for index in top_k_indices]
        return top_k_class_names


def clipseg_segmentation(image, class_list, clipseg_processor, clipseg_model, rank):
    if isinstance(class_list, str):
        print(len(image), len(class_list))
        print(type(image), len(class_list))
        print(image.shape)
        class_list = [
            class_list,
        ]
    inputs = clipseg_processor(
        text=class_list,
        images=[image] * len(class_list),
        padding=True,
        return_tensors="pt",
    ).to(rank)
    # resize inputs['pixel_values'] to the longest side of inputs['pixel_values']
    h, w = inputs["pixel_values"].shape[-2:]
    fixed_scale = (512, 512)
    inputs["pixel_values"] = F.interpolate(
        inputs["pixel_values"], size=fixed_scale, mode="bilinear", align_corners=False
    )
    outputs = clipseg_model(**inputs)
    try:
        logits = F.interpolate(
            outputs.logits[None], size=(h, w), mode="bilinear", align_corners=False
        )[0]
    except Exception as e:
        logits = F.interpolate(
            outputs.logits[None, None, ...],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]
    return logits


# フレーム数取得
def get_frame_count(video):
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return gr.update(maximum=total_frames, visible=True, value=1)


# sliderより、指定したフレームを取得
def extract_frame(frame_slider, video, template_frame_state, points_state):
    cap = cv2.VideoCapture(video)
    frame_slider -= 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    print(height, width)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    template_frame_state = frame.copy()
    points_state.clear()
    # template_frame.update(value=frame, visible=True) # これは動かない inputsにオブジェクト入れても
    # return gr.update(value=frame, visible=True) # これは動く
    return (
        gr.Image.update(value=frame, visible=True, height=480, width=860),
        template_frame_state,
        gr.update(visible=True),
        gr.update(visible=True),
        points_state,
    )  # これも動く
    # return frame # これも動く ただしvisibleなど、他の属性を更新できないから、update使った方がよさげ


# maskからbboxを取得
def mask_to_bbox(mask):
    # maskをnumpy配列に変換
    mask = mask.cpu().numpy()
    # maskの非ゼロピクセルの座標を取得
    coords = np.argwhere(mask)
    # x座標の最小値、y座標の最小値、x座標の最大値、y座標の最大値を計算
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    # bboxを計算
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    return bbox


# imgとmaskを入力し、画像の中に含まれるクラスを1つだけ取得
def ssa_example(img, mask):
    with torch.no_grad():
        img = img.copy()

        # ade20kの結果を取得
        class_ids_from_oneformer_coco = oneformer_coco_segmentation(
            Image.fromarray(img), oneformer_coco_processor, oneformer_coco_model, rank
        )
        class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(
            Image.fromarray(img),
            oneformer_ade20k_processor,
            oneformer_ade20k_model,
            rank,
        )
        # for ann in tqdm(mask, leave=False):
        valid_mask = torch.tensor(mask).bool()

        # valid_maskごとにade20kの結果を取得
        coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
        ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]

        # 結果の中から、最も多く出現したクラスを1つだけ取得
        top_k_coco_propose_classes_ids = (
            torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
        )
        top_k_ade20k_propose_classes_ids = (
            torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices
        )
        # cocoとade20kの和集合を取得
        local_class_names = set()
        local_class_names = set.union(
            local_class_names,
            set(
                [
                    CONFIG_ADE20K_ID2LABEL["id2label"][str(class_id.item())]
                    for class_id in top_k_ade20k_propose_classes_ids
                ]
            ),
        )
        local_class_names = set.union(
            local_class_names,
            set(
                (
                    [
                        CONFIG_COCO_ID2LABEL["refined_id2label"][str(class_id.item())]
                        for class_id in top_k_coco_propose_classes_ids
                    ]
                )
            ),
        )

        # 今後の処理で使う1.2倍(patch_small), 1.6倍(patch_large), 1.6倍(patch_huge、なぜかlargeと同じサイズ)のsamで切り取った画像を取得
        # valid_mask_huge_cropはSAMの領域を切り取った画像??
        scale_small = 1.2
        scale_large = 1.6
        scale_huge = 1.6

        bbox_x, bbox_y, bbox_w, bbox_h = mask_to_bbox(valid_mask)

        patch_small = mmcv.imcrop(
            img,
            np.array(
                [
                    bbox_x,
                    bbox_y,
                    bbox_x + bbox_w,
                    bbox_y + bbox_h,
                ]
            ),
            scale=scale_small,
        )
        patch_large = mmcv.imcrop(
            img,
            np.array(
                [
                    bbox_x,
                    bbox_y,
                    bbox_x + bbox_w,
                    bbox_y + bbox_h,
                ]
            ),
            scale=scale_large,
        )
        patch_huge = mmcv.imcrop(
            img,
            np.array(
                [
                    bbox_x,
                    bbox_y,
                    bbox_x + bbox_w,
                    bbox_y + bbox_h,
                ]
            ),
            scale=scale_huge,
        )
        valid_mask_huge_crop = mmcv.imcrop(
            valid_mask.numpy(),
            np.array(
                [
                    bbox_x,
                    bbox_y,
                    bbox_x + bbox_w,
                    bbox_y + bbox_h,
                ]
            ),
            scale=scale_huge,
        )

        # 1.6倍の画像(patch_large)を用いて、画像の中に含まれるクラスを取得。 画像から状況を文で説明し、その文からget_noun_phrases関数で名詞句を抽出し、返している
        try:
            op_class_list = open_vocabulary_classification_blip(
                patch_large, blip_processor, blip_model, rank
            )
            # cocoとade20kの和集合と、名詞句を抽出した結果の和集合を取得
            local_class_list = list(
                set.union(local_class_names, set(op_class_list))
            )  # , set(refined_imagenet_class_names)

            # openai/clip-vitを使って、和集合のlocal_class_listから、画像の中に含まれるクラスを3つ取得？
            mask_categories = clip_classification(
                patch_small,
                local_class_list,
                3 if len(local_class_list) > 3 else len(local_class_list),
                clip_processor,
                clip_model,
                rank,
            )
            # mask_categories(clip_classificationの戻り値)がstr型の場合、list型に変換
            if isinstance(mask_categories, str):
                mask_categories = [mask_categories]
            # 得られた3つのクラス(mask_categories)から、それがどこにあるのかをCIDAS/clipseg-rd16を使って推定？
            class_ids_patch_huge = clipseg_segmentation(
                patch_huge, mask_categories, clipseg_processor, clipseg_model, rank
            ).argmax(0)
            # テンソルに変換
            valid_mask_huge_crop = torch.tensor(valid_mask_huge_crop)

            # ????
            if valid_mask_huge_crop.shape != class_ids_patch_huge.shape:
                valid_mask_huge_crop = (
                    F.interpolate(
                        valid_mask_huge_crop.unsqueeze(0).unsqueeze(0).float(),
                        size=(
                            class_ids_patch_huge.shape[-2],
                            class_ids_patch_huge.shape[-1],
                        ),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .bool()
                )

            top_1_patch_huge = (
                torch.bincount(class_ids_patch_huge[valid_mask_huge_crop].flatten())
                .topk(1)
                .indices
            )
            top_1_mask_category = mask_categories[top_1_patch_huge.item()]

        except Exception as e:
            print("clip error")
            index = top_k_ade20k_propose_classes_ids[0].item()
            top_1_mask_category = CONFIG_ADE20K_ID2LABEL["id2label"][str(index)]

        append_classname = str(top_1_mask_category)
        return append_classname
        print(append_classname)


# SAMで処理した画像を表示する関数
def sam_example(template_frame_state, points_state, mask_state):
    if len(points_state) > 1:
        input_point = np.array(points_state)
        input_label = np.array([1] * len(points_state))
    else:
        input_point = np.array(points_state)
        input_label = np.array([1])

    print(input_point)
    print(input_label)
    # point = points[0]
    # input_point = np.array([point])
    # input_label = np.array([1])
    predictor.set_image(template_frame_state)  # 画像をembeddingにする
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    mask_state = masks[0]
    label = ssa_example(template_frame_state, masks[0])
    sections = [(masks[0], label)]
    # sam_frame = template_frame.copy()
    # mask = np.zeros_like(template_frame)
    # mask[masks[0]] = [255, 0, 0]
    # alpha = 0.5
    # sam_frame = cv2.addWeighted(sam_frame, 1, mask, alpha, 0)
    # return (template_frame, sections)
    return gr.update(value=(template_frame_state, sections), visible=True), mask_state, gr.update(visible=True)
    return gr.update(
        value=template_frame_state,
        visible=True,
    )
    return gr.update(value=sam_frame, visible=True)


# 指定されたポイントの座標にプラス印をつける関数
def draw_crosshair(template_frame, points_state, evt: gr.SelectData):
    color = (255, 0, 0)
    size = 15
    thickness = 2
    x, y = evt.index[0], evt.index[1]
    point = [x, y]
    points_state.append(point)
    img_ = template_frame.copy()
    cv2.line(img_, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img_, (x, y - size), (x, y + size), color, thickness)
    return gr.update(value=img_, visible=True), points_state


def remove_crosshair(template_frame_state, points_state):
    points_state.clear()
    img_ = template_frame_state.copy()
    return gr.update(value=img_, visible=True), points_state


def tracking(frame_slider, video, mask_state):
    frame_slider -= 1
    model.xmem.clear_memory()
    frames = []

    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    masks, logits, painted_images = model.generator(
        images=frames, template_mask=mask_state, change_color_band=True
    )
    masks = np.array(masks)
    print(masks.shape)
    painted_images = np.array(painted_images)

    frames = torch.from_numpy(np.asarray(painted_images))
    yy_mm_dd = time.strftime("%Y-%m-%d", time.localtime())
    output_path = f"results/output_{yy_mm_dd}.mp4"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")

    return gr.update(value=output_path, visible=True)


# Gradio UIのセットアップ
with gr.Blocks() as iface:
    points_state = gr.State([])
    template_frame_state = gr.State()
    mask_state = gr.State()

    with gr.Row():  # Rowの中に2つのColumnを配置すると、2列で表示される。このとき、Rowとは画面の上から下まですべての領域を指す
        with gr.Column():
            # ビデオを選択するためのUI
            video = gr.Video(type="mp4", label="Input video")
            # フレームを選択するスライダー
            frame_slider = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                label="Track start frame",
                interactive=True,
                visible=False,
            )
            # 任意のフレームの画像を表示する
            template_frame = gr.Image(
                type="numpy", interactive=True, visible=False, image_mode="RGB"
            )

            with gr.Row():
                # SSM処理するボタンを表示する
                process_button = gr.Button(
                    value="SSM", type="button", interactive=True, visible=False
                )
                # プラス印を削除するボタンを表示する
                delete_button = gr.Button(
                    value="Delete", type="button", interactive=True, visible=False
                )

        with gr.Column():
            # フレームにプラス印をつけた画像を表示する
            crosshair_frame = gr.Image(
                type="numpy",
                interactive=True,
                visible=False,
                image_mode="RGB",
                height=480,
                width=860,
            )
            # SAMで処理した画像を表示する
            img_output = gr.AnnotatedImage(
                visible=False, interactive=False, label="Output image"
            )
            # Trackingのボタンを表示する
            track_button = gr.Button(value="Track", type="button", interactive=True, visible=False)
            # Trackingの結果を表示する
            track_video = gr.Video(type="mp4", label="Output video", visible=False)

    # process buttonが押されたら、get_frame_countにvideoを入力し、get_frame_count内でframe_sliderの
    # maximum、visible、valueを更新する。(outputsは出力であって、入力してないのに、関数内でframe_sliderを更新するのは直感に反するきがする)
    video.upload(
        fn=get_frame_count,
        inputs=video,
        outputs=[frame_slider],
    )

    # frame_sliderが調整されたら、その調整した値のフレームを取得し、template_frameに表示する
    # また、template_frame_img(gr.State)にも同じフレームを保存する
    frame_slider.release(
        fn=extract_frame,
        inputs=[frame_slider, video, template_frame_state, points_state],
        outputs=[template_frame, template_frame_state, process_button, delete_button, points_state],
    )

    # template_frameにプラス印をつける
    template_frame.select(
        fn=draw_crosshair,
        inputs=[template_frame, points_state],
        outputs=[template_frame, points_state],
    )

    # process buttonが押されたら、プラス印のポイントをもとにSSAした画像とラベルをimg_outputに出力する
    process_button.click(
        fn=sam_example,
        inputs=[template_frame_state, points_state, mask_state],
        outputs=[img_output, mask_state, track_button],
    )

    # delete buttonが押されたら、プラス印を削除する
    delete_button.click(
        fn=remove_crosshair,
        inputs=[template_frame_state, points_state],
        outputs=[template_frame, points_state],
    )

    track_button.click(
        fn=tracking,
        inputs=[frame_slider, video, mask_state],
        outputs=[track_video],
    )


iface.launch(share=True)
