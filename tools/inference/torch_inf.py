"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import sys
import os
import cv2  # Added for video processing
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig
from engine.extre_module.utils import increment_path

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
CLASS_NAME = None # Visdrone
COLOR_LIST = [
    (255, 0, 0),         # 红色 (person)
    (0, 255, 0),         # 绿色 (car)
    (0, 0, 255),         # 蓝色 (bike)
    (255, 165, 0),       # 橙色 (motorcycle)
    (255, 255, 0),       # 黄色 (truck)
    (0, 255, 255),       # 青色 (bus)
    (255, 0, 255),       # 品红 (train)
    (255, 255, 255),     # 白色 (airplane)
    (128, 0, 0),         # 棕色 (dog)
    (0, 128, 0),         # 深绿色 (cat)
    (0, 0, 128),         # 深蓝色 (horse)
    (128, 128, 0),       # 橄榄色 (sheep)
    (0, 128, 128),       # 蓝绿色 (cow)
    (128, 0, 128),       # 紫色 (elephant)
    (192, 192, 192),     # 银色 (giraffe)
    (255, 99, 71),       # 番茄色 (zebra)
    (0, 255, 127),       # 春绿色 (monkey)
    (255, 105, 180),     # 深粉色 (bird)
    (70, 130, 180),      # 钢蓝色 (fish)
]

def get_color_by_class(class_id):
    # 根据类别的索引返回固定颜色
    return COLOR_LIST[class_id % len(COLOR_LIST)]  # 确保索引不越界

# 获取动态调整的字体
def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)  # 你可以指定字体路径
    except IOError:
        return ImageFont.load_default()  # 如果加载失败，使用默认字体

# font_size_factor 越大 绘制字体越大 反之亦言
# box_thickness_factor 越大 绘制框越大 反之亦言
def draw(images, labels, boxes, scores, thrh=0.4, font_size_factor=0.05, box_thickness_factor=0.005, class_name=None):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        w, h = im.size  # 获取图像宽高

        for j, b in enumerate(box):
            lab_id = int(lab[j].item())
            color = get_color_by_class(lab_id)

            # 计算框的大小和字体的大小
            box_width = b[2] - b[0]
            box_height = b[3] - b[1]
            font_size = max(int(min(box_width, box_height) * font_size_factor), 12)  # 最小字体大小为 12
            font = get_font(font_size)

            # 绘制矩形框
            box_thickness = max(int(min(w, h) * box_thickness_factor), 2)  # 框的最小厚度为2
            draw.rectangle(list(b), outline=color, width=box_thickness)

            # 绘制类别名称和分数
            text = f"{class_name[lab_id] if class_name else lab_id} {round(scrs[j].item(), 2)}"
            
            # 使用 textbbox 获取文本的宽度和高度
            text_bbox = draw.textbbox((b[0], b[1]), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]  # 文本宽度
            text_height = text_bbox[3] - text_bbox[1]  # 文本高度

            text_x = b[0]  # 文本的起始 x 坐标
            text_y = b[1] - text_height - 5  # 文本在框的上方，预留间距

            # 确保文本在图像内
            if text_x + text_width > w:
                text_x = w - text_width
            if text_y < 0:
                text_y = b[1] + 5  # 如果文本超出边界，放置到框的下方

            draw.text((text_x, text_y), text=text, fill=color, font=font)

    return im

def process_image(model, device, file_path, output_path, thrh):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    im_pil = draw([im_pil], labels, boxes, scores, thrh=thrh, class_name=CLASS_NAME)
    im_pil.save(output_path / os.path.basename(file_path))

def process_video(model, device, file_path, output_path, thrh):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path / os.path.basename(file_path), fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    if cap.isOpened():
        for _ in tqdm.tqdm(range(total_frames), desc='Processing video frames...'):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to PIL image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            w, h = frame_pil.size
            orig_size = torch.tensor([[w, h]]).to(device)

            im_data = transforms(frame_pil).unsqueeze(0).to(device)

            output = model(im_data, orig_size)
            labels, boxes, scores = output

            # Draw detections on the frame
            draw([frame_pil], labels, boxes, scores, thrh=thrh, class_name=CLASS_NAME)

            # Convert back to OpenCV image
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            # Write the frame
            out.write(frame)

    cap.release()
    out.release()


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    output_path = increment_path(args.output)
    print(RED  + f"output_dir:{str(output_path)}" + RESET)
    output_path.mkdir(parents=True, exist_ok=True)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if checkpoint.get('name', None) != None:
            CLASS_NAME = checkpoint['name']
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.isdir(file_path):
        for file_name in tqdm.tqdm(os.listdir(file_path), desc=f'Process {file_path} folder'):
            if os.path.splitext(file_name)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                process_image(model, device, os.path.join(file_path, file_name), output_path, args.thrh)
            elif os.path.splitext(file_path)[-1].lower() in ['.mp4', '.avi', '.mov']:
                process_video(model, device, os.path.join(file_path, file_name), output_path, args.thrh)
    elif os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(model, device, file_path, output_path, args.thrh)
        print("Image processing complete.")
    elif os.path.splitext(file_path)[-1].lower() in ['.mp4', '.avi', '.mov']:
        # Process as video
        process_video(model, device, file_path, output_path, args.thrh)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default='inference_results/exp')
    parser.add_argument('-t', '--thrh', type=float, default=0.2)
    parser.add_argument('-d', '--device', type=str, default='0')
    args = parser.parse_args()
    main(args)
