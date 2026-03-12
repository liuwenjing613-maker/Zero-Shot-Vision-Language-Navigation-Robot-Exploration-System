# -*- coding: utf-8 -*-
import cv2
import os
import json
import numpy as np

# ==============================================================================
# 1. 参数配置区 (请根据你的实际情况修改)
# ==============================================================================
# 你录制的视频文件路径
VIDEO_FILE_PATH = '/home/abc/InternVLA/real_test_crop.mp4'

# 视频帧提取后，图片的保存目录
IMAGE_OUTPUT_DIR = 'data/custom_video_frames'

# 最终生成的标注 JSON 文件路径
JSON_OUTPUT_PATH = 'data/custom_annotations.json'

# 帧提取采样率：每隔多少帧保存一张图片。
# 经验值：如果你的视频是 30fps，设置为 15 意味着每秒提取n 2 张图，模拟论文中 2Hz 的 System 2
FRAME_SKIP = 50

# ==============================================================================
# 2. 功能函数区 (通常无需修改)
# ==============================================================================

def extract_frames_from_video(video_path, output_dir, frame_skip):
    """
    功能：从视频文件中提取帧并保存为图片。
    """
    print(f"--- 开始从视频 '{video_path}' 提取帧 ---")
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件 '{video_path}' 不存在！")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        # 如果视频读取完毕
        if not ret:
            break

        # 根据 FRAME_SKIP 采样率判断是否保存当前帧
        if frame_count % frame_skip == 0:
            image_filename = f"frame_{saved_count:05d}.jpg"
            image_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(image_path, frame)
            saved_count += 1
        
        frame_count += 1

    cap.release()
    print(f"--- 帧提取完成 ---")
    print(f"总共处理了 {frame_count} 帧，成功保存了 {saved_count} 张图片到 '{output_dir}' 目录。")


# --- 标注功能所需的全局变量 ---
annotations_data = []
current_point = None
current_image_display = None
window_name = "Annotation Tool - Click your target"

def mouse_click_event(event, x, y, flags, params):
    """
    OpenCV 鼠标点击事件的回调函数
    """
    global current_point, current_image_display
    if event == cv2.EVENT_LBUTTONDOWN:
        current_point = [x, y]
        # 在图像上画一个红圈和中心点，方便确认
        img_copy = params['original_image'].copy()
        cv2.circle(img_copy, (x, y), 10, (0, 0, 255), 2) # 画一个半径为10的红圈
        cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1) # 画一个实心红点
        current_image_display = img_copy
        cv2.imshow(window_name, current_image_display)


def save_annotations_to_json(data, json_path):
    """
    将标注数据保存到 JSON 文件
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\n标注数据已成功保存到: {json_path}")


def annotate_frames(image_dir, json_path):
    """
    功能：遍历图片，进行人工点击标注。
    """
    global annotations_data, current_point, current_image_display
    
    print("\n--- 开始人工标注 ---")
    print("操作指南:")
    print(" - 鼠标左键: 在图片上选择目标点")
    print(" - 按 'n' 键: 确认当前标注，并进入下一张图片")
    print(" - 按 'b' 键: 返回上一张图片，重新标注")
    print(" - 按 'q' 键: 退出程序并保存所有已完成的标注")
    print("--------------------")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    # 检查是否可以从已有的 JSON 文件恢复进度
    start_index = 0
    if os.path.exists(json_path):
        print(f"检测到已存在的标注文件 '{json_path}'，将加载并从上次中断的地方继续。")
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)
        
        if annotations_data:
            last_annotated_file = os.path.basename(annotations_data[-1]['image_path'])
            if last_annotated_file in image_files:
                start_index = image_files.index(last_annotated_file) + 1

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 窗口可调整大小

    i = start_index
    while i < len(image_files):
        image_name = image_files[i]
        # 使用相对路径，方便迁移
        relative_image_path = os.path.join(os.path.basename(image_dir), image_name)
        
        full_image_path = os.path.join(image_dir, image_name)
        original_image = cv2.imread(full_image_path)
        
        if original_image is None:
            print(f"警告: 无法读取图片 {full_image_path}，已跳过。")
            i += 1
            continue

        current_image_display = original_image.copy()
        current_point = None # 重置当前点

        print(f"\n正在标注第 {i + 1}/{len(image_files)} 张图片: {image_name}")
        
        # 将原图传入回调函数，方便重绘
        cv2.setMouseCallback(window_name, mouse_click_event, {'original_image': original_image})
        cv2.imshow(window_name, current_image_display)

        while True:
            key = cv2.waitKey(1) & 0xFF

            # 按 'n' -> 下一张
            if key == ord('n'):
                if current_point is None:
                    print("错误：请先点击一个目标点再按 'n'！")
                else:
                    annotation = {
                        "image_path": relative_image_path.replace("\\", "/"), # 统一用 / 分隔符
                        "instruction": "TODO: 请在这里填写导航指令", # 预留指令字段
                        "target_pixel": current_point
                    }
                    # 如果是重新标注，则更新，否则添加
                    found = False
                    for idx, item in enumerate(annotations_data):
                        if item['image_path'] == annotation['image_path']:
                            annotations_data[idx] = annotation
                            found = True
                            break
                    if not found:
                         annotations_data.append(annotation)

                    print(f"  > 已记录目标点: {current_point}")
                    i += 1
                    break
            
            # 按 'b' -> 上一张
            elif key == ord('b'):
                if i > 0:
                    i -= 1
                    # 如果是从已有数据集中后退，需要先移除最后一个
                    if annotations_data and os.path.basename(annotations_data[-1]['image_path']) == image_files[i+1]:
                        annotations_data.pop()
                    print("  < 返回上一张")
                    break
                else:
                    print("已经是第一张图片了！")

            # 按 'q' -> 退出并保存
            elif key == ord('q'):
                save_annotations_to_json(annotations_data, json_path)
                cv2.destroyAllWindows()
                return

    save_annotations_to_json(annotations_data, json_path)
    cv2.destroyAllWindows()
    print("\n--- 所有图片已标注完毕！ ---")


# ==============================================================================
# 3. 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # 第一步：从视频中提取帧
    extract_frames_from_video(VIDEO_FILE_PATH, IMAGE_OUTPUT_DIR, FRAME_SKIP)
    
    # 第二步：对提取的帧进行人工标注
    annotate_frames(IMAGE_OUTPUT_DIR, JSON_OUTPUT_PATH)