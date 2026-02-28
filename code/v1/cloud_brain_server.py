# ==============================================================================
# 文件名: cloud_brain_server.py
# 运行位置: 云端显卡服务器 (有 RTX 3090 的机器)
# 功能: 加载 Qwen2.5-VL，提供 HTTP 接口接收图片，返回像素坐标
# 兼容: local_robot_system1.py (完整版) / local_robot_simple.py (极简版)
# ==============================================================================

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
import re
import numpy as np
import json
import warnings
from flask import Flask, request, jsonify
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ============== 1. 配置 (保持你的设置) ==============
MODEL_PATH = "/root/autodl-tmp/.autodl/models/Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda"

# ============== 2. 全局加载模型 (启动服务时加载一次) ==============
print("🚀 [云端] 正在加载大脑 (Qwen2.5-VL)...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("✅ [云端] 模型加载完成，等待指令...")
except Exception as e:
    print(f"❌ [云端] 模型加载失败: {e}")
    exit(1)

# ============== 3. 推理核心逻辑 ==============
def inference_logic(image_path, target_name, scan_mode=False):
    """
    scan_mode: 初始扫描模式，要求输出找到目标的可能性分数 (0.0-1.0)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, "无法读取图片", 0.0
    h_real, w_real = img_bgr.shape[:2]

    if scan_mode:
        # 扫描模式：输出像素坐标 + 可能性分数
        prompt_text = (
            f"The image size is {w_real} x {h_real} pixels (width x height). "
            f"For the {target_name}: If it is clearly visible, output PIXEL: (u, v) CONFIDENCE: 1.0. "
            f"If NOT visible, infer the most likely direction and output INFER: (u, v) CONFIDENCE: X "
            f"where X is 0.0-1.0 (how likely the {target_name} is in that direction, 0=unlikely, 1=very likely). "
            f"Use integer pixels. Example: INFER: (320, 240) CONFIDENCE: 0.7"
        )
    else:
        prompt_text = (
            f"The image size is {w_real} x {h_real} pixels (width x height). "
            f"Find the {target_name}. Output EXACTLY in this format:\n"
            f"STATUS: Target Locked\nPIXEL: (u, v)\n"
            f"OR\n"
            f"STATUS: Inferred\nPIXEL: (u, v)\n"
            f"- Use 'Target Locked' ONLY when you CLEARLY SEE the {target_name} in the image. "
            f"Be strict: if you are not sure, or only see similar objects, use Inferred.\n"
            f"- Use 'Inferred' when the {target_name} is NOT visible or you are uncertain. "
            f"(u,v) = direction to search (e.g. u=0 for left, u={w_real-1} for right).\n"
            f"Use integer pixel coordinates."
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    print(f"🧠 [云端] 思考中: 寻找 {target_name}...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"🤖 [云端] 原始输出: {output_text}")

    # 解析 CONFIDENCE（扫描模式）
    confidence = 0.0
    conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', output_text, re.IGNORECASE)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            pass

    # 解析：模型显式输出 STATUS + PIXEL，或兼容旧格式 PIXEL/INFER: (u,v)
    match_coords = re.search(r'(?:PIXEL|INFER):\s*\((\d+),\s*(\d+)\)', output_text, re.IGNORECASE)
    if match_coords:
        u, v = int(match_coords.group(1)), int(match_coords.group(2))
        u = max(0, min(u, w_real - 1))
        v = max(0, min(v, h_real - 1))
        # 优先用模型显式声明的 STATUS
        if re.search(r'STATUS:\s*Target\s*Locked', output_text, re.IGNORECASE):
            return (u, v), "Success", 1.0
        if re.search(r'STATUS:\s*Inferred', output_text, re.IGNORECASE):
            return (u, v), "Inferred", confidence
        # 兼容旧格式：PIXEL 即 Success，INFER 即 Inferred
        if re.search(r'PIXEL:\s*\(\d+', output_text, re.IGNORECASE):
            return (u, v), "Success", 1.0
        if re.search(r'INFER:\s*\(\d+', output_text, re.IGNORECASE):
            return (u, v), "Inferred", confidence
        # 无明确 STATUS 时保守处理：默认 Inferred，避免误判 Target Locked
        return (u, v), "Inferred", confidence

    nums = re.findall(r'\d+', output_text)
    if len(nums) >= 2:
        u = max(0, min(int(nums[0]), w_real - 1))
        v = max(0, min(int(nums[1]), h_real - 1))
        # 仅数字无格式时按 Inferred 处理，不当作 Target Locked
        return (u, v), "Inferred", 0.5

    return None, "SEARCHING", 0.0


# ============== 4. 到达验证：模型推理是否已到达目标 ==============
def verify_logic(image_path, target_name):
    """模型根据第一视角推理：是否已到达目标，一旦给出 YES 即判定成功"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return False
    prompt_text = (
        f"This is a first-person view from a robot navigating to find a {target_name}. "
        f"Has the robot reached the target {target_name}? Is the robot now at/near the {target_name}? "
        f"Answer only YES or NO."
    )
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    # 只要输出包含 YES 视为验证通过
    return "YES" in output_text.upper()


# ============== 5. API 路由 ==============
@app.route('/plan', methods=['POST'])
def plan():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400

    file = request.files['image']
    instruction = request.form.get('instruction', 'red backpack')
    verify_mode = request.form.get('verify', '0') == '1'

    temp_path = "temp_cloud_input.jpg"
    file.save(temp_path)

    if verify_mode:
        ok = verify_logic(temp_path, instruction)
        return jsonify({"status": "success" if ok else "fail", "message": "verified" if ok else "not_verified"})

    scan_mode = request.form.get('scan', '0') == '1'
    coords, msg, confidence = inference_logic(temp_path, instruction, scan_mode=scan_mode)

    if coords:
        if msg == "Inferred":
            return jsonify({
                "status": "fail",
                "u": coords[0],
                "v": coords[1],
                "message": "Inferred",
                "confidence": confidence
            })
        # msg == "Success" 或 "Success (parsed)"
        return jsonify({
            "status": "success",
            "u": coords[0],
            "v": coords[1],
            "message": msg,
            "confidence": confidence
        })
    return jsonify({"status": "fail", "message": msg, "confidence": 0.0})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
