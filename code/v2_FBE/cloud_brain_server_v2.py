# ==============================================================================
# 文件名: cloud_brain_server_v2.py
# 运行位置: 云端显卡服务器 (有 RTX 3090 的机器)
# 功能: 在 v1 基础上增加多目标序列导航支持
# 兼容: local_robot_simple_v2.py
# 改进: 支持 "First find X, then Y, finally Z" 格式的序列指令解析
# ==============================================================================

import os
import tempfile
# 必须在 import torch 之前设置，减轻长时间多请求时的显存碎片与 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
import re
import numpy as np
import json
import warnings
from flask import Flask, request, jsonify
from PIL import Image

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ============== 1. 配置 (保持你的设置) ==============
MODEL_PATH = "/root/autodl-tmp/.autodl/models/Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda"

# ============== 2. 全局加载模型 (启动服务时加载一次) ==============
print("🚀 [云端 v2] 正在加载大脑 (Qwen2.5-VL)...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("✅ [云端 v2] 模型加载完成，支持多目标序列导航...")
except Exception as e:
    print(f"❌ [云端 v2] 模型加载失败: {e}")
    exit(1)

# ============== 3. 序列指令解析 (多目标连续导航) ==============
def parse_instruction_sequence(instruction):
    """
    解析 "First find a chair, then find a plant, finally go to the door" 格式
    返回目标列表，如 ["chair", "plant", "door"]
    单目标指令直接返回 [target]
    """
    instruction = instruction.strip()
    if not instruction:
        return ["red backpack"]

    # 多目标模式：匹配 "find X" 或 "go to X" 模式，保留顺序
    targets = re.findall(
        r'(?:find|go\s+to)\s+(?:a\s+|an\s+|the\s+)?(\w+)',
        instruction,
        re.IGNORECASE
    )
    if targets:
        return [t.strip() for t in targets]

    # 单目标：可能是 "plant" 或 "red backpack" 等
    return [instruction]

# ============== 图片预处理：限制分辨率，减轻显存压力 ==============
MAX_IMAGE_SIZE = 512  # 长边最大像素，减少 KV Cache 占用（降到 512 进一步减轻 OOM）

def resize_image_if_needed(image_path):
    """若图片长边 > MAX_IMAGE_SIZE，则缩放并覆盖原文件，返回 (h, w)"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, img)
        return new_h, new_w
    return h, w

# ============== 4. 推理核心逻辑 ==============
def inference_logic(image_path, target_name, scan_mode=False, first_request=False):
    """
    scan_mode: 初始扫描模式，要求输出找到目标的可能性分数 (0.0-1.0)
    first_request: 是否为该目标的首次请求，首次请求输出推理原因，后续只输出坐标
    """
    # 限制分辨率，减少显存占用
    hw = resize_image_if_needed(image_path)
    if hw is None:
        return None, "无法读取图片", 0.0, None
    h_real, w_real = hw

    if scan_mode:
        prompt_text = (
            f"The image size is {w_real} x {h_real} pixels (width x height). "
            f"For the {target_name}: If it is clearly visible, output PIXEL: (u, v) CONFIDENCE: 1.0. "
            f"If NOT visible, infer the most likely direction and output INFER: (u, v) CONFIDENCE: X "
            f"where X is 0.0-1.0 (how likely the {target_name} is in that direction, 0=unlikely, 1=very likely). "
            f"Use integer pixels. Example: INFER: (320, 240) CONFIDENCE: 0.7"
        )
    elif first_request:
        # 首次请求：输出推理原因 + 坐标
        prompt_text = (
            f"The image size is {w_real} x {h_real} pixels (width x height). "
            f"Find the target: \"{target_name}\". "
            f"First, briefly explain your reasoning (1-2 sentences about what you see and why you chose the direction). "
            f"Then output in this format:\n"
            f"REASON: <your brief reasoning>\n"
            f"STATUS: Target Locked\nPIXEL: (u, v)\n"
            f"OR\n"
            f"REASON: <your brief reasoning>\n"
            f"STATUS: Inferred\nPIXEL: (u, v)\n"
            f"- Use 'Target Locked' ONLY when you CLEARLY SEE the EXACT target \"{target_name}\" in the image.\n"
            f"- Use 'Inferred' when the target is NOT visible. (u,v) = direction to search.\n"
            f"Use integer pixel coordinates."
        )
    else:
        # 后续请求：只输出坐标，节省 token
        prompt_text = (
            f"Image: {w_real}x{h_real}. Target: \"{target_name}\". "
            f"Output ONLY:\n"
            f"STATUS: Target Locked\nPIXEL: (u, v)\n"
            f"OR\n"
            f"STATUS: Inferred\nPIXEL: (u, v)\n"
            f"No explanation needed. Integer pixels only."
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
    # 释放中间大张量
    del image_inputs, video_inputs

    print(f"🧠 [云端] 思考中: 寻找 {target_name}...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,  # 推理必要，但推理后立即清理
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # 彻底释放所有大张量和 KV Cache
    del inputs, generated_ids, generated_ids_trimmed
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"🤖 [云端] 原始输出: {output_text}")

    # 提取推理原因（仅首次请求有）
    reason = None
    reason_match = re.search(r'REASON:\s*(.+?)(?=STATUS:|PIXEL:|$)', output_text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
        if reason:
            print(f"💭 [云端] 推理原因: {reason}")

    confidence = 0.0
    conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', output_text, re.IGNORECASE)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            pass

    match_coords = re.search(r'(?:PIXEL|INFER):\s*\((\d+),\s*(\d+)\)', output_text, re.IGNORECASE)
    if match_coords:
        u, v = int(match_coords.group(1)), int(match_coords.group(2))
        u = max(0, min(u, w_real - 1))
        v = max(0, min(v, h_real - 1))
        if re.search(r'STATUS:\s*Target\s*Locked', output_text, re.IGNORECASE):
            return (u, v), "Success", 1.0, reason
        if re.search(r'STATUS:\s*Inferred', output_text, re.IGNORECASE):
            return (u, v), "Inferred", confidence, reason
        if re.search(r'PIXEL:\s*\(\d+', output_text, re.IGNORECASE):
            return (u, v), "Success", 1.0, reason
        if re.search(r'INFER:\s*\(\d+', output_text, re.IGNORECASE):
            return (u, v), "Inferred", confidence, reason
        return (u, v), "Inferred", confidence, reason

    nums = re.findall(r'\d+', output_text)
    if len(nums) >= 2:
        u = max(0, min(int(nums[0]), w_real - 1))
        v = max(0, min(int(nums[1]), h_real - 1))
        return (u, v), "Inferred", 0.5, reason

    return None, "SEARCHING", 0.0, None


# ============== 5. 到达验证 ==============
def verify_logic(image_path, target_name):
    """模型根据第一视角推理：是否已到达精确目标（严格，避免红桌子当红衣服）"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return False
    prompt_text = (
        f"This is a first-person view from a robot. The robot was asked to find: \"{target_name}\". "
        f"Is the robot now at/near the EXACT target \"{target_name}\"? "
        f"Answer YES only if you see the exact object (e.g. for 'red clothes' you must see clothes, not red table or red chair). "
        f"If you see a different object that only partially matches, answer NO. "
        f"Answer only YES or NO."
    )
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    # 限制分辨率
    resize_image_if_needed(image_path)

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(DEVICE)
    del image_inputs, video_inputs
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    result = "YES" in output_text.upper()
    del inputs, generated_ids, generated_ids_trimmed
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


# ============== 6. API 路由 ==============

@app.route('/parse_sequence', methods=['POST'])
def parse_sequence():
    """解析多目标序列指令，返回目标列表"""
    instruction = request.form.get('instruction', '')
    targets = parse_instruction_sequence(instruction)
    return jsonify({"targets": targets, "count": len(targets)})


@app.route('/plan', methods=['POST'])
def plan():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    # 每次请求前强制 GC + 清空显存，减轻连续推理时的碎片与 OOM
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    file = request.files['image']
    instruction = request.form.get('instruction', 'red backpack')
    target_index = request.form.get('target_index', None)
    target_list_json = request.form.get('target_list', None)
    verify_mode = request.form.get('verify', '0') == '1'

    # 多目标序列模式：使用 target_list[target_index]
    if target_list_json and target_index is not None:
        try:
            target_list = json.loads(target_list_json)
            idx = int(target_index)
            if 0 <= idx < len(target_list):
                instruction = target_list[idx]
        except (json.JSONDecodeError, ValueError):
            pass

    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        file.save(temp_path)
        # 用 PIL 校验，避免 qwen_vl_utils 里 Image.open 报 UnidentifiedImageError
        try:
            with Image.open(temp_path) as im:
                im.load()
        except Exception as e:
            return jsonify({"error": "invalid image", "detail": str(e)}), 400

        if verify_mode:
            ok = verify_logic(temp_path, instruction)
            return jsonify({"status": "success" if ok else "fail", "message": "verified" if ok else "not_verified"})

        scan_mode = request.form.get('scan', '0') == '1'
        # first_request=1 表示该目标的首次请求，输出推理原因；后续只输出坐标
        first_request = request.form.get('first_request', '0') == '1'
        coords, msg, confidence, reason = inference_logic(temp_path, instruction, scan_mode=scan_mode, first_request=first_request)

        if coords:
            response_data = {
                "u": coords[0],
                "v": coords[1],
                "message": msg,
                "confidence": confidence
            }
            if msg == "Inferred":
                response_data["status"] = "fail"
            else:
                response_data["status"] = "success"
            # 首次请求时返回推理原因
            if reason:
                response_data["reason"] = reason
            return jsonify(response_data)
        return jsonify({"status": "fail", "message": msg, "confidence": 0.0})
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
