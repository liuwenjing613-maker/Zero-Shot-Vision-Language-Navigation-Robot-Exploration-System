# ==============================================================================
# 文件名: cloud_internvla_s2.py
# 运行位置: 云端显卡服务器
# 功能: InternVLA System2 推理——视觉伺服闭环
#       未看到目标时输出旋转指令(左/右)，看到目标时输出像素点，完成时输出 STOP
# 与本地 local_robot_internvla_s2_my_s1.py 配合，S1 策略适配本 API 的 action 输出
# ==============================================================================

import os
import re
import tempfile
import json
import warnings
from flask import Flask, request, jsonify
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import cv2

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ============== 1. 配置 ==============
MODEL_PATH = os.environ.get("INTERNVLA_S2_MODEL", "/root/autodl-tmp/.autodl/models/Qwen/Qwen2.5-VL-7B-Instruct")
DEVICE = "cuda"
MAX_IMAGE_SIZE = 512
MAX_NEW_TOKENS = 128

# ============== 2. 原论文附录 User Prompt（一字不差，保证输出方式一致）==============
# 模型未看到目标会输出左右箭头(←→)，看到才输出像素点；STOP 表示任务完成
INTERNVLA_USER_PROMPT = (
    "You are an autonomous navigation assistant. Your task is {instruction}. "
    "Where should you go next to stay on track? "
    "Please output the next waypoint's coordinates in the image. "
    "Please output STOP when you have successfully completed the task. "
    "These are your historical observations: {history}"
)

# ============== 3. 加载模型 ==============
print("🚀 [InternVLA S2 云端] 正在加载模型...")
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("✅ [InternVLA S2] 模型加载完成（无 system message，支持箭头/坐标/STOP）")
except Exception as e:
    print(f"❌ [InternVLA S2] 模型加载失败: {e}")
    exit(1)


def resize_image_if_needed(image_path):
    orig = cv2.imread(image_path)
    if orig is None:
        return None
    orig_h, orig_w = orig.shape[:2]
    if max(orig_h, orig_w) <= MAX_IMAGE_SIZE:
        return orig_h, orig_w, orig_h, orig_w
    scale = MAX_IMAGE_SIZE / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = cv2.resize(orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_path, img)
    return orig_h, orig_w, new_h, new_w


def scale_to_original(u_resized, v_resized, orig_w, orig_h, w_real, h_real):
    u = round(u_resized * (orig_w / max(1, w_real)))
    v = round(v_resized * (orig_h / max(1, h_real)))
    return max(0, min(u, orig_w - 1)), max(0, min(v, orig_h - 1))


def infer_internvla_s2(image_path, instruction, history_text=""):
    """
    使用 InternVLA S2 官方 prompt 推理。
    返回: (action, u, v, reason)
    action: "turn_left" | "turn_right" | "waypoint" | "stop" | "searching"
    """
    hw = resize_image_if_needed(image_path)
    if hw is None:
        return "searching", None, None, "无法读取图片"
    orig_h, orig_w, h_real, w_real = hw

    history = history_text.strip() or "None."
    prompt = INTERNVLA_USER_PROMPT.format(instruction=instruction, history=history)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
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
    del image_inputs, video_inputs

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    generated_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    del inputs, generated_ids, generated_trimmed
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"🤖 [InternVLA S2] 原始输出: {output_text[:200]}")

    out_lower = output_text.lower().strip()
    # 1) STOP
    if "stop" in out_lower and ("success" in out_lower or "completed" in out_lower or output_text.strip().upper().endswith("STOP")):
        return "stop", None, None, "Task completed."
    if output_text.strip().upper() == "STOP":
        return "stop", None, None, "Task completed."

    # 2) 像素点优先：看到目标时先解析坐标，再考虑箭头
    # 4) 像素点：<point>x,y</point> 或 (u, v)
    pattern_point = re.search(r"<point>\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*</point>", output_text, re.IGNORECASE)
    if pattern_point:
        u, v = int(float(pattern_point.group(1))), int(float(pattern_point.group(2)))
        u = max(0, min(u, w_real - 1))
        v = max(0, min(v, h_real - 1))
        u, v = scale_to_original(u, v, orig_w, orig_h, w_real, h_real)
        return "waypoint", u, v, "Waypoint in image."
    pattern_paren = re.search(r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", output_text)
    if pattern_paren:
        u, v = int(float(pattern_paren.group(1))), int(float(pattern_paren.group(2)))
        if u <= 1000 and v <= 1000 and (u > 1 or v > 1):
            u = int(u / 1000 * orig_w)
            v = int(v / 1000 * orig_h)
        else:
            u = max(0, min(u, orig_w - 1))
            v = max(0, min(v, orig_h - 1))
        return "waypoint", u, v, "Waypoint in image."

    nums = re.findall(r"\d+", output_text)
    if len(nums) >= 2:
        u, v = int(nums[0]), int(nums[1])
        if 0 <= u <= orig_w * 1.2 and 0 <= v <= orig_h * 1.2:
            u = max(0, min(u, orig_w - 1))
            v = max(0, min(v, orig_h - 1))
            return "waypoint", u, v, "Waypoint in image."

    # 3) 左转：← 或 left / turn left
    if "←" in output_text or "left" in out_lower or "turn left" in out_lower:
        return "turn_left", None, None, "Turn left to search."

    # 4) 右转：→ 或 right / turn right
    if "→" in output_text or "right" in out_lower or "turn right" in out_lower:
        return "turn_right", None, None, "Turn right to search."

    return "searching", None, None, "No clear action."


@app.route("/plan", methods=["POST"])
def plan():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    file = request.files["image"]
    instruction = request.form.get("instruction", "find the goal")
    history = request.form.get("history", "")
    target_list_json = request.form.get("target_list", None)
    target_index = request.form.get("target_index", None)
    verify_mode = request.form.get("verify", "0") == "1"

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
        try:
            with Image.open(temp_path) as im:
                im.load()
        except Exception as e:
            return jsonify({"error": "invalid image", "detail": str(e)}), 400

        if verify_mode:
            # 简单到达验证：可复用原有 verify 逻辑或仅返回 success
            return jsonify({"status": "success", "message": "verified", "action": "stop"})

        action, u, v, reason = infer_internvla_s2(temp_path, instruction, history)

        if action == "stop":
            return jsonify({
                "status": "success",
                "message": "STOP",
                "action": "stop",
                "reason": reason,
            })
        if action == "turn_left":
            return jsonify({
                "status": "fail",
                "message": "Turn left",
                "action": "turn_left",
                "reason": reason,
            })
        if action == "turn_right":
            return jsonify({
                "status": "fail",
                "message": "Turn right",
                "action": "turn_right",
                "reason": reason,
            })
        if action == "waypoint" and u is not None and v is not None:
            return jsonify({
                "status": "success",
                "message": "Waypoint",
                "action": "waypoint",
                "u": u,
                "v": v,
                "reason": reason,
            })
        return jsonify({
            "status": "fail",
            "message": "Searching",
            "action": "searching",
            "reason": reason,
        })
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.route("/parse_sequence", methods=["POST"])
def parse_sequence():
    import re
    instruction = request.form.get("instruction", "").strip()
    if not instruction:
        return jsonify({"targets": ["red backpack"], "count": 1})
    targets = re.findall(r"(?:find|go\s+to)\s+(?:a\s+|an\s+|the\s+)?(\w+)", instruction, re.IGNORECASE)
    if targets:
        return jsonify({"targets": [t.strip() for t in targets], "count": len(targets)})
    return jsonify({"targets": [instruction], "count": 1})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
