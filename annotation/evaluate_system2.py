#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
System2 推理能力评估脚本 —— 像素目标预测（Pixel Goal Prediction）
=============================================================================
功能:
  1. 加载本地标注的 data.json（上传自本地标注工具）
  2. 使用部署的VLM（Qwen-VL-2.5或其他）对每张图片进行System2推理
     - 输入: 第一视角图片 + 导航指令
     - 输出: 预测的目标像素坐标 (x, y)
  3. 与人工标注的Ground Truth进行对比，计算评估指标
  4. 所有中间结果保存为JSON/CSV/可视化图片

评估指标（参照InternVLA-N1 System2评估）:
  - 平均像素距离误差 (Mean Pixel Distance Error, MPDE)
  - 归一化像素距离误差 (Normalized PDE)
  - 成功率@k像素 (SR@50, SR@100, SR@200)
  - 归一化成功率 (NSR@0.1, NSR@0.2)
  - 方向准确率 (Directional Accuracy) - 预测方向与GT方向一致

使用方法:
  # 使用Qwen-VL-2.5 7B（推荐，与论文一致）
  python evaluate_system2.py \
      --data_json data.json \
      --model_name Qwen/Qwen2.5-VL-7B-Instruct \
      --output_dir ./eval_results

  # 使用较小模型（3090显存不够时）
  python evaluate_system2.py \
      --data_json data.json \
      --model_name Qwen/Qwen2.5-VL-3B-Instruct \
      --output_dir ./eval_results

依赖安装（RTX 3090服务器）:
  pip install transformers accelerate torch pillow numpy pandas
  pip install qwen-vl-utils  # Qwen-VL专用工具
  pip install matplotlib seaborn  # 可视化

注意：
  - Qwen2.5-VL-7B约需16~20GB显存，RTX 3090(24GB)可运行
  - 若显存不足，可使用4bit量化: --load_in_4bit
=============================================================================
"""

import os
import json
import argparse
import re
import time
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import csv

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")  # 非交互模式，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# 全局配置
# ============================================================

# 成功判定阈值（像素距离）
SUCCESS_THRESHOLDS_PX = [50, 100, 200]

# 成功判定阈值（归一化距离，相对于图片对角线长度）
SUCCESS_THRESHOLDS_NORM = [0.05, 0.10, 0.20]

# 推理超时（秒）
INFERENCE_TIMEOUT = 60

# 可视化保存分辨率
VIZ_DPI = 100


# ============================================================
# System2 推理 Prompt 设计
# ============================================================

SYSTEM_PROMPT = """You are a robot navigation assistant. Your task is to analyze a first-person view image and a navigation instruction, then identify the target navigation waypoint.

Output ONLY the pixel coordinates of the farthest visible navigation waypoint that aligns with the instruction. The waypoint should be:
- Visible in the current image
- Aligned with the described navigation direction
- As far as possible while still within the image

Output format: <point>x,y</point> where x is horizontal (0=left, image_width=right) and y is vertical (0=top, image_height=bottom).
Do NOT output anything else."""

# 用于不同VLM的prompt变体
PROMPT_TEMPLATES = {
    "qwen_vl": (
        "Navigation instruction: {instruction}\n\n"
        "Based on this first-person view image and the instruction above, "
        "identify the pixel coordinate of the navigation waypoint "
        "(where the robot should head towards). "
        "The waypoint should be the farthest point in the image that aligns with the instruction.\n\n"
        "Output ONLY in this exact format: <point>x,y</point>"
    ),
    "llava": (
        "You are a robot. Navigation instruction: {instruction}\n"
        "Where should the robot navigate to? "
        "Output the pixel coordinates as <point>x,y</point>"
    ),
    "generic": (
        "Instruction: {instruction}\n"
        "Output the target pixel as <point>x,y</point>"
    )
}


# ============================================================
# 坐标解析工具
# ============================================================

def parse_pixel_prediction(response_text: str, img_w: int, img_h: int) -> Optional[Tuple[int, int]]:
    """
    从模型输出文本中解析像素坐标预测。
    
    支持多种输出格式:
        1. <point>x,y</point>  ← 标准格式（我们要求的）
        2. (x, y) 或 [x, y]   ← 备用格式
        3. x=123, y=456        ← 键值格式
        4. Qwen-VL的<ref>坐标格式
    
    参数:
        response_text: 模型输出的原始文本
        img_w: 图片宽度（用于验证坐标合理性）
        img_h: 图片高度
    
    返回:
        (x, y) 整数元组，若解析失败返回 None
    """
    if not response_text:
        return None

    # 格式1: <point>x,y</point>
    pattern1 = r"<point>\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*</point>"
    m = re.search(pattern1, response_text)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        return _clamp_coords(x, y, img_w, img_h)

    # 格式2: Qwen-VL的归一化坐标格式 <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    # Qwen-VL有时输出归一化到0-1000的坐标
    pattern_qwen = r"\((\d+),\s*(\d+)\)"
    matches = re.findall(pattern_qwen, response_text)
    if matches:
        # 取第一个匹配
        x, y = float(matches[0][0]), float(matches[0][1])
        # 判断是否为归一化坐标（Qwen-VL用0-1000）
        if x <= 1000 and y <= 1000 and (x > 1 or y > 1):
            if x <= 1000 and y <= 1000:
                # 可能是归一化到1000的坐标，转换回像素
                x_px = int(x / 1000 * img_w)
                y_px = int(y / 1000 * img_h)
                return _clamp_coords(x_px, y_px, img_w, img_h)
        return _clamp_coords(x, y, img_w, img_h)

    # 格式3: (x, y) 或 [x, y]
    pattern3 = r"[\[\(]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\]\)]"
    m = re.search(pattern3, response_text)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        return _clamp_coords(x, y, img_w, img_h)

    # 格式4: x=123, y=456 或 x: 123, y: 456
    pattern4 = r"x[=:]\s*(\d+\.?\d*).*?y[=:]\s*(\d+\.?\d*)"
    m = re.search(pattern4, response_text, re.IGNORECASE)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        return _clamp_coords(x, y, img_w, img_h)

    # 格式5: 文中出现的两个数字（最后的备用方案）
    numbers = re.findall(r"\b(\d{2,4})\b", response_text)
    if len(numbers) >= 2:
        x, y = float(numbers[0]), float(numbers[1])
        if 0 <= x <= img_w * 1.1 and 0 <= y <= img_h * 1.1:
            return _clamp_coords(x, y, img_w, img_h)

    return None  # 解析失败


def _clamp_coords(x: float, y: float, img_w: int, img_h: int) -> Tuple[int, int]:
    """将坐标限制在图片范围内"""
    x = max(0, min(int(round(x)), img_w - 1))
    y = max(0, min(int(round(y)), img_h - 1))
    return (x, y)


# ============================================================
# 评估指标计算
# ============================================================

def compute_pixel_distance(pred_x: int, pred_y: int, gt_x: int, gt_y: int) -> float:
    """计算欧氏像素距离"""
    return math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)


def compute_normalized_distance(
    pred_x: int, pred_y: int,
    gt_x: int, gt_y: int,
    img_w: int, img_h: int
) -> float:
    """
    归一化像素距离（相对于图片对角线长度）。
    
    对角线作为归一化基准，使不同分辨率图片的指标可比。
    范围: [0, 1], 0=完全准确, 1=偏差等于对角线长度
    """
    diag = math.sqrt(img_w ** 2 + img_h ** 2)
    dist = compute_pixel_distance(pred_x, pred_y, gt_x, gt_y)
    return dist / diag if diag > 0 else 0.0


def compute_directional_accuracy(
    pred_x: int, pred_y: int,
    gt_x: int, gt_y: int,
    img_w: int, img_h: int
) -> dict:
    """
    方向准确率评估。
    
    将图片划分为以中心点为原点的4个象限，判断预测与GT是否在同一象限。
    这反映了模型对"前进方向"的大致理解是否正确。
    
    返回:
        {
            "same_quadrant": bool,       # 是否在同一象限
            "pred_quadrant": str,        # 预测所在象限
            "gt_quadrant": str,          # GT所在象限
            "angle_diff_deg": float      # 角度差（度）
        }
    """
    cx, cy = img_w / 2, img_h / 2

    def get_quadrant(x, y):
        if x >= cx and y < cy:
            return "top-right"
        elif x < cx and y < cy:
            return "top-left"
        elif x < cx and y >= cy:
            return "bottom-left"
        else:
            return "bottom-right"

    def get_angle(x, y):
        """相对于图片中心的角度（弧度）"""
        return math.atan2(y - cy, x - cx)

    pred_q = get_quadrant(pred_x, pred_y)
    gt_q = get_quadrant(gt_x, gt_y)

    pred_angle = get_angle(pred_x, pred_y)
    gt_angle = get_angle(gt_x, gt_y)
    angle_diff = abs(math.degrees(pred_angle - gt_angle))
    angle_diff = min(angle_diff, 360 - angle_diff)  # 取最小角度差

    return {
        "same_quadrant": pred_q == gt_q,
        "pred_quadrant": pred_q,
        "gt_quadrant": gt_q,
        "angle_diff_deg": round(angle_diff, 2)
    }


def compute_all_metrics(results: list) -> dict:
    """
    汇总计算所有评估指标。
    
    参数:
        results: 每条样本的预测结果列表（含dist_px, dist_norm等字段）
    
    返回:
        汇总指标字典
    """
    valid = [r for r in results if r.get("parse_success")]
    total = len(results)
    n_valid = len(valid)
    n_failed = total - n_valid

    if n_valid == 0:
        return {"error": "所有样本均解析失败", "total": total, "n_failed": n_failed}

    dist_px_list = [r["dist_px"] for r in valid]
    dist_norm_list = [r["dist_norm"] for r in valid]
    angle_diff_list = [r["direction"]["angle_diff_deg"] for r in valid]

    metrics = {
        # 基础统计
        "total_samples": total,
        "valid_predictions": n_valid,
        "failed_predictions": n_failed,
        "parse_success_rate": round(n_valid / total * 100, 2),

        # 像素距离误差
        "mean_pixel_dist": round(np.mean(dist_px_list), 2),
        "median_pixel_dist": round(np.median(dist_px_list), 2),
        "std_pixel_dist": round(np.std(dist_px_list), 2),
        "min_pixel_dist": round(np.min(dist_px_list), 2),
        "max_pixel_dist": round(np.max(dist_px_list), 2),

        # 归一化距离误差
        "mean_norm_dist": round(np.mean(dist_norm_list), 4),
        "median_norm_dist": round(np.median(dist_norm_list), 4),

        # 成功率@像素阈值
        "success_rates_px": {},
        "success_rates_norm": {},

        # 方向准确率
        "directional_accuracy": round(
            sum(r["direction"]["same_quadrant"] for r in valid) / n_valid * 100, 2
        ),
        "mean_angle_diff_deg": round(np.mean(angle_diff_list), 2),
    }

    # 计算各阈值成功率
    for thresh in SUCCESS_THRESHOLDS_PX:
        sr = sum(1 for d in dist_px_list if d <= thresh) / n_valid * 100
        metrics["success_rates_px"][f"SR@{thresh}px"] = round(sr, 2)

    for thresh in SUCCESS_THRESHOLDS_NORM:
        sr = sum(1 for d in dist_norm_list if d <= thresh) / n_valid * 100
        metrics["success_rates_norm"][f"NSR@{thresh}"] = round(sr, 2)

    return metrics


# ============================================================
# VLM模型加载与推理
# ============================================================

class System2Evaluator:
    """
    System2 评估器。
    
    封装VLM模型加载和推理，支持:
        - Qwen2.5-VL（推荐，与InternVLA-N1一致）
        - LLaVA系列
        - InternVL系列
    """

    def __init__(self, model_name: str, load_in_4bit: bool = False,
                 device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.processor = None
        self.model_type = self._detect_model_type(model_name)

        print(f"\n[模型加载] 正在加载 {model_name}")
        print(f"  模型类型: {self.model_type}")
        print(f"  4bit量化: {load_in_4bit}")
        self._load_model()

    def _detect_model_type(self, model_name: str) -> str:
        """根据模型名称判断模型类型"""
        name_lower = model_name.lower()
        if "qwen" in name_lower and "vl" in name_lower:
            return "qwen_vl"
        elif "llava" in name_lower:
            return "llava"
        elif "internvl" in name_lower or "internlm" in name_lower:
            return "internvl"
        else:
            return "generic"

    def _load_model(self):
        """加载模型和处理器"""
        from transformers import AutoProcessor, AutoModelForVision2Seq

        try:
            if self.model_type == "qwen_vl":
                self._load_qwen_vl()
            else:
                # 通用加载方式
                self._load_generic()

            print(f"  [✓] 模型加载完成")
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU显存占用: {mem:.2f} GB")

        except Exception as e:
            print(f"  [✗] 模型加载失败: {e}")
            raise

    def _load_qwen_vl(self):
        """加载Qwen2.5-VL模型"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        quantization_config = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28
        )

    def _load_generic(self):
        """通用VLM加载"""
        from transformers import AutoProcessor, AutoModelForVision2Seq

        quantization_config = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

    def predict_pixel_goal(
        self,
        image: Image.Image,
        instruction: str,
        img_w: int,
        img_h: int,
        max_new_tokens: int = 64
    ) -> dict:
        """
        使用System2（VLM）预测导航目标像素坐标。
        
        输入:
            image: PIL图片对象
            instruction: 导航指令文本
            img_w, img_h: 图片原始尺寸（用于坐标解析）
        
        返回:
            {
                "raw_response": str,      # 模型原始输出
                "pred_x": int or None,    # 预测像素x坐标
                "pred_y": int or None,    # 预测像素y坐标
                "parse_success": bool,    # 是否成功解析坐标
                "inference_time_s": float # 推理耗时（秒）
            }
        """
        start_time = time.time()

        try:
            prompt = PROMPT_TEMPLATES.get(self.model_type, PROMPT_TEMPLATES["generic"])
            prompt = prompt.format(instruction=instruction)

            if self.model_type == "qwen_vl":
                raw_response = self._infer_qwen_vl(image, prompt, max_new_tokens)
            else:
                raw_response = self._infer_generic(image, prompt, max_new_tokens)

            # 解析坐标
            coords = parse_pixel_prediction(raw_response, img_w, img_h)
            parse_success = coords is not None

            inference_time = time.time() - start_time

            return {
                "raw_response": raw_response,
                "pred_x": coords[0] if coords else None,
                "pred_y": coords[1] if coords else None,
                "parse_success": parse_success,
                "inference_time_s": round(inference_time, 3)
            }

        except Exception as e:
            return {
                "raw_response": f"ERROR: {str(e)}",
                "pred_x": None,
                "pred_y": None,
                "parse_success": False,
                "inference_time_s": round(time.time() - start_time, 3),
                "error": traceback.format_exc()
            }

    def _infer_qwen_vl(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Qwen2.5-VL推理"""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        # 只解码新生成的token
        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response.strip()

    def _infer_generic(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """通用VLM推理"""
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        response = self.processor.decode(output[0], skip_special_tokens=True)
        # 去除prompt本身，只保留生成部分
        if prompt in response:
            response = response.split(prompt)[-1]
        return response.strip()


# ============================================================
# 可视化
# ============================================================

def visualize_prediction(
    image: Image.Image,
    gt_x: int, gt_y: int,
    pred_x: Optional[int], pred_y: Optional[int],
    instruction: str,
    dist_px: Optional[float],
    record_id: int,
    output_path: str
):
    """
    生成单张预测可视化图片。
    
    在原图上绘制:
        - 绿色圆圈: Ground Truth位置
        - 红色叉号: 模型预测位置
        - 黄色虚线: GT与预测之间的连线
        - 文字信息: 指令、距离误差
    """
    vis_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis_img)

    w, h = vis_img.size

    # 尝试加载字体（服务器上可能没有字体文件，用默认字体）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # 绘制Ground Truth（绿色）
    r = 15
    draw.ellipse([gt_x - r, gt_y - r, gt_x + r, gt_y + r],
                 outline="#00ff00", width=3)
    draw.ellipse([gt_x - 5, gt_y - 5, gt_x + 5, gt_y + 5],
                 fill="#00ff00")

    # 绘制预测（红色）
    if pred_x is not None:
        draw.line([pred_x - r, pred_y, pred_x + r, pred_y], fill="#ff0000", width=3)
        draw.line([pred_x, pred_y - r, pred_x, pred_y + r], fill="#ff0000", width=3)
        draw.ellipse([pred_x - 5, pred_y - 5, pred_x + 5, pred_y + 5],
                     fill="#ff0000")
        # 连线
        draw.line([gt_x, gt_y, pred_x, pred_y], fill="#ffff00", width=2)

    # 文字信息（顶部）
    info_lines = [
        f"#{record_id} | Instruction: {instruction[:60]}...",
        f"GT: ({gt_x}, {gt_y})  |  Pred: ({pred_x}, {pred_y})"
        if pred_x else f"GT: ({gt_x}, {gt_y})  |  Pred: FAILED",
        f"Distance: {dist_px:.1f}px" if dist_px else "Distance: N/A"
    ]

    # 半透明背景
    text_h = 20
    for i, line in enumerate(info_lines):
        y_pos = i * text_h + 4
        draw.rectangle([0, y_pos, w, y_pos + text_h], fill=(0, 0, 0, 160))
        draw.text((4, y_pos + 2), line, fill="white", font=font_small)

    # 图例
    legend_x, legend_y = w - 140, h - 55
    draw.rectangle([legend_x, legend_y, w - 4, h - 4], fill=(0, 0, 0, 180))
    draw.ellipse([legend_x + 8, legend_y + 8, legend_x + 20, legend_y + 20],
                 outline="#00ff00", width=2)
    draw.text((legend_x + 24, legend_y + 8), "Ground Truth", fill="#00ff00", font=font_small)
    if pred_x:
        draw.line([legend_x + 8, legend_y + 32, legend_x + 20, legend_y + 32],
                  fill="#ff0000", width=2)
        draw.line([legend_x + 14, legend_y + 26, legend_x + 14, legend_y + 38],
                  fill="#ff0000", width=2)
        draw.text((legend_x + 24, legend_y + 28), "Prediction", fill="#ff0000", font=font_small)

    vis_img.save(output_path)


def plot_metrics_summary(metrics: dict, results: list, output_dir: str):
    """
    生成汇总统计图表，包含:
        1. 像素距离误差分布直方图
        2. 成功率柱状图
        3. 预测vs GT坐标散点图
        4. 推理时间分布
    """
    valid = [r for r in results if r.get("parse_success")]
    if not valid:
        print("[警告] 没有有效预测，跳过图表生成")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("System2 Pixel Goal Prediction - Evaluation Summary",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333366")

    # ---- 图1: 像素距离误差分布 ----
    ax1 = axes[0, 0]
    dist_px = [r["dist_px"] for r in valid]
    ax1.hist(dist_px, bins=20, color="#4cc9f0", edgecolor="#1a1a2e", alpha=0.8)
    ax1.axvline(np.mean(dist_px), color="#f72585", linewidth=2,
                label=f"Mean: {np.mean(dist_px):.1f}px")
    ax1.axvline(np.median(dist_px), color="#7209b7", linewidth=2,
                linestyle="--", label=f"Median: {np.median(dist_px):.1f}px")
    for thresh in SUCCESS_THRESHOLDS_PX:
        ax1.axvline(thresh, color="#ffd60a", linewidth=1, alpha=0.5,
                    linestyle=":", label=f"SR@{thresh}px")
    ax1.set_title("Pixel Distance Error Distribution")
    ax1.set_xlabel("Pixel Distance (px)")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    # ---- 图2: 成功率柱状图 ----
    ax2 = axes[0, 1]
    sr_labels = list(metrics.get("success_rates_px", {}).keys()) + \
                list(metrics.get("success_rates_norm", {}).keys())
    sr_values = list(metrics.get("success_rates_px", {}).values()) + \
                list(metrics.get("success_rates_norm", {}).values())

    colors = ["#4cc9f0"] * len(metrics.get("success_rates_px", {})) + \
             ["#f77f00"] * len(metrics.get("success_rates_norm", {}))

    bars = ax2.bar(range(len(sr_labels)), sr_values, color=colors, edgecolor="#1a1a2e")
    ax2.set_xticks(range(len(sr_labels)))
    ax2.set_xticklabels(sr_labels, rotation=30, ha="right", fontsize=9)
    ax2.set_title("Success Rates at Different Thresholds")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(0, 105)
    for bar, val in zip(bars, sr_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", color="white", fontsize=9)

    legend_patches = [
        mpatches.Patch(color="#4cc9f0", label="Pixel Threshold"),
        mpatches.Patch(color="#f77f00", label="Normalized Threshold")
    ]
    ax2.legend(handles=legend_patches, fontsize=8,
               facecolor="#1a1a2e", labelcolor="white")

    # ---- 图3: 预测坐标vs GT坐标散点图 ----
    ax3 = axes[1, 0]
    gt_xs = [r["gt_x"] / r["img_w"] for r in valid]
    gt_ys = [r["gt_y"] / r["img_h"] for r in valid]
    pred_xs = [r["pred_x"] / r["img_w"] for r in valid]
    pred_ys = [r["pred_y"] / r["img_h"] for r in valid]

    ax3.scatter(gt_xs, gt_ys, c="#a8ff78", s=40, alpha=0.7,
                marker="o", label="Ground Truth", zorder=3)
    ax3.scatter(pred_xs, pred_ys, c="#f72585", s=40, alpha=0.7,
                marker="x", label="Prediction", zorder=3)

    # 连线
    for gx, gy, px, py in zip(gt_xs, gt_ys, pred_xs, pred_ys):
        ax3.plot([gx, px], [gy, py], color="#ffd60a", alpha=0.2, linewidth=0.8)

    ax3.set_title("GT vs Predicted Pixel Locations (Normalized)")
    ax3.set_xlabel("x (normalized)")
    ax3.set_ylabel("y (normalized)")
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    ax3.invert_yaxis()  # 图片坐标y轴向下

    # ---- 图4: 推理时间分布 ----
    ax4 = axes[1, 1]
    infer_times = [r.get("inference_time_s", 0) for r in results]
    ax4.hist(infer_times, bins=20, color="#7209b7", edgecolor="#1a1a2e", alpha=0.8)
    ax4.axvline(np.mean(infer_times), color="#f72585", linewidth=2,
                label=f"Mean: {np.mean(infer_times):.2f}s")
    ax4.set_title("Inference Time Distribution")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "metrics_summary.png")
    plt.savefig(chart_path, dpi=VIZ_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [✓] 汇总图表已保存: {chart_path}")


# ============================================================
# 结果保存
# ============================================================

def save_results(results: list, metrics: dict, output_dir: str):
    """
    保存所有评估结果:
        1. results.json   - 每条样本详细结果
        2. metrics.json   - 汇总指标
        3. results.csv    - 方便用Excel分析
        4. failed.json    - 解析失败的样本（用于分析错误类型）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 详细结果JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  [✓] 详细结果: {results_path}")

    # 2. 汇总指标JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  [✓] 汇总指标: {metrics_path}")

    # 3. CSV格式
    csv_path = os.path.join(output_dir, "results.csv")
    fieldnames = [
        "id", "scene_id", "instruction",
        "gt_x", "gt_y", "pred_x", "pred_y",
        "parse_success", "dist_px", "dist_norm",
        "same_quadrant", "angle_diff_deg",
        "inference_time_s", "raw_response"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            if "direction" in r:
                row["same_quadrant"] = r["direction"].get("same_quadrant", "")
                row["angle_diff_deg"] = r["direction"].get("angle_diff_deg", "")
            writer.writerow(row)
    print(f"  [✓] CSV结果: {csv_path}")

    # 4. 失败案例
    failed = [r for r in results if not r.get("parse_success")]
    if failed:
        failed_path = os.path.join(output_dir, "failed.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"  [✓] 失败案例({len(failed)}条): {failed_path}")


def print_metrics_table(metrics: dict):
    """在终端打印格式化的评估结果表格"""
    print("\n" + "=" * 60)
    print("  System2 Pixel Goal Prediction 评估结果")
    print("=" * 60)
    print(f"  总样本数:          {metrics.get('total_samples', 0)}")
    print(f"  有效预测数:        {metrics.get('valid_predictions', 0)}")
    print(f"  解析成功率:        {metrics.get('parse_success_rate', 0):.1f}%")
    print("-" * 60)
    print("  像素距离误差:")
    print(f"    均值 (MPDE):     {metrics.get('mean_pixel_dist', 0):.2f} px")
    print(f"    中位数:          {metrics.get('median_pixel_dist', 0):.2f} px")
    print(f"    标准差:          {metrics.get('std_pixel_dist', 0):.2f} px")
    print("-" * 60)
    print("  成功率（像素阈值）:")
    for k, v in metrics.get("success_rates_px", {}).items():
        print(f"    {k:12s}:  {v:.1f}%")
    print("  成功率（归一化阈值）:")
    for k, v in metrics.get("success_rates_norm", {}).items():
        print(f"    {k:12s}:  {v:.1f}%")
    print("-" * 60)
    print("  方向准确率:")
    print(f"    同象限准确率:    {metrics.get('directional_accuracy', 0):.1f}%")
    print(f"    平均角度差:      {metrics.get('mean_angle_diff_deg', 0):.1f}°")
    print("=" * 60)


# ============================================================
# 主评估流程
# ============================================================

def run_evaluation(args):
    """主评估函数"""
    print("\n" + "=" * 60)
    print("  InternVLA-N1 System2 推理能力评估")
    print("  Pixel Goal Prediction Evaluation")
    print("=" * 60)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    print(f"  输出目录: {output_dir}")
    print(f"  可视化目录: {viz_dir}")

    # 加载标注数据
    print(f"\n[步骤1] 加载标注数据: {args.data_json}")
    with open(args.data_json, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"  共 {len(annotations)} 条标注")

    # 若指定评估子集
    if args.max_samples and args.max_samples < len(annotations):
        annotations = random.sample(annotations, args.max_samples) \
            if args.random_sample else annotations[:args.max_samples]
        print(f"  [子集] 使用 {len(annotations)} 条进行评估")

    # 加载模型
    print(f"\n[步骤2] 加载模型")
    evaluator = System2Evaluator(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit
    )

    # 检查是否存在断点续测结果
    checkpoint_path = os.path.join(output_dir, "checkpoint_results.json")
    results = []
    start_idx = 0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"  [断点续测] 从第 {start_idx} 条继续")

    # 逐条推理
    print(f"\n[步骤3] 开始推理评估 ({len(annotations)} 条样本)")
    print(f"  {'ID':>4}  {'场景':<15}  {'GT坐标':>12}  {'预测坐标':>12}  {'距离':>8}  {'状态'}")
    print("  " + "-" * 65)

    for idx, ann in enumerate(annotations[start_idx:], start=start_idx):
        ann_id = ann.get("id", idx + 1)
        scene_id = ann.get("scene_id", "unknown")
        img_path = ann.get("image_path", "")
        instruction = ann.get("instruction", "")
        gt = ann.get("pixel_goal", {})
        gt_x, gt_y = gt.get("x", 0), gt.get("y", 0)
        img_w = ann.get("image_width", 640)
        img_h = ann.get("image_height", 480)

        # 加载图片
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            result = {
                "id": ann_id, "scene_id": scene_id,
                "instruction": instruction,
                "gt_x": gt_x, "gt_y": gt_y,
                "img_w": img_w, "img_h": img_h,
                "pred_x": None, "pred_y": None,
                "parse_success": False,
                "raw_response": f"IMAGE_LOAD_ERROR: {e}",
                "dist_px": None, "dist_norm": None,
                "direction": {"same_quadrant": False, "angle_diff_deg": 180},
                "inference_time_s": 0
            }
            results.append(result)
            print(f"  {ann_id:>4}  {scene_id:<15}  [{gt_x:4d},{gt_y:4d}]  "
                  f"{'图片加载失败':>12}  {'':>8}  ✗")
            continue

        # 执行推理
        pred = evaluator.predict_pixel_goal(image, instruction, img_w, img_h)

        # 计算指标
        if pred["parse_success"]:
            px, py = pred["pred_x"], pred["pred_y"]
            dist_px = compute_pixel_distance(px, py, gt_x, gt_y)
            dist_norm = compute_normalized_distance(px, py, gt_x, gt_y, img_w, img_h)
            direction = compute_directional_accuracy(px, py, gt_x, gt_y, img_w, img_h)
            status = f"✓ {dist_px:.0f}px"
        else:
            px, py = None, None
            dist_px = None
            dist_norm = None
            direction = {"same_quadrant": False, "angle_diff_deg": 180,
                         "pred_quadrant": "N/A", "gt_quadrant": "N/A"}
            status = "✗ 解析失败"

        result = {
            "id": ann_id,
            "scene_id": scene_id,
            "image_path": img_path,
            "instruction": instruction,
            "gt_x": gt_x, "gt_y": gt_y,
            "img_w": img_w, "img_h": img_h,
            "pred_x": px, "pred_y": py,
            "parse_success": pred["parse_success"],
            "raw_response": pred["raw_response"],
            "dist_px": dist_px,
            "dist_norm": dist_norm,
            "direction": direction,
            "inference_time_s": pred["inference_time_s"]
        }
        results.append(result)

        print(f"  {ann_id:>4}  {scene_id:<15}  "
              f"[{gt_x:4d},{gt_y:4d}]  "
              f"[{str(px):>4},{str(py):>4}]  "
              f"{dist_px if dist_px else '---':>8}  "
              f"{status}")

        # 生成可视化
        if not args.no_viz:
            viz_path = os.path.join(viz_dir, f"pred_{ann_id:04d}.jpg")
            visualize_prediction(
                image, gt_x, gt_y, px, py,
                instruction, dist_px, ann_id, viz_path
            )

        # 每10条保存一次检查点
        if (idx + 1) % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)
            print(f"  [检查点] 已保存 {len(results)} 条结果")

    # 计算最终指标
    print("\n[步骤4] 计算汇总指标...")
    metrics = compute_all_metrics(results)
    metrics["model_name"] = args.model_name
    metrics["eval_timestamp"] = timestamp
    metrics["data_json"] = args.data_json

    # 打印指标
    print_metrics_table(metrics)

    # 生成图表
    if not args.no_viz:
        print("\n[步骤5] 生成可视化图表...")
        plot_metrics_summary(metrics, results, output_dir)

    # 保存所有结果
    print("\n[步骤6] 保存评估结果...")
    save_results(results, metrics, output_dir)

    # 清理检查点
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\n[完成] 评估完成！结果保存在: {output_dir}")
    return metrics


# ============================================================
# 入口
# ============================================================

def main():
    import random as _random
    global random
    random = _random

    parser = argparse.ArgumentParser(
        description="System2 像素目标预测评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用Qwen2.5-VL-7B评估（推荐）
  python evaluate_system2.py \\
      --data_json data.json \\
      --model_name Qwen/Qwen2.5-VL-7B-Instruct \\
      --output_dir ./eval_results

  # 显存不足时使用4bit量化
  python evaluate_system2.py \\
      --data_json data.json \\
      --model_name Qwen/Qwen2.5-VL-7B-Instruct \\
      --output_dir ./eval_results \\
      --load_in_4bit

  # 只评估前20条（快速测试）
  python evaluate_system2.py \\
      --data_json data.json \\
      --model_name Qwen/Qwen2.5-VL-3B-Instruct \\
      --output_dir ./eval_results \\
      --max_samples 20
        """
    )
    parser.add_argument("--data_json", type=str, required=True,
                        help="标注数据JSON文件路径")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HuggingFace模型名称或本地路径")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="评估结果输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数（默认全部）")
    parser.add_argument("--random_sample", action="store_true",
                        help="随机采样子集（配合--max_samples使用）")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="使用4bit量化加载模型（节省显存，适合<24GB）")
    parser.add_argument("--no_viz", action="store_true",
                        help="不生成可视化图片（加快速度）")
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()