#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_test 评估：用 Qwen 对 custom_annotations.json 做目标像素预测，与标注对比，判断模型推理能力。
数据格式：image_path, instruction, target_pixel: [x, y]
复用 annotation/evaluate_system2 的解析、指标与可视化。
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 项目根目录，便于引用 annotation
REAL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(REAL_TEST_DIR)
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image

# 从 annotation 评估脚本复用
from annotation.evaluate_system2 import (
    parse_pixel_prediction,
    compute_pixel_distance,
    compute_normalized_distance,
    compute_directional_accuracy,
    compute_all_metrics,
    print_metrics_table,
    save_results,
    visualize_prediction,
    plot_metrics_summary,
    SUCCESS_THRESHOLDS_PX,
    SUCCESS_THRESHOLDS_NORM,
)
from annotation.evaluate_system2 import System2Evaluator
# 云端评估器在 eval_s2 中定义，real_test/annotation 中无此实现
from eval_s2.evaluate_s2 import (
    CloudSystem2Evaluator,
    CLOUD_PLAN_ENDPOINT,
    CLOUD_REQUEST_TIMEOUT,
    DEFAULT_CLOUD_URL,
)


def load_custom_annotations(json_path, data_dir):
    """加载 custom_annotations.json，将 image_path 转为绝对路径。"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data_dir = os.path.abspath(data_dir)
    out = []
    for i, item in enumerate(raw):
        img_path = item.get("image_path", "")
        if not os.path.isabs(img_path):
            img_path = os.path.join(data_dir, img_path)
        out.append({
            "id": i + 1,
            "image_path": img_path,
            "instruction": item.get("instruction", ""),
            "target_pixel": item.get("target_pixel", [0, 0]),
        })
    return out


def run_evaluation(args):
    data_json = args.data_json
    data_dir = os.path.dirname(os.path.abspath(data_json))
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    print("[1] 加载标注:", data_json)
    annotations = load_custom_annotations(data_json, data_dir)
    if getattr(args, "max_samples", None) is not None:
        annotations = annotations[: args.max_samples]
        print(f"    共 {len(annotations)} 条（截断至 --max_samples）")
    else:
        print(f"    共 {len(annotations)} 条")

    print("[2] 加载模型 / 云端")
    if getattr(args, "use_cloud", False):
        evaluator = CloudSystem2Evaluator(cloud_url=args.cloud_url, timeout=CLOUD_REQUEST_TIMEOUT)
    else:
        evaluator = System2Evaluator(model_name=args.model_name, load_in_4bit=args.load_in_4bit)

    results = []
    for idx, ann in enumerate(annotations):
        img_path = ann["image_path"]
        instruction = ann["instruction"]
        gt_x, gt_y = ann["target_pixel"][0], ann["target_pixel"][1]
        ann_id = ann["id"]

        try:
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size
        except Exception as e:
            results.append({
                "id": ann_id, "instruction": instruction,
                "gt_x": gt_x, "gt_y": gt_y, "img_w": 640, "img_h": 480,
                "pred_x": None, "pred_y": None, "parse_success": False,
                "raw_response": str(e), "dist_px": None, "dist_norm": None,
                "direction": {"same_quadrant": False, "angle_diff_deg": 180,
                             "pred_quadrant": "N/A", "gt_quadrant": "N/A"},
                "inference_time_s": 0,
            })
            print(f"  [{ann_id}] 图片加载失败: {e}")
            continue

        pred = evaluator.predict_pixel_goal(image, instruction, img_w, img_h)

        if pred["parse_success"]:
            px, py = pred["pred_x"], pred["pred_y"]
            dist_px = compute_pixel_distance(px, py, gt_x, gt_y)
            dist_norm = compute_normalized_distance(px, py, gt_x, gt_y, img_w, img_h)
            direction = compute_directional_accuracy(px, py, gt_x, gt_y, img_w, img_h)
        else:
            px, py = None, None
            dist_px = dist_norm = None
            direction = {"same_quadrant": False, "angle_diff_deg": 180,
                         "pred_quadrant": "N/A", "gt_quadrant": "N/A"}

        results.append({
            "id": ann_id,
            "scene_id": "custom",
            "image_path": img_path,
            "instruction": instruction,
            "gt_x": gt_x, "gt_y": gt_y,
            "img_w": img_w, "img_h": img_h,
            "pred_x": px, "pred_y": py,
            "parse_success": pred["parse_success"],
            "raw_response": pred["raw_response"],
            "dist_px": dist_px, "dist_norm": dist_norm,
            "direction": direction,
            "inference_time_s": pred["inference_time_s"],
        })

        status = f"✓ {dist_px:.0f}px" if dist_px is not None else "✗ 解析失败"
        print(f"  [{ann_id:3d}] GT({gt_x},{gt_y}) Pred({px},{py}) {status}")

        if not args.no_viz and (px is not None or not pred["parse_success"]):
            viz_path = os.path.join(viz_dir, f"pred_{ann_id:04d}.jpg")
            visualize_prediction(image, gt_x, gt_y, px, py, instruction, dist_px, ann_id, viz_path)

    print("[3] 汇总指标")
    metrics = compute_all_metrics(results)
    metrics["model_name"] = (args.cloud_url + CLOUD_PLAN_ENDPOINT) if getattr(args, "use_cloud", False) else args.model_name
    metrics["eval_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics["data_json"] = data_json

    print_metrics_table(metrics)
    if not args.no_viz:
        plot_metrics_summary(metrics, results, output_dir)
    save_results(results, metrics, output_dir)
    print(f"\n结果已保存: {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="real_test: Qwen 目标像素预测 vs 标注，评估推理能力")
    parser.add_argument("--data_json", type=str,
                        default=os.path.join(REAL_TEST_DIR, "data", "custom_annotations.json"),
                        help="custom_annotations.json 路径")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REAL_TEST_DIR, "eval_results"),
                        help="评估结果输出目录")
    _default_model = os.environ.get("S2_EVAL_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_name", type=str, default=_default_model, help="Qwen-VL 模型名或本地路径")
    parser.add_argument("--load_in_4bit", action="store_true", help="4bit 量化")
    parser.add_argument("--use_cloud", action="store_true", help="使用云端 /plan 推理")
    parser.add_argument("--cloud_url", type=str, default=DEFAULT_CLOUD_URL, help="云端 base URL")
    parser.add_argument("--no_viz", action="store_true", help="不生成可视化")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估条数（默认全部）")
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()
