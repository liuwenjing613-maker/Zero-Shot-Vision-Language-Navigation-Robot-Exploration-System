#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 v1 / v2 / v3 三个版本详细结果的指标、数据重合、成功/失败步数、失败原因等。
输出到 ppt/ 目录：version_analysis_report.md、version_metrics.csv、failure_breakdown.csv
"""

import os
import json
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR = os.path.join(REPO_ROOT, "val", "detailed_results")
EVAL_EPISODES_JSON = os.path.join(REPO_ROOT, "val", "eval_episodes.json")
FILES = {
    "v1": os.path.join(VAL_DIR, "v1_run0_20260305_173257.json"),
    "v2": os.path.join(VAL_DIR, "v2_run0_20260305_183202.json"),
    "v3": os.path.join(VAL_DIR, "v3_run0_20260308_144718.json"),
}
SUCCESS_DIST = 3.0
MAX_STEPS = 1200
# 失败原因分类
TIMEOUT_STEPS = 1100   # 步数 >= 此视为“步数用尽/超时”
NEAR_MISS_MAX = 5.0    # min_dist 在 [3, 5] 视为“接近未达”
STUCK_TRAJ_MAX = 2.0   # trajectory_length < 此且未成功视为“几乎未移动/卡住”


def load_all():
    data = {}
    for ver, path in FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            data[ver] = json.load(f)
    return data


def episode_key(ep):
    return (ep["scene_id"], ep["instruction"].strip().rstrip("."))


def classify_failure(ep):
    """对失败 episode 做简单分类（无 failure_reason 时用数值推断）"""
    if ep.get("success"):
        return "success"
    steps = ep.get("steps", 0)
    min_d = ep.get("min_dist")
    traj = ep.get("trajectory_length", 0) or 0
    if steps >= TIMEOUT_STEPS:
        return "timeout"
    if min_d is None:
        return "unknown"
    if traj < STUCK_TRAJ_MAX:
        return "stuck"
    if SUCCESS_DIST < min_d <= NEAR_MISS_MAX:
        return "near_miss"
    if min_d > NEAR_MISS_MAX:
        return "far"
    return "other"


def get_dataset_size():
    """完整测试集条数（eval_episodes.json）"""
    if os.path.isfile(EVAL_EPISODES_JSON):
        with open(EVAL_EPISODES_JSON, "r", encoding="utf-8") as f:
            return len(json.load(f))
    return None


def run():
    dataset_total = get_dataset_size()
    data = load_all()
    # 各版本 episode 按 key 索引
    by_key = {ver: {} for ver in data}
    for ver, d in data.items():
        for ep in d["episodes"]:
            k = episode_key(ep)
            by_key[ver][k] = ep

    all_keys = set()
    for v in by_key:
        all_keys |= set(by_key[v].keys())

    # ----- 数据重合 -----
    v1_k = set(by_key["v1"].keys())
    v2_k = set(by_key["v2"].keys())
    v3_k = set(by_key["v3"].keys())
    overlap_12 = v1_k & v2_k
    overlap_13 = v1_k & v3_k
    overlap_23 = v2_k & v3_k
    overlap_123 = v1_k & v2_k & v3_k

    # ----- 各版本指标与步数 -----
    rows_metrics = []
    failure_breakdown = []  # ver, failure_type, count, episode_ids / keys 示例

    for ver in ("v1", "v2", "v3"):
        d = data[ver]
        eps = d["episodes"]
        summary = d["summary"]
        analysis = d.get("analysis", {})

        n_total = len(eps)
        n_succ = summary["success_count"]
        n_fail = n_total - n_succ
        succ_eps = [e for e in eps if e.get("success")]
        fail_eps = [e for e in eps if not e.get("success")]

        steps_succ = [e["steps"] for e in succ_eps] if succ_eps else [0]
        steps_fail = [e["steps"] for e in fail_eps] if fail_eps else [0]

        rows_metrics.append({
            "version": ver,
            "total_episodes": n_total,
            "success_count": n_succ,
            "fail_count": n_fail,
            "SR": round(summary["SR"] * 100, 2),
            "OSR": round(summary["OSR"] * 100, 2),
            "SPL": round(summary["SPL"], 4),
            "NE": round(summary["NE"], 4),
            "avg_steps_success": round(sum(steps_succ) / len(steps_succ), 1) if steps_succ else 0,
            "avg_steps_failed": round(sum(steps_fail) / len(steps_fail), 1) if steps_fail else 0,
            "max_steps_success": max(steps_succ) if steps_succ else 0,
            "min_steps_success": min(steps_succ) if steps_succ else 0,
            "max_steps_failed": max(steps_fail) if steps_fail else 0,
            "min_steps_failed": min(steps_fail) if steps_fail else 0,
        })

        # 失败原因统计
        fail_types = defaultdict(list)
        for e in fail_eps:
            t = classify_failure(e)
            fail_types[t].append(e["episode_id"])
        for ft, ids in sorted(fail_types.items(), key=lambda x: -len(x[1])):
            failure_breakdown.append({
                "version": ver,
                "failure_type": ft,
                "count": len(ids),
                "example_ids": ids[:10],
            })

    # ----- 在共同 key 上的对比（v1∩v2∩v3） -----
    common_keys = sorted(overlap_123)
    same_key_compare = []
    for k in common_keys:
        scene_id, instr = k
        row = {"scene_id": scene_id, "instruction": instr[:50]}
        for ver in ("v1", "v2", "v3"):
            ep = by_key[ver].get(k)
            if ep:
                row[f"{ver}_success"] = ep.get("success", False)
                row[f"{ver}_steps"] = ep.get("steps", 0)
                row[f"{ver}_min_dist"] = round(ep.get("min_dist", 0), 2) if ep.get("min_dist") is not None else None
                row[f"{ver}_NE"] = round(ep.get("NE", 0), 2) if ep.get("NE") is not None else None
            else:
                row[f"{ver}_success"] = None
                row[f"{ver}_steps"] = None
                row[f"{ver}_min_dist"] = None
                row[f"{ver}_NE"] = None
        same_key_compare.append(row)

    # ----- 写报告 -----
    out_dir = os.path.join(REPO_ROOT, "ppt")
    os.makedirs(out_dir, exist_ok=True)

    md_path = os.path.join(out_dir, "version_analysis_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# v1 / v2 / v3 版本指标与详情分析\n\n")
        if dataset_total is not None:
            f.write("**完整测试集**：`val/eval_episodes.json` 共 **{}** 条 episode。\n\n".format(dataset_total))
            f.write("**为何报告里是 48/51/45？** 当前分析的是三份**历史运行结果文件**，每份里**当次运行实际写入的条数**分别为 v1=48、v2=51、v3=45（可能因运行中断或未跑满）。若要在 {} 条上公平对比，需用同一份任务重新跑齐三版本。\n\n".format(dataset_total))
        f.write("数据来源：\n")
        f.write("- v1: `val/detailed_results/v1_run0_20260305_173257.json`（文件中 48 条）\n")
        f.write("- v2: `val/detailed_results/v2_run0_20260305_183202.json`（文件中 51 条）\n")
        f.write("- v3: `val/detailed_results/v3_run0_20260308_144718.json`（文件中 45 条）\n\n")

        f.write("## 1. 数据重合情况\n\n")
        f.write("**说明**：下表中「结果文件内条数」指各 JSON 内实际记录条数，与测试集总数 53 可能不一致。\n\n")
        f.write("| 集合 | 数量 | 说明 |\n")
        f.write("|------|------|------|\n")
        if dataset_total is not None:
            f.write("| **测试集总数（eval_episodes.json）** | **{}** | 完整应为 {} 条 |\n".format(dataset_total, dataset_total))
        n1, n2, n3 = len(data["v1"]["episodes"]), len(data["v2"]["episodes"]), len(data["v3"]["episodes"])
        f.write("| v1 结果文件内条数 | {} | 该次运行实际写入 |\n".format(n1))
        f.write("| v2 结果文件内条数 | {} | 该次运行实际写入 |\n".format(n2))
        f.write("| v3 结果文件内条数 | {} | 该次运行实际写入 |\n".format(n3))
        f.write("| 仅 v1 有的任务 | {} | 只在 v1 出现 |\n".format(len(v1_k - v2_k - v3_k)))
        f.write("| 仅 v2 有的任务 | {} | 只在 v2 出现 |\n".format(len(v2_k - v1_k - v3_k)))
        f.write("| 仅 v3 有的任务 | {} | 只在 v3 出现 |\n".format(len(v3_k - v1_k - v2_k)))
        f.write("| v1 ∩ v2 | {} | 两版本共有 |\n".format(len(overlap_12)))
        f.write("| v1 ∩ v3 | {} | 两版本共有 |\n".format(len(overlap_13)))
        f.write("| v2 ∩ v3 | {} | 两版本共有 |\n".format(len(overlap_23)))
        f.write("| **v1 ∩ v2 ∩ v3** | **{}** | **三版本共有（可逐条对比）** |\n\n".format(len(overlap_123)))

        f.write("## 2. 各版本汇总指标\n\n")
        f.write("| 版本 | 总 episode 数 | 成功数 | 失败数 | SR(%) | OSR(%) | SPL | NE |\n")
        f.write("|------|---------------|--------|--------|-------|--------|-----|-----|\n")
        for r in rows_metrics:
            f.write(f"| {r['version']} | {r['total_episodes']} | {r['success_count']} | {r['fail_count']} | {r['SR']} | {r['OSR']} | {r['SPL']} | {r['NE']} |\n")
        f.write("\n")

        f.write("## 3. 成功 / 失败步数统计\n\n")
        f.write("| 版本 | 成功时平均步数 | 成功时最小/最大步数 | 失败时平均步数 | 失败时最小/最大步数 |\n")
        f.write("|------|----------------|----------------------|----------------|----------------------|\n")
        for r in rows_metrics:
            f.write(f"| {r['version']} | {r['avg_steps_success']} | {r['min_steps_success']} / {r['max_steps_success']} | {r['avg_steps_failed']} | {r['min_steps_failed']} / {r['max_steps_failed']} |\n")
        f.write("\n")

        f.write("## 4. 失败原因分类（推断）\n\n")
        f.write("基于步数、min_dist、trajectory_length 推断（无显式 failure_reason 字段）：\n")
        f.write("- **timeout**: 步数 ≥ {}（接近 max_steps）\n".format(TIMEOUT_STEPS))
        f.write("- **stuck**: 轨迹长度 < {} 米且未成功（几乎未移动/卡住）\n".format(STUCK_TRAJ_MAX))
        f.write("- **near_miss**: 最近距离在 (3, 5] 米（接近目标未到达）\n")
        f.write("- **far**: 最近距离 > 5 米（未找到目标/走错）\n\n")
        f.write("| 版本 | 失败类型 | 数量 | 示例 episode_id |\n")
        f.write("|------|----------|------|------------------|\n")
        for row in failure_breakdown:
            ex = ",".join(str(i) for i in row["example_ids"][:5])
            f.write(f"| {row['version']} | {row['failure_type']} | {row['count']} | {ex} |\n")
        f.write("\n")

        f.write("## 5. 三版本共有任务上的对比（v1∩v2∩v3）\n\n")
        f.write("共有 {} 个 (scene_id, instruction) 在三版本中均出现。\n\n".format(len(common_keys)))
        # 各版本在共同任务上的成功率
        for ver in ("v1", "v2", "v3"):
            succ_on_common = sum(1 for k in common_keys if by_key[ver][k].get("success"))
            f.write(f"- **{ver}** 在共同任务上成功数: {succ_on_common} / {len(common_keys)}\n")
        f.write("\n前 20 条共同任务的逐条对比见下表（instruction 截断）：\n\n")
        f.write("| scene_id | instruction | v1 成功 | v1 步数 | v2 成功 | v2 步数 | v3 成功 | v3 步数 |\n")
        f.write("|----------|-------------|---------|---------|---------|---------|---------|----------|\n")
        for row in same_key_compare[:20]:
            v1s = "✓" if row.get("v1_success") else "✗"
            v2s = "✓" if row.get("v2_success") else "✗"
            v3s = "✓" if row.get("v3_success") else "✗"
            f.write(f"| {row['scene_id']} | {row['instruction'][:40]}... | {v1s} | {row.get('v1_steps','')} | {v2s} | {row.get('v2_steps','')} | {v3s} | {row.get('v3_steps','')} |\n")
        f.write("\n完整逐条对比已写入 `ppt/common_tasks_compare.csv`。\n")

    # CSV: 汇总指标
    import csv
    metrics_path = os.path.join(out_dir, "version_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_metrics[0].keys()))
        w.writeheader()
        w.writerows(rows_metrics)

    failure_path = os.path.join(out_dir, "failure_breakdown.csv")
    with open(failure_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["version", "failure_type", "count", "example_ids"])
        w.writeheader()
        for row in failure_breakdown:
            row["example_ids"] = ",".join(str(i) for i in row["example_ids"])
            w.writerow(row)

    compare_path = os.path.join(out_dir, "common_tasks_compare.csv")
    if same_key_compare:
        with open(compare_path, "w", newline="", encoding="utf-8") as f:
            keys = list(same_key_compare[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(same_key_compare)
    print("报告已生成: {}".format(md_path))
    print("CSV: {}, {}, {}".format(metrics_path, failure_path, compare_path))
    return md_path


if __name__ == "__main__":
    run()
