# eval_episodes.json 标注数据详细分析

本文档对 VLN 评估测试集 `eval_episodes.json` 进行统计与结构分析，便于理解数据分布与评估设计。

---

## 1. 概览

| 项目 | 数值 |
|------|------|
| **总 Episode 数** | 53 |
| **Episode ID 范围** | 0 ~ 52 |
| **场景数** | 10 个 (Matterport3D scene_id) |
| **唯一指令数** | 49 条（存在同文不同场景的重复） |
| **每条记录字段** | `scene_id`, `instruction`, `start_position`, `gt_position`, `episode_id` |

---

## 2. 场景分布

各场景的 episode 数量与对应 episode_id 如下。

| 场景 ID (scene_id) | Episode 数 | Episode ID 列表 |
|-------------------|------------|-----------------|
| TbHJrupSAjP       | 12         | 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 47 |
| X7HyMhZNoso       | 10         | 17, 18, 19, 20, 21, 22, 23, 24, 25, 49 |
| 2azQ1b91cZZ       | 8          | 0, 1, 2, 3, 4, 5, 6, 52 |
| 8194nk5LbLH       | 6          | 10, 11, 12, 13, 14, 50 |
| x8F5xyUWy9e       | 6          | 27, 28, 29, 30, 31, 48 |
| Z6MFQCViBuw       | 4          | 7, 8, 9, 51 |
| EU6Fwq7SyZv       | 2          | 15, 16 |
| zsNo4HB9uLZ       | 2          | 41, 42 |
| QUCTc6BB5sX       | 2          | 45, 46 |
| pLe4wQe7qrG       | 1          | 26 |

- **最多**：TbHJrupSAjP（12 条）
- **最少**：pLe4wQe7qrG（1 条）
- **合计**：53 条，与总 episode 数一致

---

## 3. 指令 (instruction) 统计

### 3.1 长度

| 指标 | 字符数 | 词数（按空格分） |
|------|--------|------------------|
| 最短 | 2      | 1                |
| 最长 | 40     | 9                |
| 平均 | ≈17.3  | ≈3.2             |

### 3.2 重复指令（同文不同场景/起点）

以下指令在数据集中出现多次，对应不同 episode（不同场景或不同起点/终点）：

| 指令原文 | 出现次数 | 对应 (episode_id, scene_id) |
|----------|----------|-----------------------------|
| toilet   | 2        | (2, 2azQ1b91cZZ), (17, X7HyMhZNoso) |
| stairs   | 2        | (6, 2azQ1b91cZZ), (30, x8F5xyUWy9e) |
| stairs between floors. | 3 | (43, TbHJrupSAjP), (44, TbHJrupSAjP), (50, 8194nk5LbLH) |

- 共 **3 组** 重复指令，涉及 **7** 个 episode。
- “stairs between floors.” 在同一场景 TbHJrupSAjP 出现 2 次，在 8194nk5LbLH 出现 1 次，可考察同一场景内多目标与跨场景泛化。

---

## 4. 起点–终点距离（欧氏距离）

使用 `start_position` 与 `gt_position` 计算的欧氏距离（米），仅作参考（真实评估应使用 Habitat 的 geodesic）。

| 统计量 | 数值 (m) |
|--------|----------|
| 最小   | 0.78     |
| 最大   | 14.25    |
| 平均   | ≈4.10    |

### 距离分布（欧氏距离分段）

| 距离区间 (m) | Episode 数 | 占比（约） |
|--------------|------------|------------|
| [0, 2)       | 6          | 11.3%      |
| [2, 4)       | 27         | 50.9%      |
| [4, 6)       | 14         | 26.4%      |
| [6, 8)       | 4          | 7.5%       |
| [8, 10)      | 0          | 0%         |
| [10, 15)     | 2          | 3.8%       |

约一半 episode 的起终欧氏距离在 2–4 m，中短距离任务为主，少数为 10 m 以上的长距离。

---

## 5. 数据格式说明

每条 episode 为一条 JSON 对象，例如：

```json
{
  "scene_id": "2azQ1b91cZZ",
  "instruction": "the first flowers you see",
  "start_position": [8.23, 0.13, 3.41],
  "gt_position": [14.36, 0.13, 2.01],
  "episode_id": 0
}
```

- **scene_id**: Matterport3D 场景 ID，对应 `scenes/` 下 `.glb` 场景。
- **instruction**: 自然语言导航目标，英文。
- **start_position**: 起点 `[x, y, z]`（米）。
- **gt_position**: 目标点 `[x, y, z]`（米），评估时以与该点距离 < 阈值判定成功。
- **episode_id**: 全局唯一 episode 编号 (0–52)。

---

## 6. 唯一指令列表（共 49 条）

以下为按字典序排列的全部不重复指令，便于检查覆盖的目标类型：

1. TV  
2. TV in the living room  
3. animal  
4. armchair  
5. bathtub  
6. bed in the bedroom  
7. bedside lamp  
8. black board  
9. black single sofa  
10. blue cushion  
11. blue rubbish bin  
12. blue table in the living room  
13. car.  
14. chair near the door  
15. chair under the painting  
16. clock on the wall  
17. closed door  
18. fire extinguisher beside the door  
19. fireplace  
20. first white door you see  
21. green ball  
22. gym.  
23. mirror  
24. oil painting of the man in the red coat.  
25. orange single sofa  
26. plant  
27. plant beside the fire extinguisher  
28. plant between two sofas  
29. potted plant beside the door  
30. purple cushion  
31. red bed  
32. red book on the table  
33. red stool in front of the red curtain  
34. shelf beside the door  
35. sink.  
36. stairs  
37. stairs between floors.  
38. the central pillar  
39. the first flowers you see  
40. the potted plant between the two sofas  
41. three lounge chairs  
42. toilet  
43. toilet in the bathroom.  
44. toilet.  
45. wall-themed wall art.  
46. washing machine  
47. white bouquet on the bedside table.  
48. white carpet in the bedroom  
49. yellow blanket on the sofa  

---

## 7. 小结与使用建议

- **规模**：53 episodes，10 个场景，49 条唯一指令，适合做 v1/v2/v3 等小规模对比评估。
- **场景**：TbHJrupSAjP、X7HyMhZNoso、2azQ1b91cZZ 占比较多，分析时可按场景分层统计 SR/SPL。
- **指令**：短指令（1 词）与较长描述（如 “red stool in front of the red curtain”）均有，可分别看模型对简单/复杂描述的表现。
- **重复指令**：toilet、stairs、stairs between floors. 在多个场景或同一场景多次出现，可用于检查一致性与泛化。
- **距离**：多数为 2–6 m 欧氏距离，评估时建议同时关注短距离与长距离（如 >8 m）子集的指标。

评估脚本使用本文件时，通常按 `episode_id` 顺序或按 `scene_id` 分组运行，成功判定以 `gt_position` 与 agent 终点的 geodesic/欧氏距离 < 阈值（如 2.0 m）为准；详细结果可结合 `val/detailed_results/` 中的 per-episode 输出做进一步分析。

---

*文档由对 `eval_episodes.json` 的统计生成。数据更新后建议重新跑统计并更新本文档。*
