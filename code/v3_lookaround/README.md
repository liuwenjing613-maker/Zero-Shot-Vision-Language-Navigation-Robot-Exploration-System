# v3 Lookaround（出生地环视版）

与 **v2**（`code/v2_FBE`）区分：本目录为 **出生地 4 视野选最高可能性方向** 的版本。

- **v2**：FBE + 短期记忆 + 多目标，无出生地环视。
- **v3**：在 v2 基础上增加 **出生地 4 个视野**（0° / 90° / 180° / 270°），每个视野向云端请求一次 **目标可能性（confidence 0~1）**，选可能性最高的方向转向并出发；环视与转向过程在 Robot Eye 窗口中逐帧可视化。云端需支持 `scan=1` 的 scan 模式（返回 `confidence`）。
- **转满一周**：若实际环视未转满 360°，在脚本中调小 `DEGREES_PER_TURN`（如改为 `10` 表示每步约 10°，则每 90° 用 9 步；若为 30° 则改为 `30`，每 90° 用 3 步）。

## 运行

- 云端仍使用 v2 的 `cloud_brain_server_v2.py`（`CLOUD_URL = http://127.0.0.1:5000/plan`）。
- 本地运行本目录下的 `local_robot_simple_v3.py` 即可。

```bash
# 终端1：启动云端（在 v2_FBE 或本目录均可）
python code/v2_FBE/cloud_brain_server_v2.py

# 终端2：运行 v3 环视版
python code/v3_lookaround/local_robot_simple_v3.py
```
