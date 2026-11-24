# M1 手动驾驶台

入口为现有 M1 服务的 `/live`。机器人运行在 MP3D 场景 2azQ1b91cZZ 的同一层，可选厨房拱门或 T 型路口附近作为起点。默认测试任务为寻找椅子。W/S 前进、后退 0.25 米，A/D 转向 15°，空格原地观察。网页支持回看、恢复原会话和导出 ZIP。

每个成功动作保存同步 RGB、米制深度、相机位姿、底盘位姿、720 束扫描及数值网格。地图随动作更新。Qwen 逐帧产生目标、上下文实体和导航线索，后台以帧号顺序将结果提交给 `vln_memory.online_memory.OnlineMemory`。几何、区域和岔路检测共享同一张拓扑图。导航边使用实际路程。选定 Frontier 后，手动走到观察点 0.35 米内并按观察，已提交帧才会记录抵达事件。Frontier 候选用于网页试验；对象引用使用观测 ID 与帧内对象 ID，跨帧全局对象关联属于后续对象记忆。

## 模型与预热

使用项目原有的 Qwen3.8 27B 权重、物理 GPU 3 及独立本机 Ollama 服务。先确认 18281 的监听状态及可用显存，再按项目配置启动。服务已运行时直接检查驻留状态即可。

```bash
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader)" \
OLLAMA_MODELS=/data/chenkejun/open_models/ollama OLLAMA_HOST=127.0.0.1:18281 \
OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1 \
/data/chenkejun/quant-platform/runtime/ollama-v0.40.5/bin/ollama serve
```

驾驶台服务启动时请求加载并常驻 `qwen3.8:27b`，页面模型状态读取实际驻留清单。使用 `/api/chat`、`think=false`、结构化解码，与原感知模块共用目标规格、提示词和输出校验。每个新 RGB 都使用当前会话目标推理；会话内像素文件哈希完全相同的观测复用结果，并分配独立观测编号。`vlm.json` 记录权重名称、来源、RGB 哈希和计时。原试运行会话切换模型时会保留原始记录，在新的 27B 会话中从第一帧重建语义记忆。

## 验收和存储

项目根目录下运行 `PYTHONPATH=src /data/chenkejun/vln/envs/vln-sim/bin/python -m unittest discover -s tests -v`。现有 M1 服务入口为 `evals/m1_lab/serve.py`。服务器持久化目录为 `/data/chenkejun/vln/m1-web-runs/live/<session>`。每一帧目录包括 `step.json`、`sensors.npz`、`grid.npz`、`rgb.png`；`vlm.json` 和 `memory.json` 随感知提交出现。队列异常在页面可见并可重试。刷新网页恢复原会话，服务重启后以已保存传感器和已校验的感知结果重建运行状态。