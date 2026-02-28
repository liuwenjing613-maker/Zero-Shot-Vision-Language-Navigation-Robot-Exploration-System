#!/bin/bash
# ==============================================================================
# 云端启动脚本 - InternVLA-N1 System2 服务
# 运行位置: 云端显卡服务器
# ==============================================================================

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0

# 模型路径
MODEL_PATH="/root/autodl-tmp/.autodl/models/InternVLA-N1-System2"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️ 模型不存在，正在下载..."
    echo "   这可能需要一些时间（约 8GB）"
    mkdir -p "$(dirname $MODEL_PATH)"
    
    # 尝试使用 huggingface-cli
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download InternRobotics/InternVLA-N1-System2 --local-dir "$MODEL_PATH"
    else
        # 使用 git clone
        cd "$(dirname $MODEL_PATH)"
        git clone https://huggingface.co/InternRobotics/InternVLA-N1-System2
        cd InternVLA-N1-System2
        git lfs pull
    fi
fi

# 检查依赖
echo "🔍 检查依赖..."
python3 -c "import torch; import transformers; import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ 缺少依赖，正在安装..."
    pip install torch transformers accelerate flask opencv-python pillow numpy qwen-vl-utils
fi

# 启动服务
echo "🚀 启动 InternVLA-N1 System2 服务..."
cd "$(dirname "$0")"
python3 cloud_s2_server_official.py
