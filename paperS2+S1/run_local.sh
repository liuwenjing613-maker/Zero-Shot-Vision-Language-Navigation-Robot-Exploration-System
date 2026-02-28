#!/bin/bash
# ==============================================================================
# 本地启动脚本 - InternVLA-N1 S2+S1 导航
# 运行位置: 本地 Ubuntu 机器
# ==============================================================================

# 检查云端服务是否可用
echo "🔍 检查云端服务..."
curl -s http://127.0.0.1:5000/health > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ 云端服务未启动或不可达"
    echo "   请确保:"
    echo "   1. 云端已运行 cloud_s2_server_official.py"
    echo "   2. SSH 端口转发已设置: ssh -L 5000:localhost:5000 user@cloud"
    exit 1
fi

echo "✅ 云端服务可用"

# 启动本地导航
cd "$(dirname "$0")"
python3 local_robot_s2.py
