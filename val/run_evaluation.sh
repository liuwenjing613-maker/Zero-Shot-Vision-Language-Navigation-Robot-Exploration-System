#!/bin/bash
# ==============================================================================
# VLN 评估脚本 - 在 screen 中运行，支持断点续跑
# 
# 使用方法:
#   1. 先启动云端服务器
#   2. 运行: bash /home/abc/InternVLA/val/run_evaluation.sh
#   3. 评估会在 screen 中后台运行，可以安全关闭终端
#   4. 查看进度: screen -r eval_vln
#   5. 退出 screen (不终止): Ctrl+A 然后按 D
#
# 配置说明:
#   - SR (Success Rate): 最终停止位置距离目标 < 2.0m 才算成功
#   - OSR (Oracle Success Rate): 过程中曾经到达过 < 2.0m 即算成功
#   - 最大步数: 800 步
#   - 测试集: 53 个 episode
#
# 日志文件:
#   - eval_v1.log, eval_v2.log, eval_v3.log
#   - 成功案例轨迹图: success_topdown/
#   - 评估报告: eval_report.txt
# ==============================================================================

set -e

cd /home/abc/InternVLA/val

echo "=========================================="
echo "VLN 评估启动脚本"
echo "=========================================="

# 检查是否已有评估在运行
if screen -ls | grep -q "eval_vln"; then
    echo "⚠️ 发现已有 eval_vln screen 会话"
    echo "请选择:"
    echo "  1. 进入现有会话: screen -r eval_vln"
    echo "  2. 终止并重新开始: screen -X -S eval_vln quit && bash $0"
    exit 1
fi

# 检查云端服务是否可用
echo "检查云端服务..."
if curl -s --connect-timeout 5 http://127.0.0.1:5000 > /dev/null 2>&1; then
    echo "✅ 云端服务已就绪"
else
    echo "❌ 云端服务不可用，请先启动云端服务器"
    echo "   云端启动命令: cd /path/to/cloud && python cloud_brain_server_v2.py"
    exit 1
fi

# 创建 screen 会话运行评估
echo ""
echo "在 screen 中启动评估..."

screen -dmS eval_vln bash -c '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat
cd /home/abc/InternVLA/val

echo "=========================================="
echo "VLN 评估开始 - $(date)"
echo "配置: SR/OSR 阈值=2.0m, 最大步数=800"
echo "=========================================="

echo ""
echo ">>> V1 评估 (基础导航) <<<"
python evaluate_system.py --version v1 --runs 1 2>&1 | tee eval_v1.log
echo "✅ V1 完成 - $(date)"

echo ""
echo ">>> V2 评估 (V1 + FBE) <<<"
python evaluate_system.py --version v2 --runs 1 2>&1 | tee eval_v2.log  
echo "✅ V2 完成 - $(date)"

echo ""
echo ">>> V3 评估 (V2 + 出生地环视 + 目标锁定) <<<"
python evaluate_system.py --version v3 --runs 1 2>&1 | tee eval_v3.log
echo "✅ V3 完成 - $(date)"

echo ""
echo "=========================================="
echo "全部评估完成 - $(date)"
echo "=========================================="
echo ""
echo "查看结果: cat eval_report.txt"
echo "按 Enter 退出..."
read
'

sleep 2

echo ""
echo "=========================================="
echo "✅ 评估已在后台启动"
echo "=========================================="
echo ""
echo "常用命令:"
echo "  查看实时输出:  screen -r eval_vln"
echo "  退出 screen:   Ctrl+A 然后按 D"
echo "  查看 V1 进度:  grep 'SR=' eval_v1.log | wc -l"
echo "  查看 V2 进度:  grep 'SR=' eval_v2.log | wc -l"
echo "  查看 V3 进度:  grep 'SR=' eval_v3.log | wc -l"
echo "  查看最终报告:  cat eval_report.txt"
echo ""
