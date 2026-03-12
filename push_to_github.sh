#!/bin/bash
# ==============================================================================
# 一键上传并推送到 GitHub
# 用法:
#   ./push_to_github.sh           # 默认提交说明，会询问确认
#   ./push_to_github.sh -y        # 一键同步，不询问
#   ./push_to_github.sh "fix: xxx"  # 自定义提交说明
#   ./push_to_github.sh -y "update v3 eval"
# ==============================================================================

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SKIP_CONFIRM=false
COMMIT_MSG=""
for arg in "$@"; do
    if [ "$arg" = "-y" ] || [ "$arg" = "--yes" ]; then
        SKIP_CONFIRM=true
    else
        COMMIT_MSG="$arg"
    fi
done
[ -z "$COMMIT_MSG" ] && COMMIT_MSG="sync: $(date '+%Y-%m-%d %H:%M:%S')"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ 当前目录不是 Git 仓库"
    exit 1
fi

echo "📁 仓库: $REPO_ROOT"
echo "📝 提交说明: $COMMIT_MSG"
echo ""

# 不再跟踪 .mp4 文件（从索引移除，后续由 .gitignore 忽略）
while IFS= read -r f; do
    [ -z "$f" ] && continue
    git rm --cached --ignore-unmatch "$f" 2>/dev/null && echo "  已取消跟踪: $f"
done < <(git ls-files '*.mp4' 2>/dev/null)

# 确保 .gitignore 忽略 .mp4
if ! grep -q '\.mp4' .gitignore 2>/dev/null; then
    echo "*.mp4" >> .gitignore
fi

# 添加所有变更（遵守 .gitignore）
git add -A
STATUS=$(git status --short)

if [ -z "$STATUS" ]; then
    echo "✅ 工作区干净，没有需要提交的变更"
    if git status | grep -q "Your branch is ahead of 'origin/main'"; then
        echo "📤 正在推送已有提交..."
        git push origin main
        echo "✅ 推送完成"
    else
        echo "无需推送"
    fi
    exit 0
fi

echo "📋 待提交变更:"
echo "$STATUS"
echo ""
if [ "$SKIP_CONFIRM" = false ]; then
    read -p "确认提交并推送? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
        echo "已取消"
        exit 0
    fi
fi

git commit -m "$COMMIT_MSG"
echo "📤 正在推送到 origin main..."
git push origin main
echo "✅ 已提交并推送完成"
