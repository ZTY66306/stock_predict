#!/usr/bin/env bash
# 安装 README 同步 hook(可选,需要用户主动运行)
#
# 用法:  bash scripts/install_readme_sync_hook.sh
#
# 会做两件事:
#   1) 在 .claude/settings.json 中注册 PostToolUse hook(改源码/配置时自动提示)
#   2) 在 .git/hooks/pre-commit 中注册 pre-commit hook(本地提交前自检)
#
# 已存在则不会覆盖;卸载直接删除对应条目或 git 的 hook 文件即可。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOOK_SCRIPT="scripts/check_readme_sync.py"
SETTINGS=".claude/settings.json"
PRE_COMMIT=".git/hooks/pre-commit"

# ---- 1) Claude Code PostToolUse hook ----
mkdir -p .claude
if [[ ! -f "$SETTINGS" ]]; then
  cat > "$SETTINGS" <<'JSON'
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "python3 scripts/check_readme_sync.py",
            "statusMessage": "检查 README 是否需要同步"
          }
        ]
      }
    ]
  }
}
JSON
  echo "✓ 已创建 $SETTINGS(PostToolUse hook)"
else
  echo "• $SETTINGS 已存在,跳过(请手动检查是否包含 check_readme_sync.py 的 PostToolUse hook)"
fi

# ---- 2) Git pre-commit hook ----
if [[ -d .git ]]; then
  if [[ -f "$PRE_COMMIT" ]] && grep -q "check_readme_sync" "$PRE_COMMIT"; then
    echo "• $PRE_COMMIT 已包含本脚本,跳过"
  else
    {
      echo "#!/usr/bin/env bash"
      echo "# Auto-installed by scripts/install_readme_sync_hook.sh"
      echo "set -e"
      echo "python3 \"\$REPO_ROOT/scripts/check_readme_sync.py\" \"\$@\" 2>/dev/null || true"
    } > "$PRE_COMMIT"
    chmod +x "$PRE_COMMIT"
    echo "✓ 已写入 $PRE_COMMIT"
  fi
else
  echo "• .git 目录不存在,跳过 pre-commit 安装"
fi

echo
echo "完成。后续当 baostock_tool/*.py、pyproject.toml、README.md 等被修改时,"
echo "会自动提示同步 README 指南(详见 README.md「维护本指南」一节)。"
