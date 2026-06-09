#!/usr/bin/env python3
"""README 同步检查 — 在源码/配置变更时提示同步 README.md。

本脚本是 hook 逻辑本体,可被以下任一方式调用:
    1) Claude Code 的 PostToolUse hook(写入 .claude/settings.json 后,Edit/Write 自动触发)
    2) Git pre-commit hook(本地提交前自检)
    3) CI 工作流(在 PR 检查中跑)

触发条件(均相对仓库根目录):
    baostock_tool/*.py
    pyproject.toml
    README.md
    examples/*.py
    tests/*.py

输入(JSON via stdin): {"tool_name": "...", "tool_input": {"file_path": "..."}, ...}
退出码: 0 = 正常(无目标或仅提示)
"""
from __future__ import annotations

import json
import os
import sys

WATCH_PATTERNS = (
    "baostock_tool/",
    "pyproject.toml",
    "README.md",
    "examples/",
    "tests/",
)
GUIDE_SECTION = '"维护本指南" 小节'


def is_target(path: str) -> bool:
    if not path:
        return False
    rel = os.path.relpath(path, ".").replace(os.sep, "/")
    return any(rel.startswith(p) for p in WATCH_PATTERNS)


def hint(file_path: str) -> str:
    rel = os.path.relpath(file_path, ".").replace(os.sep, "/")
    return (
        f"\n📘 README 同步提示:检测到 `{rel}` 变更,"
        f"请确认 `README.md` 的相关章节(尤其是 {GUIDE_SECTION} 里的同步对照表)仍然准确。\n"
        f"   - 新增/删除 `baostock_tool/*.py` 子模块 → 更新「项目结构」+「功能概览」+ 顶部 TOC\n"
        f"   - `cli.py` 子命令或参数变化 → 更新「CLI 速查」+ 子命令计数\n"
        f"   - `screener.SCREEN_TEMPLATES` 增删模板 → 更新「选股模板」表格\n"
        f"   - `grid_backtest` 参数/规则变化 → 更新「网格交易回测」章节与示例\n"
        f"   - 其它 API 变化 → 同步对应的示例代码与说明。\n"
    )


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    tool_input = data.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path", "")
    if not is_target(file_path):
        return 0
    print(hint(file_path))
    return 0


# ============ 直接 CLI 调用模式:python scripts/check_readme_sync.py <file> ============

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 把传入的文件路径当 hook 输入
        fp = os.path.abspath(sys.argv[1])
        if is_target(fp):
            print(hint(fp))
    else:
        sys.exit(main())
