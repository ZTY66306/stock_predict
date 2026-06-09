#!/usr/bin/env python3
"""PostToolUse hook:当源码 / 配置 / README 变更时,提示同步 README 指南。

触发条件(均在仓库根目录下匹配):
    baostock_tool/*.py
    pyproject.toml
    README.md
    examples/*.py
    tests/*.py

读取 Claude Code PostToolUse 传入的 JSON(tool_name / tool_input),匹配则输出提示。
stdout 会被合入 Claude 的上下文,exit 0 表示非阻塞。
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


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    tool_input = data.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path", "")
    if not is_target(file_path):
        return 0
    rel = os.path.relpath(file_path, ".").replace(os.sep, "/")
    print(
        f"\n📘 README 同步提示:你刚修改了 `{rel}`,"
        f"请确认 `README.md` 的相关章节(尤其是 {GUIDE_SECTION} 里的同步对照表)仍然准确。\n"
        f"   - 新增/删除 `baostock_tool/*.py` 子模块 → 更新「项目结构」+「功能概览」+ 顶部 TOC\n"
        f"   - `cli.py` 子命令或参数变化 → 更新「CLI 速查」+ 子命令计数\n"
        f"   - `screener.SCREEN_TEMPLATES` 增删模板 → 更新「选股模板」表格\n"
        f"   - `grid_backtest` 参数/规则变化 → 更新「网格交易回测」章节与示例\n"
        f"   - `pairs_trading` / `EnsembleStrategy` / `RollingRobustness` 变化 → "
        f"更新对应章节与示例\n"
        f"   - `fund_flow` / `market_overview` 接口变化 → 更新对应章节\n"
        f"   - `dca` / `paper_trader` 增删字段 → 更新「智能定投 DCA」/「实盘模拟器」章节\n"
        f"   - 新增/删除 `baostock_tool/dashboard/pages/*.py` → 更新「Web 看板」页面清单\n"
        f"   - `pyproject.toml` 新增 `[project.scripts]` 入口 → 更新「Web 看板」+「安装」\n"
        f"   - 其它 API 变化 → 同步对应的示例代码与说明。\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
