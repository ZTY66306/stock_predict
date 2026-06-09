# install_readme_sync_hook.ps1
# PowerShell 版 hook 安装脚本(Windows / PowerShell Core 通用)
#
# 用法:  powershell -ExecutionPolicy Bypass -File scripts\install_readme_sync_hook.ps1
#        或:  .\scripts\install_readme_sync_hook.ps1
#
# 会做两件事:
#   1) 在 .claude\settings.json 中注册 PostToolUse hook
#   2) 在 .git\hooks\pre-commit 中注册 pre-commit hook
#
# 已存在则不会覆盖;卸载直接删除对应条目或 git 的 hook 文件即可。

$ErrorActionPreference = "Stop"

# 解析仓库根目录(脚本所在目录的父目录)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$HookScript = "scripts\check_readme_sync.py"
$Settings   = ".claude\settings.json"
$PreCommit  = ".git\hooks\pre-commit"

# ---------- 1) Claude Code PostToolUse hook ----------
if (-not (Test-Path ".claude")) {
    New-Item -ItemType Directory -Path ".claude" -Force | Out-Null
}

if (-not (Test-Path $Settings)) {
    $content = @"
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "python scripts/check_readme_sync.py",
            "statusMessage": "检查 README 是否需要同步"
          }
        ]
      }
    ]
  }
}
"@
    $content | Set-Content -Path $Settings -Encoding UTF8
    Write-Host "✓ 已创建 $Settings(PostToolUse hook)"
} else {
    Write-Host "• $Settings 已存在,跳过(请手动检查是否包含 check_readme_sync.py 的 PostToolUse hook)"
}

# ---------- 2) Git pre-commit hook ----------
if (Test-Path ".git") {
    $needsWrite = $true
    if (Test-Path $PreCommit) {
        $existing = Get-Content $PreCommit -Raw -ErrorAction SilentlyContinue
        if ($existing -and $existing.Contains("check_readme_sync")) {
            Write-Host "• $PreCommit 已包含本脚本,跳过"
            $needsWrite = $false
        }
    }
    if ($needsWrite) {
        $hookContent = @"
#!/usr/bin/env bash
# Auto-installed by scripts/install_readme_sync_hook.ps1
# (在 Git Bash / WSL 下会被识别;纯 PowerShell 不会执行)
set -e
python "`$REPO_ROOT/scripts/check_readme_sync.py" "`$@" 2>/dev/null || true
"@
        # 注意:pre-commit 钩子 Git 期望可执行(在 Windows 上 Git for Windows 会读它)
        $hookContent | Set-Content -Path $PreCommit -Encoding UTF8
        Write-Host "✓ 已写入 $PreCommit"
    }
} else {
    Write-Host "• .git 目录不存在,跳过 pre-commit 安装"
}

Write-Host ""
Write-Host "完成。后续当 baostock_tool/*.py、pyproject.toml、README.md 等被修改时,"
Write-Host "会自动提示同步 README 指南(详见 README.md「维护本指南」一节)。"
Write-Host ""
Write-Host "Windows 用户提示:"
Write-Host "  - Git Bash / WSL 用户:`bash scripts/install_readme_sync_hook.sh`(可选)"
Write-Host "  - PowerShell 用户:本脚本已就绪,pre-commit 在 Git for Windows 下也能跑"
