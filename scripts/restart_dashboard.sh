#!/usr/bin/env bash
# 杀掉 8501 上卡住的 streamlit 进程,然后后台重启。
# 用法: bash scripts/restart_dashboard.sh
#
# 症状 → 为什么要这个脚本:
#   Streamlit 单线程处理请求,任何一次阻塞(baostock 登录慢、akshare 拉数据、
#   长 K 线请求)都会让主线程卡住,新 WebSocket 全堆在 accept 队列,
#   浏览器就报 "Connecting to Streamlit server error" / "Connection timed out"。
#   Streamlit 1.x 没有请求级超时,只能重启。

set -euo pipefail

PORT="${DASHBOARD_PORT:-8501}"
LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs"
LOG_FILE="$LOG_DIR/streamlit.log"
APP="baostock_tool/dashboard/app.py"

mkdir -p "$LOG_DIR"

# 1) 找 8501 上在听的 streamlit 进程
PIDS=$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print $0}' \
        | grep -oP 'pid=\K[0-9]+' | sort -u || true)

if [ -n "$PIDS" ]; then
  echo "→ 杀掉 8501 上的 streamlit 进程: $PIDS"
  for pid in $PIDS; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in 1 2 3 4 5; do
    sleep 1
    STILL=$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print $0}' \
             | grep -oP 'pid=\K[0-9]+' | sort -u || true)
    [ -z "$STILL" ] && break
  done
  for pid in $STILL; do kill -KILL "$pid" 2>/dev/null || true; done
else
  echo "→ 8501 没人听,直接启动"
fi

# 2) 启动
echo "→ 启动 streamlit,日志: $LOG_FILE"
cd "$(dirname "$LOG_DIR")"
nohup python -m streamlit run "$APP" \
  --server.headless true \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --browser.gatherUsageStats false \
  > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "→ pid=$NEW_PID"

# 3) 等到 healthz 通
for i in $(seq 1 15); do
  sleep 1
  if curl -sf "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then
    echo "✓ 健康检查通过 (用了 ${i}s)"
    echo "→ 打开: http://localhost:$PORT"
    exit 0
  fi
done

echo "✗ 15s 内 healthz 没通过,看日志:"
tail -30 "$LOG_FILE"
exit 1
