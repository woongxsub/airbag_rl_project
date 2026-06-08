#!/bin/bash
# noVNC 기반 시각화 스크립트 (순수 TCP, 브라우저 접속)
# 접속: http://<RUNPOD_PUBLIC_IP>:<RUNPOD_TCP_PORT_8211>

set -e

DISPLAY_NUM=:1
VNC_PORT=5901
NOVNC_PORT=8211
SCREEN="1280x720x24"

echo "[1/4] Xvfb 시작 (가상 디스플레이 ${DISPLAY_NUM})..."
pkill Xvfb 2>/dev/null || true
pkill x11vnc 2>/dev/null || true
pkill websockify 2>/dev/null || true
sleep 1

Xvfb ${DISPLAY_NUM} -screen 0 ${SCREEN} -ac +extension GLX +render -noreset &
XVFB_PID=$!
echo "  Xvfb PID: $XVFB_PID"
sleep 2

echo "[2/4] x11vnc 시작 (VNC port ${VNC_PORT})..."
x11vnc -display ${DISPLAY_NUM} -rfbport ${VNC_PORT} -nopw -forever -shared -quiet &
X11VNC_PID=$!
echo "  x11vnc PID: $X11VNC_PID"
sleep 1

echo "[3/4] noVNC WebSocket 프록시 시작 (port ${NOVNC_PORT})..."
NOVNC_DIR=$(find /usr -name "novnc" -type d 2>/dev/null | head -1)
[ -z "$NOVNC_DIR" ] && NOVNC_DIR="/usr/share/novnc"

websockify --web ${NOVNC_DIR} ${NOVNC_PORT} localhost:${VNC_PORT} &
WEBSOCKIFY_PID=$!
echo "  websockify PID: $WEBSOCKIFY_PID"
sleep 1

echo ""
echo "======================================================"
PUBLIC_IP=$(env | grep RUNPOD_PUBLIC_IP | cut -d= -f2)
TCP_PORT=$(env | grep RUNPOD_TCP_PORT_8211 | cut -d= -f2)
echo "  브라우저 접속 URL:"
echo "  http://${PUBLIC_IP}:${TCP_PORT}/vnc.html"
echo "======================================================"
echo ""
echo "[4/4] Isaac Sim 학습 시작..."

export DISPLAY=${DISPLAY_NUM}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json

cd /workspace/airbag_rl_project
/workspace/isaacsim_env/bin/python3 train.py "$@"
