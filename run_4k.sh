#!/usr/bin/env bash
# 4000ep 단순 커리큘럼 재실험 — Pure PPO + Curriculum PPO 병렬 실행
# 완료 후 자동 차트 생성
set -euo pipefail

PYTHON="/workspace/isaacsim_env/bin/python"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "[run_4k.sh] 시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Pure PPO 4000ep     → ${LOGDIR}/pure_4k_train.log"
echo "  Curriculum PPO 4000ep → ${LOGDIR}/curriculum_4k_train.log"
echo "=========================================="

# ── Pure PPO ────────────────────────────────────────────────────────────────
"$PYTHON" train.py \
    --mode train \
    --episodes 4000 \
    --seed 42 \
    --label pure_4k \
    > "${LOGDIR}/pure_4k_train.log" 2>&1 &
PID_PURE=$!
echo "[run_4k.sh] Pure PPO PID=$PID_PURE"

# ── Curriculum PPO ──────────────────────────────────────────────────────────
"$PYTHON" train.py \
    --mode curriculum \
    --episodes 4000 \
    --seed 42 \
    --label curriculum_4k \
    > "${LOGDIR}/curriculum_4k_train.log" 2>&1 &
PID_CUR=$!
echo "[run_4k.sh] Curriculum PPO PID=$PID_CUR"

echo "[run_4k.sh] 두 프로세스 완료 대기 중..."

# ── 두 프로세스 모두 대기 ─────────────────────────────────────────────────
PURE_OK=0
CUR_OK=0

wait $PID_PURE  && PURE_OK=1 || echo "[run_4k.sh] [WARN] Pure PPO 비정상 종료 (exit $?)"
wait $PID_CUR   && CUR_OK=1  || echo "[run_4k.sh] [WARN] Curriculum PPO 비정상 종료 (exit $?)"

echo ""
echo "=========================================="
echo "[run_4k.sh] 학습 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Pure PPO 정상 종료: $PURE_OK"
echo "  Curriculum PPO 정상 종료: $CUR_OK"
echo "=========================================="

# ── 체크포인트가 하나라도 있으면 차트 생성 ───────────────────────────────
if ls "${LOGDIR}/pure_4k_ppo_checkpoints.csv" \
      "${LOGDIR}/curriculum_4k_ppo_checkpoints.csv" 2>/dev/null | grep -q .; then
    echo "[run_4k.sh] 차트 생성 중..."
    "$PYTHON" make_charts_6k.py >> "${LOGDIR}/charts_4k.log" 2>&1 \
        && echo "[run_4k.sh] 차트 생성 완료 → results/comparison/" \
        || echo "[run_4k.sh] [WARN] 차트 생성 중 오류 (로그: ${LOGDIR}/charts_4k.log)"
else
    echo "[run_4k.sh] [SKIP] 체크포인트 CSV 없음 — 차트 생략"
fi

echo "[run_4k.sh] 모든 작업 완료: $(date '+%Y-%m-%d %H:%M:%S')"
