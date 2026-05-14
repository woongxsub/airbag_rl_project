import numpy as np


def rule_based_policy(angle: float) -> np.ndarray:
    """
    충돌 각도만 보고 에어백 조합 결정 (현업 ACU 방식 단순화).
    반환: shape (5,) → 각 에어백 전개 여부 (0 또는 1)
    타이밍·압력은 고정값 사용 (최적화 없음).
    """
    deploy = np.zeros(5, dtype=np.float32)

    if angle <= 45 or angle >= 315:      # 정면
        deploy[0] = 1.0
        deploy[1] = 1.0
    elif 45 < angle <= 135:              # 우측 측면
        deploy[3] = 1.0
        deploy[4] = 1.0
    elif 135 < angle <= 225:             # 후면 → 전개 없음
        pass
    elif 225 < angle <= 315:             # 좌측 측면
        deploy[2] = 1.0
        deploy[4] = 1.0

    FIXED_TIMING = 15.0   # ms
    FIXED_PRESSURE = 300.0  # kPa

    actions = np.zeros((5, 3), dtype=np.float32)
    for i in range(5):
        actions[i, 0] = deploy[i]
        actions[i, 1] = FIXED_TIMING if deploy[i] else 0.0
        actions[i, 2] = FIXED_PRESSURE if deploy[i] else 0.0

    return actions
