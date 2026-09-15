import numpy as np
from env.airbag import PRESSURE_OPT


def rule_based_policy(
    angle: float,
    is_rollover: bool = False,
    passenger_present: bool = True,
) -> np.ndarray:
    """
    Hyundai 투싼(NX4, 2025) 공식 매뉴얼 및 OCS 센서 기반 에어백 전개 매핑.

    에어백 인덱스 정의:
      index 0: 운전석 Front airbag
      index 1: 동승석 Front airbag
      index 2: 운전석 Side  airbag  (차량 좌측)
      index 3: 동승석 Side  airbag  (차량 우측)
      index 4: Curtain airbag       (좌·우 통합)

    충돌 방향별 전개 매핑 (Hyundai NX4 2025 매뉴얼 기준):
      정면  (315°~360° / 0°~45°) : Front 운전석 + 동승석
      우측 측면 (45°~135°)        : 동승석 Side + Curtain
      후면      (135°~225°)       : 미전개
      좌측 측면 (225°~315°)       : 운전석 Side + Curtain
      전복 (rollover)             : Side 양측 + Curtain  ← 각도 분기보다 우선

    is_rollover=True : 독립 rollover sensor(자이로) 트리거 — 각도 분기 무시하고 Side + Curtain 양측 전개.
    passenger_present=False : OCS 미점유 판정 시 동승석 Front(1), 동승석 Side(3) 강제 미전개.

    반환: shape (5, 3) — 각 행 [deploy, timing_ms, pressure_kPa]
    타이밍·압력은 고정값 사용 (최적화 없음, Rule-Based 베이스라인 전용).
    """
    # 타이밍 근거 (15 ms 고정값):
    #   출처1: Airbag24, "What is an SRS Control Unit? Function Explained"
    #     https://www.airbag24.de/en/blogs/guide/srs-control-unit
    #     인용: "Within 10 ms, the control unit determines crash type and severity"
    #   출처2: US Patent 6460882, "Airbag actuation event discrimination system"
    #     https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6460882
    #     인용: "The algorithm calculations begin at around 10 ms ... terminates at around 17.5 ms"
    #   출처3: AllAirbags, "How Fast Do Airbags Deploy?"
    #     https://allairbags.com/blogs/our-blog/how-fast-do-airbags-deploy
    #     인용: "Signal sent (10-15 ms): The ACU determines if airbag deployment is needed."
    #   → 3개 출처 공통 판단 구간 10~18 ms. 15 ms는 이 구간 내 합리적 고정값.
    FIXED_TIMING = 15.0  # ms

    # 압력 근거: 본 시뮬레이션 압력-감쇠 모델(Reverse U-Curve)에서
    #   감쇠 효율이 이론적으로 최대인 지점 = PRESSURE_OPT(130.0 kPa).
    #   실측 데이터가 아닌 시뮬레이션 모델상의 최적점임.
    #   Action 탐색 범위 자체는 실측 문헌(0~250 kPa) 기반으로 현실화되어 있음.
    FIXED_PRESSURE = PRESSURE_OPT  # 130.0 kPa

    deploy = np.zeros(5, dtype=np.float32)

    if is_rollover:
        # 전복: 독립 rollover sensor 트리거 — 충돌 각도 분기와 별개로 우선 처리
        deploy[2] = 1.0  # 운전석 Side
        deploy[3] = 1.0  # 동승석 Side
        deploy[4] = 1.0  # Curtain
    elif angle <= 45 or angle >= 315:   # 정면
        deploy[0] = 1.0  # 운전석 Front
        deploy[1] = 1.0  # 동승석 Front
    elif 45 < angle <= 135:             # 우측 측면
        deploy[3] = 1.0  # 동승석 Side
        deploy[4] = 1.0  # Curtain
    elif 135 < angle <= 225:            # 후면 → 전개 없음
        pass
    elif 225 < angle <= 315:            # 좌측 측면
        deploy[2] = 1.0  # 운전석 Side
        deploy[4] = 1.0  # Curtain

    # OCS 동승자 미점유: 동승석 에어백 강제 미전개 (운전석 무관)
    if not passenger_present:
        deploy[1] = 0.0  # 동승석 Front
        deploy[3] = 0.0  # 동승석 Side

    actions = np.zeros((5, 3), dtype=np.float32)
    for i in range(5):
        actions[i, 0] = deploy[i]
        actions[i, 1] = FIXED_TIMING   if deploy[i] else 0.0
        actions[i, 2] = FIXED_PRESSURE if deploy[i] else 0.0

    return actions
