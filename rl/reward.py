import numpy as np

HIC_SAFE = 700.0
CHEST_G_SAFE = 60.0
CHEST_COMPRESSION_SAFE = 50.0
FEMUR_SAFE = 10000.0   # N (10kN)
NECK_SAFE = 3100.0     # N (3.1kN)

NO_DEPLOY_PENALTY = -2.0   # 에어백을 하나도 안 전개 시 페널티


def compute_hic(head_acc_history: list, dt: float = 0.001) -> float:
    """
    간소화된 HIC 근사: 최대 가속도 기반.
    실제 HIC = max[(t2-t1) * mean_acc^2.5] 이지만
    프로토타입에서는 peak_acc^2.5 * 0.015로 근사.
    추후 가속도 시계열이 안정화되면 정식 적분으로 교체.
    """
    if not head_acc_history:
        return 0.0
    peak = max(np.linalg.norm(a) for a in head_acc_history)
    peak_g = peak / 9.81
    return float(peak_g ** 2.5 * 0.015)


def compute_reward(
    hic: float,
    chest_g: float,
    chest_compression_mm: float,
    femur_n: float,
    neck_n: float,
    deploy_flags: list,
) -> float:
    r = 0.0
    r -= hic / HIC_SAFE
    r -= chest_g / CHEST_G_SAFE
    r -= chest_compression_mm / CHEST_COMPRESSION_SAFE
    r -= femur_n / FEMUR_SAFE
    r -= neck_n / NECK_SAFE

    if sum(deploy_flags) == 0:
        r += NO_DEPLOY_PENALTY

    return float(r)
