# 실험 A vs B — 폭발 임계값 비교 (200ep PPO, --exclude-explosions)

| 항목 | 실험 A (기존: HIC15>1,000,000) | 실험 B (신규: HIC15>270,000 또는 chest_g>3,885) |
|---|---|---|
| 임계값 | `EXPLOSION_HIC_THRESHOLD=1_000_000` (chest_g 조건 없음) | `EXPLOSION_HIC_THRESHOLD=270_000`, `EXPLOSION_CHEST_THRESHOLD=3_885` |
| 학습 episodes | 200 | 200 |
| 소요 시간 | 39.5분 | 39.2분 |
| **폭발 제외 비율** | 111/200 (**55.5%**) | 100/200 (**50.0%**) |
| **비폭발 HIC15 median** | **14,908** (n=89) | **15,070** (n=100) |
| **비폭발 chest_g median** | **772.9g** (n=89) | **851.1g** (n=100) |
| critic_loss mean / std | 6.408 / 0.518 | 6.193 / 0.539 |
| critic_loss min–max | 5.513 – 7.364 | 5.264 – 7.287 |
| critic_loss updates 수 | 17 | 20 |

## 해석

- **폭발 제외 비율**: 임계값을 1M→270K/3885g로 낮추자 제외 비율이 55.5%→50.0%로 오히려 **감소**했다. 새 임계값이 더 엄격함(chest_g 조건 추가 + HIC15 컷 하향)에도 제외 비율이 줄어든 것은, 두 실행이 서로 다른 PPO 정책 궤적(랜덤 시나리오 샘플링 + 정책 업데이트)을 따라가기 때문 — 임계값 자체보다 실행 간 분산(run-to-run variance)의 영향이 크다는 뜻이다. 200ep는 통계적으로 작은 표본이라 이 차이를 임계값 효과로 단정하기는 어렵다.
- **비폭발 HIC15 median**: A=14,908 vs B=15,070으로 거의 동일 (차이 ~1%). 새 임계값이 "비폭발"로 분류하는 잔여 집단의 HIC15 분포 자체는 크게 달라지지 않았다.
- **비폭발 chest_g median**: A=772.9g vs B=851.1g. B가 다소 높지만 여전히 새 chest_g 컷(3,885g)보다 한참 낮다 — chest_g 기준 추가가 "비폭발" 그룹의 chest_g 분포를 크게 왜곡하지는 않았다.
- **critic_loss 안정성**: std 기준 A=0.518, B=0.539로 거의 동일 수준(B가 근소하게 더 큰 변동성이지만 유의미한 차이로 보기 어려움). 두 경우 모두 critic_loss가 5.3~7.4 범위에서 진동 — 200ep(버퍼 batch=256 기준 17~20회 업데이트)로는 수렴 여부를 판단하기엔 표본이 부족하다.

## 한계

- 각 실험 1회씩만 실행 (seed 미고정, `--seed` 옵션 없이 실행) → run-to-run 분산과 임계값 변경 효과를 분리할 수 없음. 결론을 신뢰하려면 동일 seed로 A/B 각각 복수 회 반복하거나, 더 많은 episode(예: 1000+)로 재실행 권장.
- 200ep는 체크포인트가 100ep 단위 2개뿐이라 추세선 판단에 한계가 있음.

## 산출물

- `results/logs/A_ppo_*.npy`, `A_ppo_checkpoints.csv`, `results/models/A_ppo_final.pt`
- `results/logs/B_ppo_*.npy`, `B_ppo_checkpoints.csv`, `results/models/B_ppo_final.pt`
