"""
Rule-Based 정책 실행 검증 스크립트.

목적:
  1. 3개 에피소드 Rule-Based 정책 실행 (headless, debug)
  2. 첫 에피소드 [Human] 링크 이름·질량 출력 → Hybrid III 기준값 비교
  3. NaN 발생 여부 + 발생 시점 추적
  4. 5개 안전 지표(HIC15/Nij/chest_g/chest_3ms/compression) 출력

실행:
  cd /workspace/airbag_rl_project
  python test_verify.py
"""

import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
from env.airbag_env import AirbagEnv, PHYSICS_DT, CONTROL_DT, COLLISION_STEPS
from baseline.rule_based import rule_based_policy
from rl.reward import (
    compute_hic15, compute_nij, compute_chest_g,
    compute_chest_3ms_clip, compute_chest_compression_mm, compute_femur_force_n,
    HIC_SAFE, NIJ_SAFE, CHEST_G_SAFE, CHEST_3MS_SAFE, CHEST_COMPRESSION_SAFE,
)

# Hybrid III 기준값 (보상함수의 HEAD_MASS_KG, THIGH_MASS_KG와 같은 출처)
HYBRID_III_HEAD_KG  = 4.54
HYBRID_III_THIGH_KG = 8.55

N_EPISODES = 3


def _get_link_masses(physics_view, link_names: list) -> dict:
    """
    physics tensors API로 링크별 질량 취득 시도.
    실패 시 USD MassAPI fallback.
    반환: {link_name: mass_kg} — 취득 실패 링크는 None
    """
    masses_arr = None

    # 1차: physics_view.get_masses() (Isaac Sim 4/5/6 공통 API)
    try:
        raw = physics_view.get_masses()        # (n_envs, n_links) 또는 (n_links,)
        arr = np.asarray(raw).flatten()
        if len(arr) >= len(link_names):
            masses_arr = arr[:len(link_names)]
    except Exception as e:
        print(f"  [질량] physics_view.get_masses() 실패: {e}")

    # 2차: USD MassAPI fallback
    if masses_arr is None:
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics
            stage  = omni.usd.get_context().get_stage()
            human  = stage.GetPrimAtPath("/World/human")
            m_dict = {}
            if human.IsValid():
                for prim in Usd.PrimRange(human):
                    name = prim.GetName()
                    if prim.HasAPI(UsdPhysics.MassAPI):
                        api = UsdPhysics.MassAPI(prim)
                        attr = api.GetMassAttr()
                        if attr.IsValid():
                            m_dict[name] = float(attr.Get())
            return m_dict
        except Exception as e:
            print(f"  [질량] USD MassAPI fallback 실패: {e}")
            return {}

    return {name: float(masses_arr[i]) for i, name in enumerate(link_names)}


def _print_link_masses(human, ep_idx: int):
    """첫 에피소드에서만 링크 이름·질량 출력 + Hybrid III 비교."""
    if ep_idx != 0:
        return
    pv = human._physics_view
    if pv is None:
        print("  [질량] physics_view 없음 — 링크 질량 취득 불가")
        return

    names = human._link_names
    masses = _get_link_masses(pv, names)

    print(f"\n  {'─'*56}")
    print(f"  링크 이름·질량 (Hybrid III 기준: 두부={HYBRID_III_HEAD_KG}kg, 대퇴={HYBRID_III_THIGH_KG}kg)")
    print(f"  {'─'*56}")

    head_name  = names[human._head_idx]  if human._head_idx  < len(names) else "?"
    torso_name = names[human._torso_idx] if human._torso_idx < len(names) else "?"
    thigh_name = names[human._thigh_idx] if human._thigh_idx < len(names) else "?"

    for i, name in enumerate(names):
        m = masses.get(name)
        tag = ""
        if name == head_name:
            tag = f"← head  [idx {human._head_idx}]"
            if m is not None:
                diff = ((m - HYBRID_III_HEAD_KG) / HYBRID_III_HEAD_KG) * 100
                tag += f"  Hybrid III {HYBRID_III_HEAD_KG}kg → {'+' if diff>0 else ''}{diff:.1f}%"
        elif name == torso_name:
            tag = f"← torso [idx {human._torso_idx}]"
        elif name == thigh_name:
            tag = f"← thigh [idx {human._thigh_idx}]"
            if m is not None:
                diff = ((m - HYBRID_III_THIGH_KG) / HYBRID_III_THIGH_KG) * 100
                tag += f"  Hybrid III {HYBRID_III_THIGH_KG}kg → {'+' if diff>0 else ''}{diff:.1f}%"
        mass_str = f"{m:.3f} kg" if m is not None else "N/A"
        print(f"  [{i:2d}] {name:<28} {mass_str:>10}   {tag}")
    print(f"  {'─'*56}")


def _check_nan(col, step: int) -> bool:
    """현재 수집된 샘플에서 NaN 여부 체크."""
    for arr, label in [
        (col.head_acc_g,  "head_acc_g"),
        (col.torso_acc_g, "torso_acc_g"),
    ]:
        if arr and not np.isfinite(arr[-1]):
            print(f"  !! NaN 감지: {label}[-1]={arr[-1]}  step={step}")
            return True
    if col.head_acc_3d:
        if not np.all(np.isfinite(col.head_acc_3d[-1])):
            print(f"  !! NaN 감지: head_acc_3d[-1]={col.head_acc_3d[-1]}  step={step}")
            return True
    return False


def run_episode(env: AirbagEnv, ep_idx: int) -> dict:
    obs, _ = env.reset()
    scenario = env.scenario

    angle             = scenario["angle"]
    is_rollover       = bool(scenario.get("is_rollover", False))
    passenger_present = bool(scenario.get("passenger_present", True))

    action_matrix = rule_based_policy(angle, is_rollover, passenger_present)
    action = np.concatenate([
        action_matrix[:, 0],
        action_matrix[:, 1] / 30.0,
        action_matrix[:, 2] / 600.0,
    ])

    deploy_names = ["FrontDrv", "FrontPass", "SideDrv", "SidePass", "Curtain"]
    deploy_str = "+".join(
        deploy_names[i] for i in range(5) if action_matrix[i, 0] > 0.5
    ) or "없음"

    print(f"\n{'='*64}")
    print(f"  에피소드 {ep_idx+1}/{N_EPISODES}")
    print(f"  각도={angle:.1f}°  속도={scenario['speed']:.1f}km/h  "
          f"rollover={is_rollover}  passenger={passenger_present}")
    print(f"  벨트={'ON' if scenario['seatbelt'] else 'OFF'}  "
          f"신장={scenario['height']:.2f}m  체중={scenario['weight']:.1f}kg")
    print(f"  → 전개: {deploy_str}")
    print(f"{'='*64}")

    # 첫 에피소드: 링크 질량 출력
    _print_link_masses(env.human, ep_idx)

    nan_detected = False
    done = False
    total_r = 0.0
    info = {}

    while not done:
        obs, r, done, _, info = env.step(action)
        total_r += r
        cur_step = env._step

        if _check_nan(env.collector, cur_step):
            nan_detected = True
            print(f"  NaN 발생 → 에피소드 중단 (step={cur_step})")
            break

    col = env.collector
    dt  = PHYSICS_DT
    n_head  = len(col.head_acc_g)
    n_torso = len(col.torso_acc_g)
    n_thigh = len(col.thigh_acc_3d)

    if done and info:
        hic15       = info["hic15"]
        chest_g_val = info["chest_g"]
        chest_3ms   = info["chest_3ms"]
        compression = info["chest_compression_mm"]
        nij         = info["nij"]
        femur_n     = info["femur_n"]          # 로깅용 (보상 제외)
    else:
        hic15       = compute_hic15(col.head_acc_g, dt)
        chest_g_val = compute_chest_g(col.torso_acc_g)
        chest_3ms   = compute_chest_3ms_clip(col.torso_acc_g, dt)
        compression = compute_chest_compression_mm(col.torso_pos_history)
        nij         = compute_nij(col.head_acc_3d)
        femur_n     = compute_femur_force_n(col.thigh_acc_3d)

    def _flag(val, safe):
        if not np.isfinite(val):
            return "!! NaN/Inf"
        return "✓" if val <= safe else f"✗ EXCEEDED ({val/safe:.1f}×)"

    print(f"\n  [안전 지표]  (수집 샘플: head={n_head}  torso={n_torso}  thigh={n_thigh})")
    print(f"  HIC15             : {hic15:>12.1f}     기준 {HIC_SAFE}       {_flag(hic15, HIC_SAFE)}")
    print(f"  Nij               : {nij:>12.4f}     기준 {NIJ_SAFE}         {_flag(nij, NIJ_SAFE)}")
    print(f"  chest_g           : {chest_g_val:>12.2f} g  기준 {CHEST_G_SAFE} g    {_flag(chest_g_val, CHEST_G_SAFE)}")
    print(f"  chest_3ms         : {chest_3ms:>12.2f} g  기준 {CHEST_3MS_SAFE} g    {_flag(chest_3ms, CHEST_3MS_SAFE)}")
    print(f"  chest_compression : {compression:>12.2f} mm 기준 {CHEST_COMPRESSION_SAFE} mm   {_flag(compression, CHEST_COMPRESSION_SAFE)}")
    print(f"  femur_n (로깅만)  : {femur_n:>12.1f} N  기준 {10000}")
    print(f"  NaN 발생여부      : {'예 (상세 위 참조)' if nan_detected else '없음'}")
    print(f"  에피소드 총 보상  : {total_r:.4f}")

    return {
        "ep": ep_idx + 1,
        "angle": angle,
        "speed": scenario["speed"],
        "hic15": hic15,
        "nij": nij,
        "chest_g": chest_g_val,
        "chest_3ms": chest_3ms,
        "compression": compression,
        "femur_n": femur_n,
        "reward": total_r,
        "nan": nan_detected,
        "n_samples_head": n_head,
        "n_samples_torso": n_torso,
    }


def main():
    print("\n" + "=" * 64)
    print("  에어백 RL — Rule-Based 검증 실행")
    print("=" * 64)

    env = AirbagEnv(headless=True, debug=True)
    results = []

    for ep_idx in range(N_EPISODES):
        result = run_episode(env, ep_idx)
        results.append(result)

    env.close()

    print(f"\n{'='*64}")
    print(f"  전체 에피소드 요약  ({N_EPISODES}개)")
    print(f"  {'EP':>2} | {'각도':>6} | {'속도':>6} | {'HIC15':>10} | "
          f"{'Nij':>6} | {'chest_g':>7} | {'3ms':>6} | {'comp':>5} | {'NaN':>4}")
    print(f"  {'-'*2}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*4}")
    for r in results:
        print(f"  {r['ep']:>2} | {r['angle']:>6.1f} | {r['speed']:>5.1f}k | "
              f"{r['hic15']:>10.1f} | {r['nij']:>6.3f} | "
              f"{r['chest_g']:>6.2f}g | {r['chest_3ms']:>5.2f}g | "
              f"{r['compression']:>4.1f}mm | {'Y' if r['nan'] else 'N':>4}")
    print(f"{'='*64}\n")

    sim_app.close()


if __name__ == "__main__":
    main()
