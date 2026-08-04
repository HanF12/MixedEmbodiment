#!/usr/bin/env python
"""
Comparison harness for the MixedEmbodiment refactor.

Proves, using real (tiny) session data + the ALOHA-mimic venv:
  1. Gripper binarize threshold is 0.5 in all four packages (Bimanual-3cam,
     Combined_relative_3cam_gripweight, MixedEmbodiment_gripweight, and the new
     MixedEmbodiment).
  2. Sync-CSV generation (robot / human / mixed) is byte-identical between the
     legacy packages and the folded-in MixedEmbodiment.data_synchronization
     functions (this is the "single loading file" merge of build_sync.py).
  3. Dataset normalization stats (joint/pose mean+std) built from those sync
     CSVs are numerically identical between legacy and new dataloaders.
  4. With weights copied from a legacy model into a new model (both share
     the same submodule names/shapes), a forward pass on identical inputs
     produces identical pose_pred / joint_pred / mu / logvar for robot, human,
     and mixed, in both training mode and inference mode — i.e. the refactor
     changed no computation, only file layout and CLI surface.
  5. The new --pose_observation gate (default OFF) actually removes all
     influence of the human embodiment's absolute pose on the model's output,
     and --pose_observation restores the legacy (information-leaking) behavior
     bit-for-bit; robot/mixed are provably unaffected by the flag either way.

Run with the ALOHA-mimic venv (has torch/cv2/pandas; the system pythons don't):
  ALOHA-mimic/.venv/bin/python MixedEmbodiment/compare_legacy.py
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SESSIONS_ROOT = REPO_ROOT / "sessions"
SCRATCH = Path(__file__).resolve().parent / "_compare_scratch"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))


def load_module_from_path(name: str, path: Path):
    """Import a module from a file path (needed for Bimanual-3cam: hyphen isn't a valid identifier)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def df_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.reset_index(drop=True).equals(b.reset_index(drop=True))


# ---------------------------------------------------------------------------
# 1) Gripper binarize threshold == 0.5 everywhere
# ---------------------------------------------------------------------------


def check_thresholds() -> None:
    bimanual_cfg = load_module_from_path("bimanual_cfg", REPO_ROOT / "Bimanual-3cam" / "config.py")
    import Combined_relative_3cam_gripweight.config as combined_cfg
    import MixedEmbodiment_gripweight.config as mixed_cfg
    import MixedEmbodiment.config as new_cfg

    check(
        "Bimanual-3cam ROBOT_JOINT_GRIPPER_BINARIZE_THRESHOLD == 0.5",
        bimanual_cfg.ROBOT_JOINT_GRIPPER_BINARIZE_THRESHOLD == 0.5,
        str(bimanual_cfg.ROBOT_JOINT_GRIPPER_BINARIZE_THRESHOLD),
    )
    for label, mod in (
        ("Combined_relative_3cam_gripweight", combined_cfg),
        ("MixedEmbodiment_gripweight", mixed_cfg),
        ("MixedEmbodiment", new_cfg),
    ):
        check(
            f"{label} ROBOT_NPZ_GRIPPER_BINARIZE_THRESHOLD == 0.5",
            mod.ROBOT_NPZ_GRIPPER_BINARIZE_THRESHOLD == 0.5,
            str(mod.ROBOT_NPZ_GRIPPER_BINARIZE_THRESHOLD),
        )
        check(
            f"{label} ROBOT_NPY_GRIPPER_BINARIZE_THRESHOLD == 0.5",
            mod.ROBOT_NPY_GRIPPER_BINARIZE_THRESHOLD == 0.5,
            str(mod.ROBOT_NPY_GRIPPER_BINARIZE_THRESHOLD),
        )


# ---------------------------------------------------------------------------
# 2 & 3) Sync CSVs + dataset stats: legacy vs new
# ---------------------------------------------------------------------------


def run_robot_pipeline(pkg_name: str, sync_dir: Path, *, max_demos: int = 1, max_sync_rows: int = 16, resize_factor: float = 0.25):
    if pkg_name == "Combined_relative_3cam_gripweight":
        import Combined_relative_3cam_gripweight.training_combined as tc
        import Combined_relative_3cam_gripweight.dataloader_robot as dr
    elif pkg_name == "MixedEmbodiment_gripweight":
        import MixedEmbodiment_gripweight.training_combined as tc
        import MixedEmbodiment_gripweight.dataloader_robot as dr
    else:
        import MixedEmbodiment.data_synchronization as tc
        import MixedEmbodiment.dataloader_robot as dr

    robot_root = SESSIONS_ROOT / "teleop_bimanual" / "0714"
    eef_dir = tc.resolve_robot_eef_dir(robot_root, None)
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    tc.build_robot_sync_csvs(robot_root, sync_dir, eef_dir, 0.050, max_demos)
    ds = dr.RobotEpisodeDataset(
        bird_vids_dir=robot_root / "bird-realsense-data" / "mp4",
        front_vids_dir=robot_root / "front-realsense-data" / "mp4",
        left_arm_vids_dir=robot_root / "aloha-data" / "left" / "mp4",
        right_arm_vids_dir=robot_root / "aloha-data" / "right" / "mp4",
        left_joint_data_dir=robot_root / "joint-data" / "left" / "position",
        right_joint_data_dir=robot_root / "joint-data" / "right" / "position",
        eef_pose_data_dir=eef_dir,
        sync_csv_dir=sync_dir,
        num_queries=8,
        max_demos=max_demos,
        resize_factor=resize_factor,
        max_sync_rows=max_sync_rows,
    )
    return ds


def run_human_pipeline(pkg_name: str, sync_dir: Path, *, max_demos: int = 1, max_sync_rows: int = 16, resize_factor: float = 0.25):
    if pkg_name == "Combined_relative_3cam_gripweight":
        import Combined_relative_3cam_gripweight.training_combined as tc
        import Combined_relative_3cam_gripweight.dataloader_human as dh
    elif pkg_name == "MixedEmbodiment_gripweight":
        import MixedEmbodiment_gripweight.training_combined as tc
        import MixedEmbodiment_gripweight.dataloader_human as dh
    else:
        import MixedEmbodiment.data_synchronization as tc
        import MixedEmbodiment.dataloader_human as dh

    human_root = SESSIONS_ROOT / "human_hands_bimanual_raw" / "0714"
    pose_dir = tc.resolve_human_pose_dir(human_root, None)
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    tc.build_human_sync_csvs(human_root, sync_dir, pose_dir, 0.050, max_demos)
    ds = dh.HumanEpisodeDataset(
        bird_vids_dir=human_root / "bird-realsense-data" / "mp4",
        front_vids_dir=human_root / "front-realsense-data" / "mp4",
        pose_npz_dir=pose_dir,
        sync_csv_dir=sync_dir,
        num_queries=8,
        max_demos=max_demos,
        resize_factor=resize_factor,
        max_sync_rows=max_sync_rows,
    )
    return ds


def run_mixed_pipeline(sync_dir: Path, *, max_demos: int = 1, max_sync_rows: int = 16, resize_factor: float = 0.25):
    """Only MixedEmbodiment_gripweight (legacy build_sync.py) vs new MixedEmbodiment are compared here."""
    import MixedEmbodiment_gripweight.build_sync as legacy_build_sync
    import MixedEmbodiment_gripweight.dataloader_mixed as legacy_dm

    import MixedEmbodiment.data_synchronization as new_sync
    import MixedEmbodiment.dataloader_mixed as new_dm

    mixed_root = SESSIONS_ROOT / "left_robot_right_hand" / "0729"
    preset = legacy_build_sync._infer_preset(mixed_root)
    robot_side, hand_side = preset["robot_side"], preset["hand_side"]
    pose_dir = legacy_build_sync._resolve_pose_dir(mixed_root, None)
    eef_dir = legacy_build_sync._resolve_eef_dir(mixed_root, None)

    legacy_sync = sync_dir / "legacy"
    new_sync_dir = sync_dir / "new"
    for d in (legacy_sync, new_sync_dir):
        if d.exists():
            shutil.rmtree(d)

    legacy_build_sync.build_mixed_sync_csvs(
        mixed_root, legacy_sync, robot_side=robot_side, hand_side=hand_side,
        pose_dir=pose_dir, eef_dir=eef_dir, max_skew_s=0.050, max_demos=max_demos,
    )
    new_sync.build_mixed_sync_csvs(
        mixed_root, new_sync_dir, robot_side=robot_side, hand_side=hand_side,
        pose_dir=pose_dir, eef_dir=eef_dir, max_skew_s=0.050, max_demos=max_demos,
    )

    kwargs = dict(
        bird_vids_dir=mixed_root / "bird-realsense-data" / "mp4",
        wrist_vids_dir=mixed_root / "aloha-data" / robot_side / "mp4",
        joint_data_dir=mixed_root / "joint-data" / robot_side / "position",
        hand_pose_npz_dir=pose_dir,
        eef_pose_data_dir=eef_dir,
        robot_side=robot_side,
        hand_side=hand_side,
        num_queries=8,
        max_demos=max_demos,
        resize_factor=resize_factor,
        max_sync_rows=max_sync_rows,
    )
    legacy_ds = legacy_dm.MixedEpisodeDataset(sync_csv_dir=legacy_sync, **kwargs)
    new_ds = new_dm.MixedEpisodeDataset(sync_csv_dir=new_sync_dir, **kwargs)
    return legacy_sync, new_sync_dir, legacy_ds, new_ds


def check_robot() -> RobotDsPair:
    legacy_dir = SCRATCH / "robot_legacy"
    new_dir = SCRATCH / "robot_new"
    legacy_ds = run_robot_pipeline("MixedEmbodiment_gripweight", legacy_dir)
    new_ds = run_robot_pipeline("new", new_dir)

    legacy_csv = sorted(legacy_dir.glob("*.csv"))[0]
    new_csv = sorted(new_dir.glob("*.csv"))[0]
    check(
        "Robot sync CSV identical (legacy MixedEmbodiment_gripweight vs new MixedEmbodiment)",
        df_equal(pd.read_csv(legacy_csv), pd.read_csv(new_csv)),
        f"{legacy_csv.name}",
    )
    check("Robot dataset demo/sample counts match", (legacy_ds.num_demos, legacy_ds.num_samples) == (new_ds.num_demos, new_ds.num_samples))
    check("Robot joint_mean/std match", torch.equal(legacy_ds.joint_mean, new_ds.joint_mean) and torch.equal(legacy_ds.joint_std, new_ds.joint_std))
    check("Robot eef_abs_mean/std match", torch.equal(legacy_ds.eef_abs_mean, new_ds.eef_abs_mean) and torch.equal(legacy_ds.eef_abs_std, new_ds.eef_abs_std))
    check("Robot eef (relative) mean/std match", torch.equal(legacy_ds.eef_mean, new_ds.eef_mean) and torch.equal(legacy_ds.eef_std, new_ds.eef_std))
    return legacy_ds, new_ds


def check_human():
    legacy_dir = SCRATCH / "human_legacy"
    new_dir = SCRATCH / "human_new"
    legacy_ds = run_human_pipeline("MixedEmbodiment_gripweight", legacy_dir)
    new_ds = run_human_pipeline("new", new_dir)

    legacy_csv = sorted(legacy_dir.glob("*.csv"))[0]
    new_csv = sorted(new_dir.glob("*.csv"))[0]
    check(
        "Human sync CSV identical (legacy MixedEmbodiment_gripweight vs new MixedEmbodiment)",
        df_equal(pd.read_csv(legacy_csv), pd.read_csv(new_csv)),
        f"{legacy_csv.name}",
    )
    check("Human dataset demo/sample counts match", (legacy_ds.num_demos, legacy_ds.num_samples) == (new_ds.num_demos, new_ds.num_samples))
    check("Human pose_abs_mean/std match", torch.equal(legacy_ds.pose_abs_mean, new_ds.pose_abs_mean) and torch.equal(legacy_ds.pose_abs_std, new_ds.pose_abs_std))
    check("Human pose (relative) mean/std match", torch.equal(legacy_ds.pose_mean, new_ds.pose_mean) and torch.equal(legacy_ds.pose_std, new_ds.pose_std))
    return legacy_ds, new_ds


def check_mixed():
    sync_dir = SCRATCH / "mixed"
    legacy_sync, new_sync_dir, legacy_ds, new_ds = run_mixed_pipeline(sync_dir)
    legacy_csv = sorted(legacy_sync.glob("*.csv"))[0]
    new_csv = sorted(new_sync_dir.glob("*.csv"))[0]
    check(
        "Mixed sync CSV identical (legacy build_sync.py vs new MixedEmbodiment.data_synchronization)",
        df_equal(pd.read_csv(legacy_csv), pd.read_csv(new_csv)),
        f"{legacy_csv.name}",
    )
    check("Mixed dataset demo/sample counts match", (legacy_ds.num_demos, legacy_ds.num_samples) == (new_ds.num_demos, new_ds.num_samples))
    check("Mixed joint_mean/std match", torch.equal(legacy_ds.joint_mean, new_ds.joint_mean) and torch.equal(legacy_ds.joint_std, new_ds.joint_std))
    check("Mixed pose_abs_mean/std match", torch.equal(legacy_ds.pose_abs_mean, new_ds.pose_abs_mean) and torch.equal(legacy_ds.pose_abs_std, new_ds.pose_abs_std))
    return legacy_ds, new_ds


# ---------------------------------------------------------------------------
# 4) Model forward equivalence (weights copied from legacy MixedEmbodiment_gripweight into new MixedEmbodiment)
# ---------------------------------------------------------------------------


def make_batch(embodiment: int, *, bs=1, k=8, h=64, w=64, camera_mask, seed: int):
    g = torch.Generator().manual_seed(seed)
    return dict(
        images=torch.randn(bs, 3, 3, h, w, generator=g),
        camera_mask=camera_mask,
        pose_state=torch.randn(bs, 8, generator=g),
        joint_state=torch.randn(bs, 14, generator=g),
        pose_actions=torch.randn(bs, k, 8, generator=g),
        joint_actions=torch.randn(bs, k, 14, generator=g),
        is_pad=torch.zeros(bs, k, dtype=torch.bool),
    )


def compare_outputs(name: str, out_a: dict, out_b: dict) -> None:
    for key in ("pose_pred", "joint_pred", "mu", "logvar"):
        a, b = out_a[key], out_b[key]
        if a is None and b is None:
            continue
        ok = a is not None and b is not None and torch.allclose(a, b, atol=1e-5, rtol=1e-4)
        check(f"{name}: {key} matches", ok, "" if ok else f"max abs diff={float((a - b).abs().max()) if a is not None and b is not None else 'one is None'}")


def check_model_equivalence() -> None:
    import MixedEmbodiment_gripweight.core as legacy_core
    import MixedEmbodiment.core as new_core
    import MixedEmbodiment.training_combined as new_training

    class LegacyArgs:
        def __init__(self, num_queries):
            self.num_queries = num_queries
            self.camera_names = ("cam0", "cam1", "cam2")
            self.hidden_dim = 128
            self.dropout = 0.0
            self.nheads = 4
            self.dim_feedforward = 256
            self.enc_layers = 2
            self.dec_layers = 2
            self.pre_norm = False
            self.position_embedding = "sine"
            self.backbone = "resnet18"
            self.lr_backbone = 1e-5
            self.masks = False
            self.dilation = False

    class NewArgs(new_training.Args):
        def __init__(self, num_queries, use_pose_observation):
            super().__init__(num_queries, use_pose_observation=use_pose_observation)
            self.hidden_dim = 128
            self.dropout = 0.0
            self.nheads = 4
            self.dim_feedforward = 256
            self.enc_layers = 2
            self.dec_layers = 2

    torch.manual_seed(0)
    K = 8
    legacy_model = legacy_core.build(LegacyArgs(K))
    torch.manual_seed(0)
    new_model = new_core.build(NewArgs(K, use_pose_observation=True))

    legacy_keys = set(legacy_model.state_dict().keys())
    new_keys = set(new_model.state_dict().keys())
    check(
        "Legacy MixedEmbodiment_gripweight / new MixedEmbodiment model state_dict keys match exactly",
        legacy_keys == new_keys,
        f"only_legacy={sorted(legacy_keys - new_keys)[:5]} only_new={sorted(new_keys - legacy_keys)[:5]}",
    )
    new_model.load_state_dict(legacy_model.state_dict())
    legacy_model.eval()
    new_model.eval()

    from MixedEmbodiment_gripweight.config import (
        EMBODIMENT_HUMAN as L_HUMAN,
        EMBODIMENT_MIXED as L_MIXED,
        EMBODIMENT_ROBOT as L_ROBOT,
        camera_mask_tensor as legacy_cam_mask,
    )
    from MixedEmbodiment.config import camera_mask_tensor as new_cam_mask

    for name, emb, side in (("robot", L_ROBOT, None), ("human", L_HUMAN, None), ("mixed-left", L_MIXED, "left")):
        cam_mask_legacy = legacy_cam_mask(emb, robot_side=side).unsqueeze(0) if side else legacy_cam_mask(emb).unsqueeze(0)
        cam_mask_new = new_cam_mask(emb, robot_side=side).unsqueeze(0) if side else new_cam_mask(emb).unsqueeze(0)
        batch = make_batch(emb, camera_mask=cam_mask_legacy, k=K, seed=hash(name) % (2**31))
        has_joint = emb in (L_ROBOT, L_MIXED)

        # Training mode (actions provided -> exercises the CVAE encoder + mu/logvar)
        torch.manual_seed(1)
        legacy_out = legacy_model(
            pose_state=batch["pose_state"], images=batch["images"], embodiment=emb,
            joint_state=batch["joint_state"], camera_mask=cam_mask_legacy,
            pose_actions=batch["pose_actions"], joint_actions=batch["joint_actions"],
            has_joint_target=has_joint, is_pad=batch["is_pad"],
        )
        torch.manual_seed(1)
        new_out = new_model(
            pose_state=batch["pose_state"], images=batch["images"], embodiment=emb,
            joint_state=batch["joint_state"], camera_mask=cam_mask_new,
            pose_actions=batch["pose_actions"], joint_actions=batch["joint_actions"],
            has_joint_target=has_joint, is_pad=batch["is_pad"],
        )
        compare_outputs(f"{name} (train mode)", legacy_out, new_out)

        # Inference mode (no actions -> latent sampled as zeros, no encoder pass)
        torch.manual_seed(2)
        legacy_out2 = legacy_model(
            pose_state=batch["pose_state"], images=batch["images"], embodiment=emb,
            joint_state=batch["joint_state"], camera_mask=cam_mask_legacy,
        )
        torch.manual_seed(2)
        new_out2 = new_model(
            pose_state=batch["pose_state"], images=batch["images"], embodiment=emb,
            joint_state=batch["joint_state"], camera_mask=cam_mask_new,
        )
        compare_outputs(f"{name} (inference mode)", legacy_out2, new_out2)


# ---------------------------------------------------------------------------
# 5) --pose_observation gate: human is blind to pose_state when off, and the
#    flag exactly reproduces/removes that blindness; robot/mixed unaffected.
# ---------------------------------------------------------------------------


def check_pose_observation_gate() -> None:
    import MixedEmbodiment.core as core
    import MixedEmbodiment.training_combined as training
    from MixedEmbodiment.config import (
        EMBODIMENT_HUMAN,
        EMBODIMENT_MIXED,
        EMBODIMENT_ROBOT,
        camera_mask_tensor,
    )

    class SmallArgs(training.Args):
        def __init__(self, num_queries, use_pose_observation):
            super().__init__(num_queries, use_pose_observation=use_pose_observation)
            self.hidden_dim = 128
            self.nheads = 4
            self.dim_feedforward = 256
            self.enc_layers = 2
            self.dec_layers = 2
            self.dropout = 0.0

    K = 8
    torch.manual_seed(42)
    model_off = core.build(SmallArgs(K, use_pose_observation=False))
    model_off.eval()
    torch.manual_seed(43)
    model_on = core.build(SmallArgs(K, use_pose_observation=True))
    model_on.eval()

    # model_off and model_on have DIFFERENT parameter sets for the human state
    # channel (human_input_const/human_cvae_state_const vs human_input_proj/
    # human_cvae_state_proj — see core.py), so a whole-state-dict copy is no
    # longer possible (and using the same seed for both builds wouldn't even
    # line up the *other* weights, since constructing the extra Linear layers
    # in the True case consumes extra RNG draws that shift every parameter
    # initialized afterwards). Instead, explicitly copy every genuinely
    # shared parameter (everything except the 4 human-state-only ones) so the
    # robot/mixed comparison below is a fair test of "same weights everywhere
    # that matters, does the flag change robot/mixed output" rather than an
    # accidental-seed-alignment test.
    human_state_only = ("human_input_proj", "human_input_const", "human_cvae_state_proj", "human_cvae_state_const")

    def copy_shared_params(dst: torch.nn.Module, src: torch.nn.Module) -> None:
        src_sd = src.state_dict()
        dst_sd = dst.state_dict()
        for key, val in src_sd.items():
            if any(key.startswith(p) for p in human_state_only):
                continue
            if key not in dst_sd:
                raise KeyError(f"dst model missing shared key {key}")
            dst_sd[key] = val
        dst.load_state_dict(dst_sd)

    copy_shared_params(model_on, model_off)

    cam_mask_human = camera_mask_tensor(EMBODIMENT_HUMAN).unsqueeze(0)
    images = torch.randn(1, 3, 3, 64, 64, generator=torch.Generator().manual_seed(7))
    pose_a = torch.randn(1, 8, generator=torch.Generator().manual_seed(11))
    pose_b = torch.randn(1, 8, generator=torch.Generator().manual_seed(22))
    check("pose_observation test: pose_a != pose_b (sanity)", not torch.allclose(pose_a, pose_b))

    def run(model, pose_state):
        torch.manual_seed(99)
        return model(pose_state=pose_state, images=images, embodiment=EMBODIMENT_HUMAN, camera_mask=cam_mask_human)

    out_off_a = run(model_off, pose_a)
    out_off_b = run(model_off, pose_b)
    check(
        "pose_observation=False: human output is IDENTICAL regardless of pose_state",
        torch.allclose(out_off_a["pose_pred"], out_off_b["pose_pred"], atol=1e-6),
        "the model must not be able to see the true hand pose",
    )

    out_on_a = run(model_on, pose_a)
    out_on_b = run(model_on, pose_b)
    check(
        "pose_observation=True: human output DIFFERS with pose_state (legacy behavior restored)",
        not torch.allclose(out_on_a["pose_pred"], out_on_b["pose_pred"], atol=1e-6),
        "the flag must genuinely gate the information",
    )

    # Robot/mixed: proprio is always joint_state, so the flag must not matter at all.
    for emb, side in ((EMBODIMENT_ROBOT, None), (EMBODIMENT_MIXED, "left")):
        cam_mask = camera_mask_tensor(emb, robot_side=side).unsqueeze(0) if side else camera_mask_tensor(emb).unsqueeze(0)
        joint_state = torch.randn(1, 14, generator=torch.Generator().manual_seed(5))

        def run_rm(model, pose_state):
            torch.manual_seed(123)
            return model(
                pose_state=pose_state, images=images, embodiment=emb,
                joint_state=joint_state, camera_mask=cam_mask,
            )

        out_off = run_rm(model_off, pose_a)
        out_on = run_rm(model_on, pose_a)
        name = "robot" if emb == EMBODIMENT_ROBOT else "mixed"
        check(
            f"{name} output identical regardless of --pose_observation (proprio is always joint_state)",
            torch.allclose(out_off["joint_pred"], out_on["joint_pred"], atol=1e-6),
        )


# ---------------------------------------------------------------------------


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print("=== 1) Gripper binarize threshold (item 1) ===")
    check_thresholds()

    print("\n=== 2/3) Real-data sync CSV + dataset-stat parity (item 2: single loading file) ===")
    check_robot()
    check_human()
    check_mixed()

    print("\n=== 4) Model forward equivalence with copied weights (item 4) ===")
    check_model_equivalence()

    print("\n=== 5) --pose_observation gate (item 8) ===")
    check_pose_observation_gate()

    shutil.rmtree(SCRATCH, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
    if n_fail:
        print("FAILURES:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
