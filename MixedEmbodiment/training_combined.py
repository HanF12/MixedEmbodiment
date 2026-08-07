"""
MixedEmbodiment ACT training — one CLI, three selectable modalities
(robot / human / mixed), replacing the separate Combined_relative_3cam_gripweight
and MixedEmbodiment_gripweight packages.

- True 3-camera architecture: [bird, left_wrist, right_wrist] (no front model slot)
- Pose = xyz+gripper only (8D); rot6d dropped at load
- Pose actions relative to chunk anchor; joint actions absolute
- Shared pose head (human primary + robot/mixed aux); robot/mixed-only joint head
- Modality routing (no embodiment embedding)
- Input is always a `sessions/<kind>/<date>/...` tree (--sessions_root); which of
  the three modalities to train on is chosen with --embodiments, and how many
  demos to use from each is independently controlled with
  --robot_max_demos / --human_max_demos / --mixed_max_demos
- Schedule: each epoch = one full pass over the longest active-modality loader
  (shorter ones recycled); batch/lr/recon/kl defaults match the two predecessor
  packages
- Gripper emphasis ("gripweight"): --gripper_loss_weight scales recon error on
  pose grip dims (3,7) and joint grip dims (6,13); default 5.0
- Human proprio: hard-removed (no adapter / no placeholder token); human predicts
  relative pose from images + CVAE latent only; see core.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MixedEmbodiment.config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRIPPER_LOSS_WEIGHT,
    DEFAULT_HAND_LAMBDA,
    DEFAULT_KL_WEIGHT,
    DEFAULT_LR,
    DEFAULT_MIXED_LAMBDA,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_QUERIES,
    DEFAULT_RECON_LOSS,
    DEFAULT_SESSIONS_ROOT,
    DEFAULT_WEIGHT_DECAY,
    EMBODIMENT_HUMAN,
    EMBODIMENT_MIXED,
    EMBODIMENT_NAMES,
    EMBODIMENT_ROBOT,
    GRIPPER_BINARIZE_THRESHOLD,
    GRIPPER_INDICES,
    MODEL_CAMERA_NAMES,
    NUM_CAMERAS,
    POSE_DIM,
    POSE_GRIP_INDICES,
    ROBOT_JOINT_DIM,
    build_run_metadata,
    default_run_name,
    discover_sessions_roots,
    gripper_dim_weights,
    save_run_metadata,
)
from MixedEmbodiment.core import build, kl_divergence  # noqa: E402
from MixedEmbodiment.data_synchronization import (  # noqa: E402
    build_human_sync_csvs,
    build_mixed_sync_csvs,
    build_robot_sync_csvs,
    infer_mixed_preset,
    resolve_human_pose_dir,
    resolve_robot_eef_dir,
)
from MixedEmbodiment.dataloader_human import HumanEpisodeDataset, collate_homogeneous  # noqa: E402
from MixedEmbodiment.dataloader_mixed import (  # noqa: E402
    ConcatMixedEpisodeDataset,
    MixedEpisodeDataset,
)
from MixedEmbodiment.dataloader_robot import RobotEpisodeDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MixedEmbodiment ACT training (robot / human / mixed, one CLI)"
    )
    p.add_argument(
        "--sessions_root",
        type=str,
        default=str(DEFAULT_SESSIONS_ROOT),
        help=(
            f"Top-level sessions tree (default: {DEFAULT_SESSIONS_ROOT}). The only "
            "supported input shape: discovers teleop_bimanual/<date> (robot), "
            "human_hands_bimanual_raw/<date> (human), and "
            "left_robot_right_hand|right_robot_left_hand/<date> (mixed) underneath it."
        ),
    )
    p.add_argument(
        "--embodiments",
        type=str,
        default="robot,human,mixed",
        help=(
            "Which of the three predecessor training regimes to run: 'robot' "
            "(matches Bimanual-3cam), 'robot,human' (matches "
            "Combined_relative_3cam_gripweight), or 'robot,human,mixed' (matches "
            "MixedEmbodiment_gripweight, the default). No other combination is "
            "accepted — robot is always the anchor, and mixed requires human. A "
            "requested modality that isn't found under --sessions_root is skipped "
            "with a warning; training fails only if zero modalities end up active."
        ),
    )
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument(
        "--weights_on_home",
        action="store_true",
        help=(
            "Save checkpoints under <package>/weights_home/ on the /home filesystem "
            "(real directory, not the weights/ symlink to /data). Ignored if --output_dir is set. "
            "Default without this flag: /data/hfang09/<package>/weights."
        ),
    )
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("-e", "--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    p.add_argument("-b", "--batch", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("-q", "--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
    p.add_argument("-g", "--gpu_number", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--max_skew_s", type=float, default=0.050, help="Sync tolerance in seconds.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--robot_max_demos", type=int, default=None,
        help="Use only the first N robot demos (sorted demo IDs). None = all.",
    )
    p.add_argument(
        "--human_max_demos", type=int, default=None,
        help="Use only the first N human demos (sorted demo IDs). None = all.",
    )
    p.add_argument(
        "--mixed_max_demos", type=int, default=None,
        help=(
            "Use only the first N mixed demos PER mixed session root (sorted demo IDs); "
            "e.g. with both left_robot_right_hand and right_robot_left_hand present, "
            "N applies to each (N=10 -> 20 total). None = all."
        ),
    )
    p.add_argument("--resize_factor", type=float, default=1.0, help="Scale frames before encoding (<=1.0).")
    p.add_argument(
        "--jpeg_in_ram", action="store_true",
        help="Store synced RGB frames as JPEG bytes in RAM instead of raw arrays.",
    )
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--max_sync_rows", type=int, default=None, help="Cap synced rows per demo (debug/smoke).")
    p.add_argument("--save_every_epochs", type=int, default=1000)
    p.add_argument(
        "--save_after_epochs",
        type=int,
        default=4000,
        help="Only start saving checkpoints (best + periodic) after this many epochs.",
    )
    p.add_argument("--pose_loss_weight", type=float, default=1.0)
    p.add_argument("--joint_loss_weight", type=float, default=1.0)
    p.add_argument(
        "--gripper_loss_weight",
        type=float,
        default=DEFAULT_GRIPPER_LOSS_WEIGHT,
        help=(
            "Gripweight: scale per-element recon error on gripper dims in pose+joint "
            f"actions (pose indices {POSE_GRIP_INDICES}, joint indices {GRIPPER_INDICES}). "
            f"Default {DEFAULT_GRIPPER_LOSS_WEIGHT}; use 1.0 for unweighted baseline."
        ),
    )
    p.add_argument("--kl_weight", type=float, default=DEFAULT_KL_WEIGHT)
    p.add_argument("--hand_lambda", type=float, default=DEFAULT_HAND_LAMBDA, help="Human loss scale (EgoMimic).")
    p.add_argument(
        "--mixed_lambda",
        type=float,
        default=None,
        help=(
            "Mixed loss scale. Default (omit): auto = len(mixed_loader) / steps_per_epoch, "
            "so each mixed batch's total weight per epoch matches one pass without recycling "
            "(recycle count * lambda = 1). Pass an explicit float to override "
            f"(legacy unweighted was {DEFAULT_MIXED_LAMBDA})."
        ),
    )
    p.add_argument(
        "--gripper_binarize_threshold",
        type=float,
        default=GRIPPER_BINARIZE_THRESHOLD,
        help=(
            "Shared threshold used to binarize robot/mixed gripper channels at load, "
            "for both EEF/hand pose (NPZ) and joint (NPY) grippers "
            f"(default {GRIPPER_BINARIZE_THRESHOLD}). Human hand poses are not re-binarized."
        ),
    )
    p.add_argument("--reconstruction_loss", type=str, choices=("mse", "l1"), default=DEFAULT_RECON_LOSS)
    p.add_argument(
        "--joint_modality_update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average active-modality losses then one optimizer step (default). "
        "Use --no-joint_modality_update for alternating single-modality steps.",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mixed-embodiment-3cam-act")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Sync, load one batch per active modality, run one train step each, then exit.",
    )
    return p.parse_args()


class Args:
    def __init__(self, num_queries: int):
        self.num_queries = num_queries
        self.camera_names = list(MODEL_CAMERA_NAMES)
        self.hidden_dim = 512
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 3200
        self.enc_layers = 4
        self.dec_layers = 7
        self.pre_norm = False
        self.position_embedding = "sine"
        self.backbone = "resnet18"
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False


# The only three supported combinations, matching the three predecessor folders
# exactly: Bimanual-3cam (robot alone — different model class, but this is the
# nearest equivalent run within this architecture), Combined_relative_3cam_gripweight
# (robot+human, always required together), and MixedEmbodiment_gripweight
# (robot+human+mixed). robot is always the anchor; mixed requires human. There is
# no human-alone, mixed-alone, or human+mixed-without-robot mode — those never
# existed in any predecessor and aren't validated against one.
VALID_EMBODIMENT_SETS = (
    frozenset({"robot"}),
    frozenset({"robot", "human"}),
    frozenset({"robot", "human", "mixed"}),
)


def parse_embodiments(spec: str) -> set[str]:
    requested = {tok.strip().lower() for tok in spec.split(",") if tok.strip()}
    unknown = requested - set(EMBODIMENT_NAMES)
    if unknown:
        raise ValueError(f"Unknown --embodiments entries {sorted(unknown)}; choose from {EMBODIMENT_NAMES}")
    if not requested:
        raise ValueError("--embodiments must name at least one of robot,human,mixed")
    if frozenset(requested) not in VALID_EMBODIMENT_SETS:
        raise ValueError(
            f"--embodiments={sorted(requested)} is not supported. Only robot / "
            "robot,human / robot,human,mixed are supported (matching Bimanual-3cam / "
            "Combined_relative_3cam_gripweight / MixedEmbodiment_gripweight respectively) "
            "— robot is always the anchor, and mixed requires human."
        )
    return requested


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    # persistent_workers keeps workers alive across iter() restarts (short robot
    # loader recycles often). pin_memory + non_blocking .to() overlap H2D with compute.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_homogeneous,
    )


def masked_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_pad: torch.Tensor,
    *,
    kind: str,
    dim_mask: torch.Tensor | None = None,
    dim_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over valid scalar elements; optional dim mask + per-dim weights (gripper boost)."""
    if kind == "mse":
        err = F.mse_loss(pred, target, reduction="none")
    elif kind == "l1":
        err = F.l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unknown reconstruction_loss={kind}")
    if dim_weights is not None:
        w = dim_weights.to(device=err.device, dtype=err.dtype).reshape(1, 1, -1)
        if w.shape[-1] != err.shape[-1]:
            raise ValueError(f"dim_weights last dim {w.shape[-1]} != action dim {err.shape[-1]}")
        err = err * w
    valid = (~is_pad).unsqueeze(-1).expand_as(err)
    if dim_mask is not None:
        dm = dim_mask.to(device=err.device, dtype=err.dtype)
        if dm.ndim == 1:
            dm = dm.view(1, 1, -1)
        elif dm.ndim == 2:
            dm = dm.unsqueeze(1)
        valid = valid & (dm > 0)
    if not valid.any():
        return pred.new_zeros(())
    return err[valid].mean()


def compute_batch_losses(
    model,
    batch,
    device,
    *,
    pose_w,
    joint_w,
    kl_w,
    recon_kind,
    hand_lambda,
    mixed_lambda,
    gripper_w,
) -> dict:
    """
    EgoMimic-style losses:
      human: hand_lambda * (pose_recon + kl)
      mixed: mixed_lambda * (pose_recon + joint_recon(+dim mask) + kl)
      robot: pose_recon + joint_recon + kl
    Gripper dims in pose/joint actions are scaled by gripper_w inside recon.
    """
    pose_state = batch["pose_state"].to(device, non_blocking=True)
    joint_state = batch["joint_state"].to(device, non_blocking=True)
    images = batch["images"].to(device, non_blocking=True)
    pose_actions = batch["pose_actions"].to(device, non_blocking=True)
    joint_actions = batch["joint_actions"].to(device, non_blocking=True)
    is_pad = batch["is_pad"].to(device, non_blocking=True)
    camera_mask = batch["camera_mask"].to(device, non_blocking=True)
    embodiment = int(batch["embodiment"])
    has_joint = bool(batch["has_joint_target"])
    joint_dim_mask = batch.get("joint_loss_mask")
    if joint_dim_mask is not None:
        joint_dim_mask = joint_dim_mask.to(device, non_blocking=True)

    out = model(
        pose_state=pose_state,
        images=images,
        embodiment=embodiment,
        joint_state=joint_state,
        camera_mask=camera_mask,
        pose_actions=pose_actions,
        joint_actions=joint_actions,
        has_joint_target=has_joint,
        is_pad=is_pad,
    )
    pose_pred = out["pose_pred"]
    joint_pred = out["joint_pred"]
    mu, logvar = out["mu"], out["logvar"]
    total_kld, *_ = kl_divergence(mu, logvar)
    kld = total_kld[0]

    pose_dim_w = torch.tensor(
        gripper_dim_weights(POSE_DIM, POSE_GRIP_INDICES, gripper_w), device=device, dtype=pose_pred.dtype
    )
    pose_loss = masked_recon_loss(pose_pred, pose_actions, is_pad, kind=recon_kind, dim_weights=pose_dim_w)
    if has_joint:
        if joint_pred is None:
            raise RuntimeError("Robot/mixed batch missing joint_pred")
        joint_dim_w = torch.tensor(
            gripper_dim_weights(ROBOT_JOINT_DIM, GRIPPER_INDICES, gripper_w), device=device, dtype=joint_pred.dtype
        )
        joint_loss = masked_recon_loss(
            joint_pred, joint_actions, is_pad, kind=recon_kind, dim_mask=joint_dim_mask, dim_weights=joint_dim_w
        )
        base = pose_w * pose_loss + joint_w * joint_loss + kl_w * kld
        loss = float(mixed_lambda) * base if embodiment == EMBODIMENT_MIXED else base
        joint_pred_shape = tuple(joint_pred.shape)
    else:
        joint_loss = pose_pred.new_zeros(())
        loss = float(hand_lambda) * (pose_w * pose_loss + kl_w * kld)
        joint_pred_shape = None

    name = {EMBODIMENT_ROBOT: "robot", EMBODIMENT_HUMAN: "human", EMBODIMENT_MIXED: "mixed"}.get(
        embodiment, f"emb{embodiment}"
    )
    return {
        "loss": loss,
        "pose_loss": pose_loss.detach(),
        "joint_loss": joint_loss.detach(),
        "kld": kld.detach(),
        "embodiment": name,
        "has_joint_target": has_joint,
        "pose_pred_shape": tuple(pose_pred.shape),
        "joint_pred_shape": joint_pred_shape,
    }


def train_step_single(model, optimizer, scaler, batch, device, use_amp: bool, loss_cfg: dict) -> dict:
    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=use_amp):
        stats = compute_batch_losses(model, batch, device, **loss_cfg)
        loss = stats["loss"]
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach().cpu()),
        "pose_loss": float(stats["pose_loss"].cpu()),
        "joint_loss": float(stats["joint_loss"].cpu()),
        "kld": float(stats["kld"].cpu()),
        "embodiment": stats["embodiment"],
        "pose_pred_shape": stats["pose_pred_shape"],
        "joint_pred_shape": stats["joint_pred_shape"],
    }


def train_step_joint_modalities(model, optimizer, scaler, batches: list, device, use_amp: bool, loss_cfg: dict) -> dict:
    """One optimizer step from the mean loss over whichever modality batches are active."""
    if not batches:
        raise ValueError("train_step_joint_modalities requires at least one batch")
    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=use_amp):
        stats_list = [compute_batch_losses(model, b, device, **loss_cfg) for b in batches]
        loss = sum(s["loss"] for s in stats_list) / float(len(stats_list))
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    out = {
        "loss": float(loss.detach().cpu()),
        "kld": float((sum(s["kld"] for s in stats_list) / len(stats_list)).detach().cpu()),
        "robot_pose_loss": 0.0,
        "robot_joint_loss": 0.0,
        "human_pose_loss": 0.0,
        "mixed_pose_loss": 0.0,
        "mixed_joint_loss": 0.0,
    }
    for s in stats_list:
        if s["embodiment"] == "robot":
            out["robot_pose_loss"] = float(s["pose_loss"].cpu())
            out["robot_joint_loss"] = float(s["joint_loss"].cpu())
        elif s["embodiment"] == "human":
            out["human_pose_loss"] = float(s["pose_loss"].cpu())
        elif s["embodiment"] == "mixed":
            out["mixed_pose_loss"] = float(s["pose_loss"].cpu())
            out["mixed_joint_loss"] = float(s["joint_loss"].cpu())
    return out


def next_batch_recycling(loader_iter, loader):
    """Fetch the next batch; restart the loader (recycle demos) if exhausted."""
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def resolve_mixed_lambda(
    mixed_loader,
    steps_per_epoch: int,
    explicit: float | None,
) -> tuple[float, bool]:
    """
    Resolve mixed_lambda.

    Auto (explicit is None): len(mixed_loader) / steps_per_epoch.
    With recycling, each mixed batch is visited ~steps/N_m times per epoch; scaling
    by N_m/steps makes the cumulative weight per mixed batch equal to 1 — same as
    seeing each mixed batch once with lambda=1 and no recycle.
    """
    if explicit is not None:
        return float(explicit), False
    if mixed_loader is None or steps_per_epoch <= 0:
        return float(DEFAULT_MIXED_LAMBDA), True
    return float(len(mixed_loader)) / float(steps_per_epoch), True


def steps_per_epoch_from_loaders(*loaders) -> int:
    lengths = [len(loader) for loader in loaders if loader is not None]
    return max(1, max(lengths)) if lengths else 1


def main() -> None:
    cli = parse_args()
    active = parse_embodiments(cli.embodiments)
    pkg = Path(__file__).resolve().parent

    sessions_root = Path(cli.sessions_root).expanduser().resolve()
    discovered = discover_sessions_roots(sessions_root)
    robot_root = discovered["robot_data_root"] if "robot" in active else None
    human_root = discovered["human_data_root"] if "human" in active else None
    mixed_roots = list(discovered["mixed_data_roots"]) if "mixed" in active else []

    if "robot" in active and robot_root is None:
        print(f"WARNING: --embodiments requested 'robot' but no teleop_bimanual/<date> found under {sessions_root}; skipping.")
    if "human" in active and human_root is None:
        print(f"WARNING: --embodiments requested 'human' but no human_hands_bimanual_raw/<date> found under {sessions_root}; skipping.")
    if "mixed" in active and not mixed_roots:
        print(f"WARNING: --embodiments requested 'mixed' but no left/right_robot_*_hand/<date> found under {sessions_root}; skipping.")

    if robot_root is None and human_root is None and not mixed_roots:
        raise ValueError(
            f"No data found under --sessions_root={sessions_root} for --embodiments={sorted(active)}. "
            "Expected teleop_bimanual/<date>, human_hands_bimanual_raw/<date>, and/or "
            "left_robot_right_hand|right_robot_left_hand/<date>."
        )

    if cli.output_dir:
        weights_root = Path(cli.output_dir).expanduser().resolve()
    elif cli.weights_on_home:
        # Real dir on /home — do not use pkg/weights (that is a symlink to /data).
        weights_root = (pkg / "weights_home").resolve()
    else:
        weights_root = Path(f"/data/hfang09/{pkg.name}/weights")
    run_name = cli.run_name or default_run_name()
    output_dir = weights_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints -> {output_dir}")

    # Per-run sync tree avoids mushing CSVs across sessions_root / prior runs.
    sync_root = pkg / "m-synced-csvs" / run_name

    gripper_thr = float(cli.gripper_binarize_threshold)
    # mixed_lambda filled after loaders exist (may be auto from recycle ratio).
    loss_cfg = {
        "pose_w": float(cli.pose_loss_weight),
        "joint_w": float(cli.joint_loss_weight),
        "kl_w": float(cli.kl_weight),
        "recon_kind": str(cli.reconstruction_loss),
        "hand_lambda": float(cli.hand_lambda),
        "mixed_lambda": float(DEFAULT_MIXED_LAMBDA),
        "gripper_w": float(cli.gripper_loss_weight),
    }

    robot_ds = None
    human_ds = None
    mixed_ds = None
    robot_eef_dir = None
    human_pose_dir = None
    mixed_sync_dirs: list[Path] = []

    if robot_root is not None:
        robot_eef_dir = resolve_robot_eef_dir(robot_root)
        robot_sync = sync_root / f"{robot_root.name}_robot"
        build_robot_sync_csvs(robot_root, robot_sync, robot_eef_dir, cli.max_skew_s, cli.robot_max_demos)
        robot_ds = RobotEpisodeDataset(
            bird_vids_dir=robot_root / "bird-realsense-data" / "mp4",
            front_vids_dir=robot_root / "front-realsense-data" / "mp4",
            left_arm_vids_dir=robot_root / "aloha-data" / "left" / "mp4",
            right_arm_vids_dir=robot_root / "aloha-data" / "right" / "mp4",
            left_joint_data_dir=robot_root / "joint-data" / "left" / "position",
            right_joint_data_dir=robot_root / "joint-data" / "right" / "position",
            eef_pose_data_dir=robot_eef_dir,
            sync_csv_dir=robot_sync,
            num_queries=cli.num_queries,
            max_demos=cli.robot_max_demos,
            resize_factor=cli.resize_factor,
            max_sync_rows=cli.max_sync_rows,
            jpeg_in_ram=cli.jpeg_in_ram,
            jpeg_quality=cli.jpeg_quality,
            npz_gripper_binarize_threshold=gripper_thr,
            npy_gripper_binarize_threshold=gripper_thr,
        )
        np.savez(
            output_dir / "normalization_stats_robot.npz",
            joint_mean=robot_ds.joint_mean.numpy(),
            joint_std=robot_ds.joint_std.numpy(),
            eef_pose_mean=robot_ds.eef_mean.numpy(),
            eef_pose_std=robot_ds.eef_std.numpy(),
            eef_pose_rel_mean=robot_ds.eef_rel_mean.numpy(),
            eef_pose_rel_std=robot_ds.eef_rel_std.numpy(),
            eef_pose_abs_mean=robot_ds.eef_abs_mean.numpy(),
            eef_pose_abs_std=robot_ds.eef_abs_std.numpy(),
            pose_action_space=np.asarray("relative_to_chunk_anchor"),
            qpos_mean=robot_ds.joint_mean.numpy(),
            qpos_std=robot_ds.joint_std.numpy(),
        )

    if human_root is not None:
        human_pose_dir = resolve_human_pose_dir(human_root)
        human_sync = sync_root / f"{human_root.name}_human"
        build_human_sync_csvs(human_root, human_sync, human_pose_dir, cli.max_skew_s, cli.human_max_demos)
        human_ds = HumanEpisodeDataset(
            bird_vids_dir=human_root / "bird-realsense-data" / "mp4",
            front_vids_dir=human_root / "front-realsense-data" / "mp4",
            pose_npz_dir=human_pose_dir,
            sync_csv_dir=human_sync,
            num_queries=cli.num_queries,
            max_demos=cli.human_max_demos,
            resize_factor=cli.resize_factor,
            max_sync_rows=cli.max_sync_rows,
            jpeg_in_ram=cli.jpeg_in_ram,
            jpeg_quality=cli.jpeg_quality,
        )
        np.savez(
            output_dir / "normalization_stats_human.npz",
            pose_mean=human_ds.pose_mean.numpy(),
            pose_std=human_ds.pose_std.numpy(),
            pose_rel_mean=human_ds.pose_rel_mean.numpy(),
            pose_rel_std=human_ds.pose_rel_std.numpy(),
            pose_abs_mean=human_ds.pose_abs_mean.numpy(),
            pose_abs_std=human_ds.pose_abs_std.numpy(),
            pose_action_space=np.asarray("relative_to_chunk_anchor"),
        )

    if mixed_roots:
        mixed_children: list[MixedEpisodeDataset] = []
        for i, mixed_root in enumerate(mixed_roots):
            preset = infer_mixed_preset(mixed_root)
            if preset is None:
                raise ValueError(
                    f"Could not infer robot/hand sides for mixed root {mixed_root}. "
                    "Folder path must contain left_robot_right_hand or right_robot_left_hand."
                )
            robot_side, hand_side = preset["robot_side"], preset["hand_side"]
            mixed_sync = sync_root / f"{mixed_root.parent.name}_{mixed_root.name}_mixed"
            mixed_sync_dirs.append(mixed_sync)
            mixed_pose_dir = resolve_human_pose_dir(mixed_root)
            mixed_eef_dir = resolve_robot_eef_dir(mixed_root)
            print(
                f"Mixed child[{i}]: root={mixed_root} robot_side={robot_side} "
                f"hand_side={hand_side} pose_dir={mixed_pose_dir} eef_dir={mixed_eef_dir}"
            )
            build_mixed_sync_csvs(
                mixed_root,
                mixed_sync,
                robot_side=robot_side,
                hand_side=hand_side,
                pose_dir=mixed_pose_dir,
                eef_dir=mixed_eef_dir,
                max_skew_s=cli.max_skew_s,
                max_demos=cli.mixed_max_demos,
            )
            child = MixedEpisodeDataset(
                bird_vids_dir=mixed_root / "bird-realsense-data" / "mp4",
                wrist_vids_dir=mixed_root / "aloha-data" / robot_side / "mp4",
                joint_data_dir=mixed_root / "joint-data" / robot_side / "position",
                hand_pose_npz_dir=mixed_pose_dir,
                eef_pose_data_dir=mixed_eef_dir,
                sync_csv_dir=mixed_sync,
                robot_side=robot_side,
                hand_side=hand_side,
                num_queries=cli.num_queries,
                max_demos=cli.mixed_max_demos,
                resize_factor=cli.resize_factor,
                max_sync_rows=cli.max_sync_rows,
                jpeg_in_ram=cli.jpeg_in_ram,
                jpeg_quality=cli.jpeg_quality,
                npz_gripper_binarize_threshold=gripper_thr,
                npy_gripper_binarize_threshold=gripper_thr,
            )
            mixed_children.append(child)

        mixed_ds = mixed_children[0] if len(mixed_children) == 1 else ConcatMixedEpisodeDataset(mixed_children)
        np.savez(
            output_dir / "normalization_stats_mixed.npz",
            joint_mean=mixed_ds.joint_mean.numpy(),
            joint_std=mixed_ds.joint_std.numpy(),
            pose_mean=mixed_ds.pose_mean.numpy(),
            pose_std=mixed_ds.pose_std.numpy(),
            pose_rel_mean=mixed_ds.pose_rel_mean.numpy(),
            pose_rel_std=mixed_ds.pose_rel_std.numpy(),
            pose_abs_mean=mixed_ds.pose_abs_mean.numpy(),
            pose_abs_std=mixed_ds.pose_abs_std.numpy(),
            mixed_roots=np.asarray([str(p) for p in mixed_roots]),
            pose_action_space=np.asarray("relative_to_chunk_anchor"),
        )

    meta = build_run_metadata(
        embodiments=sorted(active),
        sessions_root=sessions_root,
        robot_data_root=robot_root,
        human_data_root=human_root,
        mixed_data_roots=mixed_roots,
        num_queries=cli.num_queries,
        max_skew_s=cli.max_skew_s,
        gripper_binarize_threshold=gripper_thr,
        pose_loss_weight=cli.pose_loss_weight,
        joint_loss_weight=cli.joint_loss_weight,
        gripper_loss_weight=cli.gripper_loss_weight,
        kl_weight=cli.kl_weight,
        reconstruction_loss=cli.reconstruction_loss,
        joint_modality_update=cli.joint_modality_update,
        hand_lambda=cli.hand_lambda,
        mixed_lambda=DEFAULT_MIXED_LAMBDA,  # overwritten below after auto-resolve
        num_epochs=cli.epochs,
        batch_size=cli.batch,
        lr=cli.lr,
    )
    meta["jpeg_in_ram"] = bool(cli.jpeg_in_ram)
    meta["jpeg_quality"] = int(cli.jpeg_quality) if cli.jpeg_in_ram else None

    device = torch.device("cpu" if cli.cpu or not torch.cuda.is_available() else f"cuda:{cli.gpu_number}")
    print(f"Using device: {device}; active embodiments: {sorted(active)}; human_proprio=hard_removed")
    model = build(Args(cli.num_queries)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cli.lr, weight_decay=cli.weight_decay)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    robot_loader = make_loader(robot_ds, cli.batch, cli.num_workers, shuffle=True) if robot_ds is not None else None
    human_loader = make_loader(human_ds, cli.batch, cli.num_workers, shuffle=True) if human_ds is not None else None
    mixed_loader = make_loader(mixed_ds, cli.batch, cli.num_workers, shuffle=True) if mixed_ds is not None else None
    active_loaders = [l for l in (robot_loader, human_loader, mixed_loader) if l is not None]
    steps_per_epoch = steps_per_epoch_from_loaders(*active_loaders)
    mixed_lambda, mixed_lambda_auto = resolve_mixed_lambda(mixed_loader, steps_per_epoch, cli.mixed_lambda)
    cli.mixed_lambda = mixed_lambda
    loss_cfg["mixed_lambda"] = mixed_lambda
    meta["mixed_lambda"] = mixed_lambda
    meta["mixed_lambda_auto"] = bool(mixed_lambda_auto)
    meta["steps_per_epoch"] = int(steps_per_epoch)
    if robot_loader is not None:
        meta["robot_loader_batches"] = len(robot_loader)
        meta["robot_num_demos"] = len(robot_ds)
    if human_loader is not None:
        meta["human_loader_batches"] = len(human_loader)
        meta["human_num_demos"] = len(human_ds)
    if mixed_loader is not None:
        meta["mixed_loader_batches"] = len(mixed_loader)
        meta["mixed_num_demos"] = len(mixed_ds)
        if mixed_lambda_auto:
            print(
                f"mixed_lambda auto={mixed_lambda:.6g} "
                f"(= mixed_batches/steps_per_epoch = {len(mixed_loader)}/{steps_per_epoch}; "
                f"equiv. to one pass over mixed without recycling)"
            )
        else:
            print(f"mixed_lambda={mixed_lambda:.6g} (explicit)")
    save_run_metadata(output_dir, meta)

    if cli.dry_run:
        print("--- MixedEmbodiment dry run ---")
        for name, loader in (("robot", robot_loader), ("human", human_loader), ("mixed", mixed_loader)):
            if loader is None:
                continue
            batch = next(iter(loader))
            stats = train_step_single(model, optimizer, scaler, batch, device, use_amp, loss_cfg)
            print(f"{name.capitalize()} step OK: {stats}")
        assert hasattr(model, "pose_action_head")
        print(
            f"Shared pose_action_head id={id(model.pose_action_head)} "
            f"joint_action_head id={id(model.joint_action_head)}"
        )
        print("Dry run passed.")
        return

    wandb_run = None
    if cli.wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed")
        wandb_run = wandb.init(
            project=cli.wandb_project,
            entity=cli.wandb_entity,
            name=cli.wandb_run_name,
            mode=cli.wandb_mode,
            config=vars(cli),
        )

    best = float("inf")
    step = 0
    print(
        f"Training schedule: {cli.epochs} epochs x {steps_per_epoch} steps/epoch "
        f"(active={sorted(active)}; batches robot={len(robot_loader) if robot_loader else 0}, "
        f"human={len(human_loader) if human_loader else 0}, mixed={len(mixed_loader) if mixed_loader else 0}; "
        f"shorter modalities recycled), batch={cli.batch}, lr={cli.lr}, recon={cli.reconstruction_loss}, "
        f"kl_weight={cli.kl_weight}, hand_lambda={cli.hand_lambda}, mixed_lambda={cli.mixed_lambda}, K={cli.num_queries}"
    )
    for epoch in range(cli.epochs):
        model.train()
        robot_iter = iter(robot_loader) if robot_loader is not None else None
        human_iter = iter(human_loader) if human_loader is not None else None
        mixed_iter = iter(mixed_loader) if mixed_loader is not None else None
        running = {
            "robot_pose": 0.0, "robot_joint": 0.0, "human_pose": 0.0, "mixed_pose": 0.0, "mixed_joint": 0.0,
            "n_r": 0, "n_h": 0, "n_m": 0, "n_joint": 0,
        }
        loop = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{cli.epochs}", unit="step")
        for _ in loop:
            batches = []
            if robot_loader is not None:
                b, robot_iter = next_batch_recycling(robot_iter, robot_loader)
                batches.append(b)
            if human_loader is not None:
                b, human_iter = next_batch_recycling(human_iter, human_loader)
                batches.append(b)
            if mixed_loader is not None:
                b, mixed_iter = next_batch_recycling(mixed_iter, mixed_loader)
                batches.append(b)

            if cli.joint_modality_update:
                stats = train_step_joint_modalities(model, optimizer, scaler, batches, device, use_amp, loss_cfg)
                step += 1
                running["robot_pose"] += stats["robot_pose_loss"]
                running["robot_joint"] += stats["robot_joint_loss"]
                running["human_pose"] += stats["human_pose_loss"]
                running["mixed_pose"] += stats["mixed_pose_loss"]
                running["mixed_joint"] += stats["mixed_joint_loss"]
                if robot_loader is not None:
                    running["n_r"] += 1
                    running["n_joint"] += 1
                if human_loader is not None:
                    running["n_h"] += 1
                if mixed_loader is not None:
                    running["n_m"] += 1
                if wandb_run is not None:
                    log = {"train/kl_loss": stats["kld"], "train/total_loss": stats["loss"]}
                    if robot_loader is not None:
                        log["train/robot_pose_loss"] = stats["robot_pose_loss"]
                        log["train/robot_joint_loss"] = stats["robot_joint_loss"]
                    if human_loader is not None:
                        log["train/human_pose_loss"] = stats["human_pose_loss"]
                    if mixed_loader is not None:
                        log["train/mixed_pose_loss"] = stats["mixed_pose_loss"]
                        log["train/mixed_joint_loss"] = stats["mixed_joint_loss"]
                    wandb.log(log, step=step)
            else:
                for batch in batches:
                    stats = train_step_single(model, optimizer, scaler, batch, device, use_amp, loss_cfg)
                    step += 1
                    if stats["embodiment"] == "robot":
                        running["robot_pose"] += stats["pose_loss"]
                        running["robot_joint"] += stats["joint_loss"]
                        running["n_r"] += 1
                        running["n_joint"] += 1
                    elif stats["embodiment"] == "human":
                        running["human_pose"] += stats["pose_loss"]
                        running["n_h"] += 1
                    else:
                        running["mixed_pose"] += stats["pose_loss"]
                        running["mixed_joint"] += stats["joint_loss"]
                        running["n_m"] += 1
                        running["n_joint"] += 1
                    if wandb_run is not None:
                        prefix = stats["embodiment"]
                        log = {f"train/{prefix}_pose_loss": stats["pose_loss"], "train/kl_loss": stats["kld"], "train/total_loss": stats["loss"]}
                        if prefix in ("robot", "mixed"):
                            log[f"train/{prefix}_joint_loss"] = stats["joint_loss"]
                        wandb.log(log, step=step)

            loop.set_postfix(
                rp=running["robot_pose"] / max(1, running["n_r"]),
                rj=running["robot_joint"] / max(1, running["n_joint"]),
                hp=running["human_pose"] / max(1, running["n_h"]),
                mp=running["mixed_pose"] / max(1, running["n_m"]),
            )

        avg_rp = running["robot_pose"] / max(1, running["n_r"])
        avg_rj = running["robot_joint"] / max(1, running["n_joint"])
        avg_hp = running["human_pose"] / max(1, running["n_h"])
        avg_mp = running["mixed_pose"] / max(1, running["n_m"])
        parts = []
        if robot_loader is not None:
            parts += [avg_rp, avg_rj]
        if human_loader is not None:
            parts.append(avg_hp)
        if mixed_loader is not None:
            parts.append(avg_mp)
        avg = sum(parts) / float(len(parts))
        print(
            f"Epoch {epoch+1}: robot_pose={avg_rp:.6f} robot_joint={avg_rj:.6f} "
            f"human_pose={avg_hp:.6f} mixed_pose={avg_mp:.6f} avg={avg:.6f}"
        )
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_robot_pose": avg_rp,
                    "train/epoch_robot_joint": avg_rj,
                    "train/epoch_human_pose": avg_hp,
                    "train/epoch_mixed_pose": avg_mp,
                    "train/epoch_avg": avg,
                },
                step=step,
            )

        if (epoch + 1) >= cli.save_after_epochs:
            if avg < best:
                best = avg
                torch.save(model.state_dict(), output_dir / "mixed_act_best.pth")
                print(f"Saved new best -> {output_dir / 'mixed_act_best.pth'}")
            if cli.save_every_epochs > 0 and (epoch + 1) % cli.save_every_epochs == 0:
                torch.save(model.state_dict(), output_dir / f"mixed_act_epoch_{epoch+1}.pth")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
