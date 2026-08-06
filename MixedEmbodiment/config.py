"""
MixedEmbodiment ACT constants and helpers.

One package, three CLI-selectable modalities (robot / human / mixed), replacing
the separate Combined_relative_3cam_gripweight and MixedEmbodiment_gripweight
folders. Architecture and conventions are unchanged from those two packages:

True 3-camera architecture — front is NOT a model slot.
Camera slot order is fixed as:
  [bird, left_wrist, right_wrist]  -> model cams cam0..cam2

Shared pose (human hands / robot EEF / mixed hand+EEF), EgoMimic-style shared head:
  8D = left 4 + right 4
  each side: xyz(3) + gripper/open(1)
  Source NPZ layout is still [2, 10] = xyz(3)+rot6d(6)+grip(1); rot is dropped at load.

Robot joints:
  14D = left 7 + right 7  (still absolute; mixed zeros the inactive arm)
  Robot/mixed EEF pose grippers (NPZ) and joint grippers (NPY) are binarized at
  load with a shared threshold (--gripper_binarize_threshold).

Robot / mixed proprio: joints only [14] (robot_input_proj + robot CVAE)
Human proprio: none. The human embodiment has no proprio observation at all —
  no adapter, no placeholder token — and must predict relative pose purely
  from video + the CVAE latent (see core.py module docstring).

Third modality (mixed):
  left_robot_right_hand + right_robot_left_hand are one modality (ConcatDataset).
  Needs bird + active wrist + active-arm joints + hand-side pose + robot-side EEF.
  Bimanual NPZs still store both sides; only the correct side slots are read.

Input layout: always the `sessions/<kind>/<date>/...` tree (see
discover_sessions_roots). There is no support for arbitrary explicit data roots —
every run points at one `--sessions_root` and the three modalities are discovered
underneath it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_NUM_QUERIES = 45  # keep Combined horizon (EgoMimic uses 100)

# EgoMimic-matched training defaults (same as Combined_relative_3cam / MixedEmbodiment)
DEFAULT_NUM_EPOCHS = 10000
# One epoch = one full pass over the longest active-modality loader
# (max over whichever of robot/human/mixed are active); shorter ones are recycled.
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 2e-5  # linearly scaled from 1e-5 @ batch 8
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_KL_WEIGHT = 10.0
DEFAULT_HAND_LAMBDA = 1.0
# Tunable mixed modality loss scale (also exposed as --mixed_lambda).
# L_mixed = mixed_lambda * (pose + joint + KL) before equal mean over modalities.
DEFAULT_MIXED_LAMBDA = 0.1
DEFAULT_RECON_LOSS = "l1"
# Gripper emphasis (the "gripweight" feature): scales per-element recon error on
# gripper dims of both the shared pose head and the robot/mixed joint head.
DEFAULT_GRIPPER_LOSS_WEIGHT = 5.0

# Temporal convention for action chunks:
# Chunk is built starting at the current observation index t.
# Pose actions are relative to that first observation:
#   pose_actions[k] = pose[t+k] - pose[t]
# Joint actions remain absolute: joint_actions[k] = joints[t+k].
ACTION_CHUNK_STARTS_AT_CURRENT = True
POSE_ACTION_RELATIVE_TO_CHUNK_ANCHOR = True

# No embodiment ever receives absolute pose as an observation. Robot/mixed
# proprio is always joint_state; human has no proprio observation at all —
# only images + the CVAE latent drive the prediction, and the model is scored
# purely on how well it predicts the *relative* pose chunk. See core.py's
# module docstring for the (now removed) legacy --pose_observation history.

# --- Robot joints ---
ROBOT_JOINT_DIM = 14
JOINT_DIM_PER_ARM = 7
LEFT_ARM_SLICE = slice(0, JOINT_DIM_PER_ARM)
RIGHT_ARM_SLICE = slice(JOINT_DIM_PER_ARM, ROBOT_JOINT_DIM)
GRIPPER_INDICES = (JOINT_DIM_PER_ARM - 1, ROBOT_JOINT_DIM - 1)
ROBOT_STATE_DIM = ROBOT_JOINT_DIM  # proprio = joints only (EgoMimic)

# --- Raw NPZ pose layout (before dropping rotation) ---
RAW_HAND_DIM = 10  # xyz(3) + rot6d(6) + grip(1)
RAW_XYZ_SLICE = slice(0, 3)
RAW_GRIP_INDEX = 9

# --- Shared pose used in training (xyz + gripper only) ---
POSE_DIM_PER_SIDE = 4  # xyz(3) + grip(1)
POSE_DIM = 8  # left 4 + right 4
# Gripper dims in flattened [8] pose: left grip, right grip
POSE_GRIP_INDICES = (POSE_DIM_PER_SIDE - 1, POSE_DIM - 1)  # (3, 7)

# Single shared gripper-binarize threshold (--gripper_binarize_threshold).
# Historically NPZ (EEF/hand pose) and NPY (joint) thresholds were configurable
# independently, but every run in this repo used the same value for both, so
# the CLI now exposes one flag. The two constants below stay independently
# addressable in code (e.g. for a future divergence) but default to the same
# CLI-provided value.
GRIPPER_BINARIZE_THRESHOLD = 0.5
ROBOT_NPZ_GRIPPER_BINARIZE_THRESHOLD = GRIPPER_BINARIZE_THRESHOLD  # EEF/hand pose from NPZ
ROBOT_NPY_GRIPPER_BINARIZE_THRESHOLD = GRIPPER_BINARIZE_THRESHOLD  # joint grippers from NPY
HUMAN_STATE_DIM = POSE_DIM
HUMAN_PROPRIO_DIM = POSE_DIM
ROBOT_PROPRIO_DIM = ROBOT_JOINT_DIM  # EgoMimic: joints only

# --- Shared camera layout (NO front slot in the model) ---
CAMERA_ORDER = ("bird", "left_wrist", "right_wrist")
MODEL_CAMERA_NAMES = ("cam0", "cam1", "cam2")
NUM_CAMERAS = 3

ROBOT_CAMERA_MASK = (1, 1, 1)  # bird + both wrists
HUMAN_CAMERA_MASK = (1, 0, 0)  # bird only; wrist slots zeroed
# Mixed: bird + active wrist; inactive wrist masked / zeroed.
MIXED_CAMERA_MASK_LEFT_ROBOT = (1, 1, 0)   # bird + left wrist
MIXED_CAMERA_MASK_RIGHT_ROBOT = (1, 0, 1)  # bird + right wrist

EMBODIMENT_ROBOT = 0
EMBODIMENT_HUMAN = 1
EMBODIMENT_MIXED = 2
EMBODIMENT_NAMES = ("robot", "human", "mixed")
EMBODIMENT_IDS = {"robot": EMBODIMENT_ROBOT, "human": EMBODIMENT_HUMAN, "mixed": EMBODIMENT_MIXED}

# Sync CSVs still include front_index for timestamp alignment (not a model cam).
ROBOT_SYNC_INDEX_COLUMNS = (
    "left_joint_index",
    "right_joint_index",
    "left_index",
    "right_index",
    "bird_index",
    "front_index",
    "eef_pose_index",
)

HUMAN_SYNC_INDEX_COLUMNS = (
    "bird_index",
    "front_index",
    "pose_index",
)

MIXED_SYNC_INDEX_COLUMNS = (
    "bird_index",
    "front_index",
    "wrist_index",
    "joint_index",
    "hand_pose_index",
    "eef_pose_index",
)

ROBOT_EEF_RELDIR = Path("joint-data") / "combined_npz_commonframe"
ROBOT_EEF_COORD_FRAME = (
    "arm_fk_targetframe_commonframe: left/right arm-base FK poses "
    "transformed into a shared midline frame. "
    "Training uses xyz+gripper only (rotation dropped)."
)

# Default pose NPZ layout under each modality root (sessions layout).
HUMAN_POSE_RELDIR = Path("bird-realsense-data") / "combined_npz_targetframe"

# Top-level sessions tree. This is the ONLY supported data-input shape
# (item 7): every run points at one --sessions_root and robot / human / mixed
# roots are discovered underneath it.
DEFAULT_SESSIONS_ROOT = Path("sessions")
ROBOT_SESSION_KIND = "teleop_bimanual"
HUMAN_SESSION_KIND = "human_hands_bimanual_raw"
MIXED_SESSION_KINDS = ("left_robot_right_hand", "right_robot_left_hand")

MAX_STATE_DIM = max(POSE_DIM, ROBOT_JOINT_DIM)


def _looks_like_session_date_root(path: Path) -> bool:
    """A date folder is usable if it has bird video or joint-data."""
    if not path.is_dir():
        return False
    return (path / "bird-realsense-data").is_dir() or (path / "joint-data").is_dir()


def list_session_date_roots(sessions_root: str | Path, kind: str) -> list[Path]:
    """List date roots under sessions/<kind>/<date> (sorted)."""
    base = Path(sessions_root).expanduser().resolve() / kind
    if not base.is_dir():
        return []
    return [p for p in sorted(base.iterdir()) if _looks_like_session_date_root(p)]


def discover_sessions_roots(
    sessions_root: str | Path = DEFAULT_SESSIONS_ROOT,
) -> dict[str, Path | list[Path] | None]:
    """
    Resolve robot / human / mixed date roots from a single sessions/ tree.

    Robot / human: latest date folder under each kind (lexicographic max).
    Mixed: all date folders under left_robot_right_hand and right_robot_left_hand
    (one Concat modality).
    """
    root = Path(sessions_root).expanduser().resolve()
    robot_dates = list_session_date_roots(root, ROBOT_SESSION_KIND)
    human_dates = list_session_date_roots(root, HUMAN_SESSION_KIND)
    mixed_roots: list[Path] = []
    for kind in MIXED_SESSION_KINDS:
        mixed_roots.extend(list_session_date_roots(root, kind))
    return {
        "sessions_root": root,
        "robot_data_root": robot_dates[-1] if robot_dates else None,
        "human_data_root": human_dates[-1] if human_dates else None,
        "mixed_data_roots": mixed_roots,
    }


def default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_camera_names(camera_names: list[str] | tuple[str, ...]) -> None:
    if tuple(camera_names) != MODEL_CAMERA_NAMES:
        raise ValueError(
            f"Expected camera_names={MODEL_CAMERA_NAMES}, got {tuple(camera_names)}. "
            "MixedEmbodiment uses fixed slots [bird, left_wrist, right_wrist]."
        )


def concat_bimanual_joints(
    left_step: np.ndarray | torch.Tensor,
    right_step: np.ndarray | torch.Tensor,
    *,
    rec_id: str = "",
    binarize_grippers: bool = True,
    gripper_threshold: float = ROBOT_NPY_GRIPPER_BINARIZE_THRESHOLD,
) -> torch.Tensor:
    """
    Robot qpos: [7 left, 7 right] -> [14].

    By default, gripper joints (last channel per arm; flattened indices 6 and 13)
    are binarized with the shared gripper_binarize_threshold.
    """
    left_tensor = torch.as_tensor(left_step, dtype=torch.float32).reshape(-1)
    right_tensor = torch.as_tensor(right_step, dtype=torch.float32).reshape(-1)
    if left_tensor.numel() < JOINT_DIM_PER_ARM or right_tensor.numel() < JOINT_DIM_PER_ARM:
        raise ValueError(
            f"Expected >= {JOINT_DIM_PER_ARM} joints/arm for {rec_id or 'sample'}, "
            f"got left={left_tensor.numel()} right={right_tensor.numel()}"
        )
    out = torch.cat([left_tensor[:JOINT_DIM_PER_ARM], right_tensor[:JOINT_DIM_PER_ARM]], dim=0)
    if binarize_grippers:
        thr = float(gripper_threshold)
        for idx in GRIPPER_INDICES:
            out[idx] = 1.0 if float(out[idx]) >= thr else 0.0
    return out


def flatten_bimanual_pose(pose_t: np.ndarray | torch.Tensor, *, rec_id: str = "") -> torch.Tensor:
    """
    Extract xyz+gripper from one bimanual pose timestep and flatten.

    Expected input shape: [2, 10] (raw NPZ layout).
    Output shape: [8] = [left_xyz(3), left_grip(1), right_xyz(3), right_grip(1)].
    Rotation (indices 3:9) is discarded.
    """
    pose = torch.as_tensor(pose_t, dtype=torch.float32)
    if pose.shape != (2, RAW_HAND_DIM):
        raise ValueError(
            f"Expected pose timestep shape (2, {RAW_HAND_DIM}) for {rec_id or 'sample'}, "
            f"got {tuple(pose.shape)}"
        )
    left = torch.cat([pose[0, RAW_XYZ_SLICE], pose[0, RAW_GRIP_INDEX : RAW_GRIP_INDEX + 1]], dim=0)
    right = torch.cat([pose[1, RAW_XYZ_SLICE], pose[1, RAW_GRIP_INDEX : RAW_GRIP_INDEX + 1]], dim=0)
    flat = torch.cat([left, right], dim=0)
    if flat.numel() != POSE_DIM:
        raise ValueError(f"Internal pose flatten error: got {flat.numel()} != {POSE_DIM}")
    if not torch.isfinite(flat).all():
        raise ValueError(f"Non-finite pose values for {rec_id or 'sample'}")
    return flat


def binarize_flat_pose_grippers(
    flat_pose: torch.Tensor | np.ndarray,
    *,
    threshold: float = ROBOT_NPZ_GRIPPER_BINARIZE_THRESHOLD,
) -> torch.Tensor:
    """
    Binarize left/right gripper dims in a flattened [8] pose.

    grip -> 1.0 if grip >= threshold else 0.0. XYZ dims unchanged.
    """
    flat = torch.as_tensor(flat_pose, dtype=torch.float32).clone()
    if flat.numel() != POSE_DIM:
        raise ValueError(f"Expected flat pose dim {POSE_DIM}, got {flat.numel()}")
    thr = float(threshold)
    for idx in POSE_GRIP_INDICES:
        flat[idx] = 1.0 if float(flat[idx]) >= thr else 0.0
    return flat


# Backward-compatible alias
flatten_hand_pose = flatten_bimanual_pose


def stack_camera_tensors(
    bird_frame: torch.Tensor,
    left_frame: torch.Tensor,
    right_frame: torch.Tensor,
) -> torch.Tensor:
    """Stack to [3, C, H, W] in CAMERA_ORDER (bird, left_wrist, right_wrist)."""
    return torch.stack([bird_frame, left_frame, right_frame], dim=0)


def extract_side_pose4(pose_t: np.ndarray | torch.Tensor, slot: int, *, rec_id: str = "") -> torch.Tensor:
    """Extract xyz+gripper [4] for one side from raw [2, 10] pose."""
    pose = torch.as_tensor(pose_t, dtype=torch.float32)
    if pose.shape != (2, RAW_HAND_DIM):
        raise ValueError(
            f"Expected pose timestep shape (2, {RAW_HAND_DIM}) for {rec_id or 'sample'}, "
            f"got {tuple(pose.shape)}"
        )
    if int(slot) not in (0, 1):
        raise ValueError(f"slot must be 0 or 1, got {slot}")
    out = torch.cat(
        [pose[int(slot), RAW_XYZ_SLICE], pose[int(slot), RAW_GRIP_INDEX : RAW_GRIP_INDEX + 1]],
        dim=0,
    )
    if not torch.isfinite(out).all():
        raise ValueError(f"Non-finite side pose (slot={slot}) for {rec_id or 'sample'}")
    return out


def compose_mixed_flat_pose(
    hand_t: np.ndarray | torch.Tensor,
    eef_t: np.ndarray | torch.Tensor,
    *,
    hand_slot: int,
    robot_slot: int,
    binarize_eef_gripper: bool = True,
    eef_gripper_threshold: float = ROBOT_NPZ_GRIPPER_BINARIZE_THRESHOLD,
    rec_id: str = "",
) -> torch.Tensor:
    """
    Build [8] pose from two bimanual NPZ timesteps that each store BOTH sides,
    but only ONE side is valid per file:

      hand NPZ  [2,10]: read hand_slot only (other hand is NaN / invalid)
      EEF NPZ   [2,10]: read robot_slot only (other arm is NaN / invalid)

    Result fills both halves of the flat [8] (hand side + robot side). There is
    no leftover "invalid" pose dim in the training target — the invalid NPZ
    sides are never copied in.
    """
    if int(hand_slot) == int(robot_slot):
        raise ValueError(f"hand_slot and robot_slot must differ, got {hand_slot}")
    hand4 = extract_side_pose4(hand_t, int(hand_slot), rec_id=rec_id)
    eef4 = extract_side_pose4(eef_t, int(robot_slot), rec_id=rec_id).clone()
    if binarize_eef_gripper:
        eef4[3] = 1.0 if float(eef4[3]) >= float(eef_gripper_threshold) else 0.0
    flat = torch.zeros(POSE_DIM, dtype=torch.float32)
    hs = int(hand_slot) * POSE_DIM_PER_SIDE
    rs = int(robot_slot) * POSE_DIM_PER_SIDE
    flat[hs : hs + POSE_DIM_PER_SIDE] = hand4
    flat[rs : rs + POSE_DIM_PER_SIDE] = eef4
    return flat


def single_arm_joint_vector(
    arm_qpos: np.ndarray | torch.Tensor,
    *,
    robot_side: str,
    rec_id: str = "",
    binarize_grippers: bool = True,
    gripper_threshold: float = ROBOT_NPY_GRIPPER_BINARIZE_THRESHOLD,
) -> torch.Tensor:
    """
    Pack one arm's joints into a [14] vector; inactive arm is zeros.

    Mixed sessions store joints as one-sided arrays [T, 7] under
    joint-data/<robot_side>/position (the other arm folder does not exist).
    """
    arm = torch.as_tensor(arm_qpos, dtype=torch.float32).reshape(-1)
    if arm.numel() < JOINT_DIM_PER_ARM:
        raise ValueError(
            f"Expected >= {JOINT_DIM_PER_ARM} joints for {rec_id or 'sample'}, got {arm.numel()}"
        )
    out = torch.zeros(ROBOT_JOINT_DIM, dtype=torch.float32)
    if robot_side == "left":
        out[LEFT_ARM_SLICE] = arm[:JOINT_DIM_PER_ARM]
        grip_idx = GRIPPER_INDICES[0]
    elif robot_side == "right":
        out[RIGHT_ARM_SLICE] = arm[:JOINT_DIM_PER_ARM]
        grip_idx = GRIPPER_INDICES[1]
    else:
        raise ValueError(f"robot_side must be 'left' or 'right', got {robot_side!r}")
    if binarize_grippers:
        thr = float(gripper_threshold)
        out[grip_idx] = 1.0 if float(out[grip_idx]) >= thr else 0.0
    return out


def joint_loss_mask_for_side(robot_side: str | None = None, *, full: bool = False) -> torch.Tensor:
    """Float mask [14]; 1 = supervise that joint dim."""
    if full or robot_side is None:
        return torch.ones(ROBOT_JOINT_DIM, dtype=torch.float32)
    mask = torch.zeros(ROBOT_JOINT_DIM, dtype=torch.float32)
    if robot_side == "left":
        mask[LEFT_ARM_SLICE] = 1.0
    elif robot_side == "right":
        mask[RIGHT_ARM_SLICE] = 1.0
    else:
        raise ValueError(f"robot_side must be 'left' or 'right', got {robot_side!r}")
    return mask


def camera_mask_tensor(
    embodiment: int,
    *,
    robot_side: str | None = None,
) -> torch.Tensor:
    if int(embodiment) == EMBODIMENT_ROBOT:
        mask = list(ROBOT_CAMERA_MASK)
    elif int(embodiment) == EMBODIMENT_HUMAN:
        mask = list(HUMAN_CAMERA_MASK)
    elif int(embodiment) == EMBODIMENT_MIXED:
        if robot_side == "left":
            mask = list(MIXED_CAMERA_MASK_LEFT_ROBOT)
        elif robot_side == "right":
            mask = list(MIXED_CAMERA_MASK_RIGHT_ROBOT)
        else:
            raise ValueError("EMBODIMENT_MIXED requires robot_side='left' or 'right'")
    else:
        raise ValueError(f"Unknown embodiment id {embodiment}")
    return torch.tensor(mask, dtype=torch.float32)


def gripper_dim_weights(action_dim: int, grip_indices: tuple[int, ...], weight: float) -> list[float]:
    """Per-dim multipliers for recon loss; gripper indices get `weight`, others 1.0."""
    w = float(weight)
    out = [1.0] * int(action_dim)
    for idx in grip_indices:
        i = int(idx)
        if not (0 <= i < len(out)):
            raise IndexError(f"gripper index {i} out of range for action_dim={action_dim}")
        out[i] = w
    return out


def build_run_metadata(
    *,
    embodiments: list[str],
    sessions_root: str | Path,
    robot_data_root: str | Path | None,
    human_data_root: str | Path | None,
    mixed_data_roots: list[str | Path] | tuple[str | Path, ...] | None,
    num_queries: int,
    max_skew_s: float,
    gripper_binarize_threshold: float,
    pose_loss_weight: float = 1.0,
    joint_loss_weight: float = 1.0,
    gripper_loss_weight: float = DEFAULT_GRIPPER_LOSS_WEIGHT,
    kl_weight: float = DEFAULT_KL_WEIGHT,
    reconstruction_loss: str = DEFAULT_RECON_LOSS,
    joint_modality_update: bool = True,
    hand_lambda: float = DEFAULT_HAND_LAMBDA,
    mixed_lambda: float = DEFAULT_MIXED_LAMBDA,
    steps_per_epoch: int | None = None,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
) -> dict[str, Any]:
    mixed_roots_list = [str(p) for p in (mixed_data_roots or [])]
    return {
        "variant": "mixed-embodiment-relative-3cam-gripweight",
        "embodiments": list(embodiments),
        "pose_layout": "xyz+gripper only (8D); rot6d dropped at load",
        "pose_action_space": "relative_to_chunk_anchor",
        "pose_observation": False,
        "pose_observation_semantics": (
            "No embodiment ever receives absolute pose as an observation. Human "
            "has no proprio observation at all (no adapter, no placeholder "
            "token) and predicts the relative pose chunk from images + latent "
            "only. Robot/mixed proprio is always joint_state."
        ),
        "embodiment_cue": "modality routing (separate projs/heads/cams), no embedding token",
        "mixed_policy": (
            "third modality = Concat(left_robot_right_hand, right_robot_left_hand); "
            "proprio/CVAE use robot joint path; pose [8]=hand slot + robot EEF slot "
            "(only correct bimanual NPZ sides loaded); joint loss masked to active arm; "
            "camera_mask keeps bird + active wrist"
        ),
        "camera_order": list(CAMERA_ORDER),
        "model_camera_names": list(MODEL_CAMERA_NAMES),
        "num_cameras": NUM_CAMERAS,
        "front_camera_in_model": False,
        "front_camera_for_sync_only": True,
        "pose_dim": POSE_DIM,
        "robot_joint_dim": ROBOT_JOINT_DIM,
        "robot_proprio_dim": ROBOT_PROPRIO_DIM,
        "human_proprio_dim": HUMAN_PROPRIO_DIM,
        "robot_camera_mask": list(ROBOT_CAMERA_MASK),
        "human_camera_mask": list(HUMAN_CAMERA_MASK),
        "mixed_camera_mask_left_robot": list(MIXED_CAMERA_MASK_LEFT_ROBOT),
        "mixed_camera_mask_right_robot": list(MIXED_CAMERA_MASK_RIGHT_ROBOT),
        "num_queries": int(num_queries),
        "action_chunk_starts_at_current": bool(ACTION_CHUNK_STARTS_AT_CURRENT),
        "pose_action_relative_to_chunk_anchor": bool(POSE_ACTION_RELATIVE_TO_CHUNK_ANCHOR),
        "action_chunk_convention": (
            "pose_actions[k] = pose[t+k] - pose[t] (delta vs first observation in the chunk); "
            "pose_actions[0] is zeros; at inference absolute_pose[k] = anchor_pose[t] + denorm(pred[k]). "
            "joint_actions[k] remain absolute joints at t+k."
        ),
        "sessions_root": str(sessions_root),
        "robot_data_root": str(robot_data_root) if robot_data_root is not None else None,
        "human_data_root": str(human_data_root) if human_data_root is not None else None,
        "mixed_data_roots": mixed_roots_list,
        "robot_eef_reldir": str(ROBOT_EEF_RELDIR),
        "human_pose_reldir": str(HUMAN_POSE_RELDIR),
        "gripper_binarize_threshold": float(gripper_binarize_threshold),
        "gripper_semantics": (
            "shared last per-side dim: human open/close left as-is; "
            f"robot/mixed EEF pose (NPZ) and joint (NPY) grippers binarized at load "
            f"with threshold {float(gripper_binarize_threshold)}; "
            "relative pose deltas include gripper dims"
        ),
        "max_skew_s": float(max_skew_s),
        "pose_loss_weight": float(pose_loss_weight),
        "joint_loss_weight": float(joint_loss_weight),
        "gripper_loss_weight": float(gripper_loss_weight),
        "gripper_loss_weighting": (
            "per-element recon error on gripper dims (pose indices "
            f"{POSE_GRIP_INDICES}, joint indices {GRIPPER_INDICES}) "
            f"scaled by gripper_loss_weight={float(gripper_loss_weight)}; "
            "non-gripper dims remain weight 1.0"
        ),
        "kl_weight": float(kl_weight),
        "hand_lambda": float(hand_lambda),
        "mixed_lambda": float(mixed_lambda),
        "reconstruction_loss": str(reconstruction_loss),
        "joint_modality_update": bool(joint_modality_update),
        "num_epochs": int(num_epochs),
        "epoch_length": (
            "max over active-modality loader lengths; shorter modalities recycled; "
            "loss = mean over active modalities per step"
        ),
        "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch is not None else None,
        "batch_size": int(batch_size),
        "lr": float(lr),
    }


def save_run_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata_path


def load_run_metadata(run_dir: str | Path) -> dict[str, Any] | None:
    metadata_path = Path(run_dir) / "run_metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def validate_run_metadata(metadata: dict[str, Any], *, num_queries: int | None = None) -> None:
    if tuple(metadata.get("camera_order", [])) != CAMERA_ORDER:
        raise ValueError(
            f"Saved camera_order {metadata.get('camera_order')} does not match expected {CAMERA_ORDER}"
        )
    if tuple(metadata.get("model_camera_names", [])) != MODEL_CAMERA_NAMES:
        raise ValueError(
            "Saved model camera names do not match the fixed MixedEmbodiment camera order"
        )
    saved_joint_dim = metadata.get("robot_joint_dim", metadata.get("robot_state_dim", -1))
    if int(saved_joint_dim) != ROBOT_JOINT_DIM:
        raise ValueError(
            f"Saved robot_joint_dim {saved_joint_dim} does not match {ROBOT_JOINT_DIM}"
        )
    if num_queries is not None and int(metadata.get("num_queries", -1)) != int(num_queries):
        raise ValueError(
            f"Saved num_queries {metadata.get('num_queries')} does not match CLI {num_queries}"
        )
