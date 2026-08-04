"""Shared path / video helpers for MixedEmbodiment dataloaders."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def prune_orphan_sync_csvs(sync_dir: Path, keep_ids: set[str] | list[str]) -> int:
    """Remove CSVs whose stems are not in keep_ids (stale leftovers from another data tree)."""
    if not sync_dir.is_dir():
        return 0
    keep = set(keep_ids)
    removed = 0
    for p in sync_dir.glob("*.csv"):
        if p.stem not in keep:
            p.unlink()
            removed += 1
    if removed:
        print(f"Pruned {removed} orphan sync CSV(s) from {sync_dir}")
    return removed


def demo_id_from_hash_filename(path: str | Path) -> str:
    """video_recording_...#demo_id.ext -> demo_id"""
    name = Path(path).name
    if "#" not in name:
        raise ValueError(f"Expected '#' in filename: {name}")
    return name.split("#", 1)[1].rsplit(".", 1)[0]


def demo_id_from_joint_npy(path: str | Path, prefix: str = "joint_position_") -> str:
    name = Path(path).name
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected joint file name: {name}")
    return name[len(prefix) :].rsplit(".", 1)[0]


def demo_id_from_pose_npz(path: str | Path) -> str:
    """
    ...#human_hands_bimanual_raw_YYYYMMDDHHMMSS_wilor_rgbd_pose.npz
      -> human_hands_bimanual_raw_YYYYMMDDHHMMSS
    """
    stem = demo_id_from_hash_filename(path)
    for suffix in ("_wilor_rgbd_pose", "_mediapipe_rgbd_pose", "_hand_pose"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def demo_id_from_robot_eef_npz(path: str | Path) -> str:
    """
    teleop_bimanual_YYYYMMDDHHMMSS_arm_fk_pose_targetframe_commonframe.npz
      -> teleop_bimanual_YYYYMMDDHHMMSS

    Also accepts hashed names if present.
    """
    name = Path(path).name
    stem = name.split("#", 1)[1].rsplit(".", 1)[0] if "#" in name else Path(path).stem
    for suffix in (
        "_arm_fk_pose_targetframe_commonframe",
        "_arm_fk_pose_targetframe",
        "_arm_fk_pose_commonframe",
        "_arm_fk_pose",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def relative_pose_chunk(
    absolute_steps: list[torch.Tensor],
    *,
    anchor: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Convert absolute pose steps to deltas vs the chunk anchor (first observation).

      relative[k] = absolute[k] - anchor

    The anchor must be the pose at the first observation in the chunk (t), not the
    previous timestep. With chunk start at t, relative[0] is zeros.
    """
    anchor_t = torch.as_tensor(anchor, dtype=torch.float32).reshape(-1)
    out: list[torch.Tensor] = []
    for step in absolute_steps:
        step_t = torch.as_tensor(step, dtype=torch.float32).reshape(-1)
        if step_t.numel() != anchor_t.numel():
            raise ValueError(
                f"Pose dim mismatch for relative chunk: step={step_t.numel()} anchor={anchor_t.numel()}"
            )
        out.append(step_t - anchor_t)
    return out


def absolute_pose_from_relative(
    relative_actions: torch.Tensor | np.ndarray,
    *,
    anchor: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """
    Inference: re-anchor a relative pose chunk to the first observation pose.

      absolute[k] = anchor + relative[k]

    relative_actions: [K, D] or [D]
    anchor: [D] absolute pose at the chunk's first observation.
    """
    rel = torch.as_tensor(relative_actions, dtype=torch.float32)
    anc = torch.as_tensor(anchor, dtype=torch.float32).reshape(-1)
    if rel.ndim == 1:
        if rel.numel() != anc.numel():
            raise ValueError(f"Pose dim mismatch: rel={rel.numel()} anchor={anc.numel()}")
        return rel + anc
    if rel.ndim != 2 or rel.shape[-1] != anc.numel():
        raise ValueError(f"Expected relative [K,{anc.numel()}], got {tuple(rel.shape)}")
    return rel + anc.unsqueeze(0)


def compute_relative_pose_stats(
    pose_data: list[torch.Tensor],
    *,
    demo_start_idx: list[int],
    demo_lengths: list[int],
    num_queries: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean/std of chunk-anchored relative pose targets over the full dataset.

    For every valid start index t inside each episode:
      delta[k] = pose[t+k] - pose[t]  for k = 0 .. min(K, remaining)-1
    """
    deltas: list[torch.Tensor] = []
    k = int(num_queries)
    for ep_start, ep_len in zip(demo_start_idx, demo_lengths):
        demo_end = int(ep_start) + int(ep_len)
        for t_off in range(int(ep_len)):
            sample_idx = int(ep_start) + t_off
            anchor = torch.as_tensor(pose_data[sample_idx], dtype=torch.float32).reshape(-1)
            slice_end = min(demo_end, sample_idx + k)
            for j in range(sample_idx, slice_end):
                step = torch.as_tensor(pose_data[j], dtype=torch.float32).reshape(-1)
                deltas.append(step - anchor)
    if not deltas:
        raise RuntimeError("No relative pose samples available to compute stats")
    all_d = torch.stack(deltas, dim=0)
    mean = all_d.mean(dim=0)
    std = all_d.std(dim=0).clamp(min=1e-2)
    return mean, std


def normalize_future_chunk(
    raw_steps: list[torch.Tensor],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    num_queries: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize valid future steps, then pad with exact zeros in normalized space.

    Caller is responsible for the action-space of raw_steps (absolute or relative).
    For relative pose targets, pass deltas vs the chunk anchor so that
    actions[0] is near zero before normalization.

    Returns:
      actions: [K, D] float32
      is_pad:  [K] bool
    """
    mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
    dim = int(mean.numel())
    valid = [((torch.as_tensor(s, dtype=torch.float32).reshape(-1) - mean) / std) for s in raw_steps]
    raw_len = len(valid)
    pad_len = int(num_queries) - raw_len
    if pad_len < 0:
        raise ValueError(f"raw_steps longer than num_queries ({raw_len} > {num_queries})")
    if pad_len > 0:
        valid.extend([torch.zeros(dim, dtype=torch.float32)] * pad_len)
    is_pad = torch.zeros(int(num_queries), dtype=torch.bool)
    if pad_len > 0:
        is_pad[-pad_len:] = True
    actions = torch.stack(valid, dim=0)
    return actions, is_pad


def denormalize_actions(
    norm_actions: torch.Tensor | np.ndarray,
    *,
    mean: torch.Tensor | np.ndarray,
    std: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Invert z-normalization: raw = norm * std + mean."""
    x = torch.as_tensor(norm_actions, dtype=torch.float32)
    m = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
    s = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
    return x * s + m


def denormalize_relative_pose_to_absolute(
    norm_relative_actions: torch.Tensor | np.ndarray,
    *,
    action_mean: torch.Tensor | np.ndarray,
    action_std: torch.Tensor | np.ndarray,
    anchor_pose: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """
    Inference helper: denormalize relative pose preds and anchor to the first
    observation in the chunk.

      abs[k] = anchor_pose + (pred_norm[k] * std + mean)
    """
    rel = denormalize_actions(norm_relative_actions, mean=action_mean, std=action_std)
    return absolute_pose_from_relative(rel, anchor=anchor_pose)


def index_paths_by_demo_id(paths: list[Path], id_fn) -> dict[str, Path]:
    return {id_fn(p): p for p in paths}


def build_image_transform(transform: str = "resnet_normalization"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if "resnet_normalization" in transform:
        return transforms.Compose([transforms.ToTensor(), normalize])
    return transforms.ToTensor()


def load_video_frames(
    video_path: Path,
    *,
    resize_factor: float = 1.0,
    label: str = "Video",
) -> list[np.ndarray]:
    """Load full RGB video as list of HxWx3 uint8 arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    read_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        read_count += 1
        if resize_factor != 1.0:
            h, w = frame_bgr.shape[:2]
            frame_bgr = cv2.resize(
                frame_bgr,
                (int(w * resize_factor), int(h * resize_factor)),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"  - {label} '{video_path.name}' -> read {read_count}/{total} frames")
    return frames


def encode_frame_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    """Compress an HxWx3 RGB uint8 frame to JPEG bytes (kept in RAM)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 RGB, got shape={getattr(rgb, 'shape', None)} dtype={getattr(rgb, 'dtype', None)}")
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def decode_frame_jpeg(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode JPEG bytes back to HxWx3 RGB uint8."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("JPEG decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def store_frame(
    rgb: np.ndarray,
    *,
    jpeg_in_ram: bool,
    jpeg_quality: int = 90,
) -> np.ndarray | bytes:
    """Optionally JPEG-compress a frame before keeping it in the dataset list."""
    if jpeg_in_ram:
        return encode_frame_jpeg(rgb, quality=jpeg_quality)
    return rgb


def load_frame(stored: np.ndarray | bytes | bytearray | memoryview) -> np.ndarray:
    """Load a stored frame (raw RGB array or JPEG bytes) to HxWx3 uint8 RGB."""
    if isinstance(stored, (bytes, bytearray, memoryview)):
        return decode_frame_jpeg(stored)
    return stored


def frame_nbytes(stored: np.ndarray | bytes | bytearray | memoryview) -> int:
    if isinstance(stored, (bytes, bytearray, memoryview)):
        return len(stored)
    return int(np.asarray(stored).nbytes)


def zero_rgb_like(ref: np.ndarray) -> np.ndarray:
    return np.zeros_like(ref)


def pad_state(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad trailing zeros so robot [14] can sit in a [20] buffer if needed."""
    state = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
    if state.numel() == target_dim:
        return state
    if state.numel() > target_dim:
        raise ValueError(f"Cannot pad state of dim {state.numel()} down to {target_dim}")
    out = torch.zeros(target_dim, dtype=torch.float32)
    out[: state.numel()] = state
    return out
