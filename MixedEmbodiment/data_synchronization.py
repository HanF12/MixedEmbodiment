"""
Timestamp synchronization for MixedEmbodiment training — the single "loading"
file for all three modalities.

Low-level per-demo synchronizers (nearest-window multi-stream alignment):
  1) synchronize_robot_bimanual  — joints + 4 cameras + EEF pose timeline
  2) synchronize_human_hands    — bird + front + hand-pose NPZ timeline
  3) synchronize_mixed_hand_robot — one wrist + one arm + hand NPZ + EEF NPZ

High-level directory-discovery / orchestration (previously split between
training_combined.py and the standalone MixedEmbodiment_gripweight/build_sync.py
module — folded in here so there is one place that turns a sessions/<kind>/<date>
folder into sync CSVs):
  4) resolve_robot_eef_dir / resolve_human_pose_dir — locate the pose NPZ dirs
  5) infer_mixed_preset — robot_side/hand_side from a left_robot_right_hand /
     right_robot_left_hand folder name
  6) build_robot_sync_csvs / build_human_sync_csvs / build_mixed_sync_csvs —
     discover per-demo files under a data root and write one sync CSV per demo

Orientation / valid_rot is never used for frame validity (training drops rot6d).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from MixedEmbodiment.config import HUMAN_POSE_RELDIR, ROBOT_EEF_RELDIR  # noqa: E402
from MixedEmbodiment.dataloader_utils import (  # noqa: E402
    demo_id_from_hash_filename,
    demo_id_from_joint_npy,
    demo_id_from_pose_npz,
    demo_id_from_robot_eef_npz,
    prune_orphan_sync_csvs,
)


def _assert_sorted(name: str, values: np.ndarray, *, strict: bool = True) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {values.shape}")
    if len(values) <= 1:
        return
    diffs = np.diff(values)
    ok = np.all(diffs > 0) if strict else np.all(diffs >= 0)
    if not ok:
        raise AssertionError(
            f"{name} timestamps are not {'strictly ' if strict else ''}non-decreasing "
            f"(min_diff={float(diffs.min())})"
        )


def _assert_finite_1d(name: str, values: np.ndarray) -> np.ndarray:
    ts = np.asarray(values, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        raise ValueError(f"{name} timestamps are empty")
    if not np.isfinite(ts).all():
        bad = int((~np.isfinite(ts)).sum())
        raise ValueError(f"{name} has {bad} non-finite timestamps")
    return ts


def make_unique_increasing_timeline(timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse duplicate timestamps (common in bag-rate pose NPZs).

    Returns:
      unique_ts: [M] strictly increasing timestamps
      src_index: [M] original index into the full pose array for each unique_ts
    """
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        return ts, np.zeros((0,), dtype=np.int64)
    unique_ts, first_idx = np.unique(ts, return_index=True)
    # np.unique sorts by value; for time keys that is chronological order.
    _assert_sorted("unique_pose_ts", unique_ts, strict=True)
    return unique_ts.astype(np.float64), first_idx.astype(np.int64)


def _sync_streams(
    stream_names: list[str],
    stream_ts: list[np.ndarray],
    out_csv: str | Path,
    *,
    index_columns: list[str],
    max_skew_s: float,
    debug: bool,
    label: str,
    write_csv: bool = True,
) -> pd.DataFrame:
    """Generic multi-stream nearest-window synchronizer (same algorithm as Bimanual)."""
    n = len(stream_ts)
    if n != len(stream_names) or n != len(index_columns):
        raise ValueError("stream_names, stream_ts, and index_columns must have the same length")

    for name, ts in zip(stream_names, stream_ts):
        _assert_sorted(name, ts, strict=True)

    idxs = [0] * n
    lengths = [len(ts) for ts in stream_ts]
    rows = []
    master = 0

    while all(idx < size for idx, size in zip(idxs, lengths)):
        pivot = max(float(stream_ts[s][idxs[s]]) for s in range(n))

        for s in range(n):
            while idxs[s] < lengths[s] and pivot - float(stream_ts[s][idxs[s]]) > max_skew_s:
                idxs[s] += 1

        if not all(idx < size for idx, size in zip(idxs, lengths)):
            break

        values = [float(stream_ts[s][idxs[s]]) for s in range(n)]
        t_min = min(values)
        t_max = max(values)

        if t_max - t_min <= max_skew_s:
            rows.append((master, *idxs, *values, t_max - t_min))
            master += 1
            for s in range(n):
                idxs[s] += 1
        else:
            earliest = int(np.argmin(values))
            idxs[earliest] += 1

    time_cols = [f"{c.replace('_index', '')}_time" for c in index_columns]
    all_cols = ["master_index", *index_columns, *time_cols, "time_diff"]
    df = pd.DataFrame(rows, columns=all_cols)
    if not debug:
        df = df[["master_index", *index_columns]]

    if write_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Synced {len(df)} {label} -> {out_path} (debug={debug})")
    return df


def xyz_gripper_valid_mask(
    *,
    valid_pos: np.ndarray | None = None,
    valid_open: np.ndarray | None = None,
    n_frames: int | None = None,
    required_slots: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Per-frame mask: True iff required slots have valid xyz + open/gripper.

    - ``required_slots`` defaults to both hands/arms ``(0, 1)``.
      Pass ``(0,)`` or ``(1,)`` for single-hand / single-arm demos.
    - Orientation / ``valid_rot`` is never consulted.
    - Each validity array is expected as [T, 2] (left=0, right=1).
    """
    arrays = [a for a in (valid_pos, valid_open) if a is not None]
    if not arrays:
        if n_frames is None:
            raise ValueError("Need at least one validity array or n_frames")
        return np.ones((int(n_frames),), dtype=bool)

    t = int(arrays[0].shape[0])
    if n_frames is not None and int(n_frames) != t:
        raise ValueError(f"n_frames={n_frames} != validity length {t}")

    slots = tuple(int(s) for s in (required_slots if required_slots is not None else (0, 1)))
    if not slots:
        raise ValueError("required_slots must be non-empty")
    for s in slots:
        if s not in (0, 1):
            raise ValueError(f"required_slots entries must be 0 or 1, got {slots}")

    ok = np.ones((t,), dtype=bool)
    for name, arr in (("valid_pos", valid_pos), ("valid_open", valid_open)):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=bool)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"{name} must have shape [T, 2], got {a.shape}")
        if a.shape[0] != t:
            raise ValueError(f"{name} length {a.shape[0]} != {t}")
        ok &= a[:, list(slots)].all(axis=1)
    return ok


def _validate_index_column(df: pd.DataFrame, col: str, length: int, *, label: str) -> None:
    if col not in df.columns or len(df) == 0:
        return
    idx = df[col].to_numpy(dtype=np.int64)
    if (idx < 0).any() or (idx >= length).any():
        raise ValueError(
            f"{label}: {col} out of range for length={length} "
            f"(min={int(idx.min())}, max={int(idx.max())})"
        )


# ---------------------------------------------------------------------------
# 1) Robot bimanual sync
# ---------------------------------------------------------------------------


def synchronize_robot_bimanual(
    left_joint_ts: np.ndarray,
    right_joint_ts: np.ndarray,
    left_cam_ts: np.ndarray,
    right_cam_ts: np.ndarray,
    bird_ts: np.ndarray,
    front_ts: np.ndarray,
    out_csv: str | Path,
    *,
    eef_ts: np.ndarray | None = None,
    max_skew_s: float = 0.050,
    debug: bool = False,
    valid_pos: np.ndarray | None = None,
    valid_rot: np.ndarray | None = None,
    valid_open: np.ndarray | None = None,
    require_full_eef_pose: bool = True,
) -> pd.DataFrame:
    """
    Synchronize robot streams into one CSV (7 streams when EEF is provided).

    Index columns:
      left_joint_index, right_joint_index, left_index, right_index,
      bird_index, front_index, eef_pose_index

    When require_full_eef_pose is True, keep rows whose EEF index has valid
    xyz + gripper for both arms. Orientation / valid_rot is ignored.
    """
    del valid_rot  # orientation never gates sync validity
    left_j = _assert_finite_1d("left_joint", left_joint_ts)
    right_j = _assert_finite_1d("right_joint", right_joint_ts)
    left_c = _assert_finite_1d("left_cam", left_cam_ts)
    right_c = _assert_finite_1d("right_cam", right_cam_ts)
    bird = _assert_finite_1d("bird_cam", bird_ts)
    front = _assert_finite_1d("front_cam", front_ts)

    if eef_ts is None:
        raise ValueError(
            "eef_ts is required for robot sync. "
            "Pass timestamps from joint-data/combined_npz_commonframe."
        )

    eef_raw = _assert_finite_1d("eef_pose", eef_ts)
    eef_unique, eef_src_index = make_unique_increasing_timeline(eef_raw)

    tmp_csv = Path(out_csv).with_suffix(".tmp.csv")
    df = _sync_streams(
        ["left_joint", "right_joint", "left_cam", "right_cam", "bird_cam", "front_cam", "eef_pose"],
        [left_j, right_j, left_c, right_c, bird, front, eef_unique],
        tmp_csv,
        index_columns=[
            "left_joint_index",
            "right_joint_index",
            "left_index",
            "right_index",
            "bird_index",
            "front_index",
            "eef_pose_unique_index",
        ],
        max_skew_s=max_skew_s,
        debug=True,
        label="robot septuplets",
        write_csv=True,
    )
    try:
        tmp_csv.unlink(missing_ok=True)
    except TypeError:
        if tmp_csv.exists():
            tmp_csv.unlink()

    n_before = len(df)
    if len(df) > 0:
        uniq_idx = df["eef_pose_unique_index"].to_numpy(dtype=np.int64)
        df["eef_pose_index"] = eef_src_index[uniq_idx]
    else:
        df["eef_pose_index"] = []

    if require_full_eef_pose and len(df) > 0:
        missing = [name for name, arr in (("valid_pos", valid_pos), ("valid_open", valid_open)) if arr is None]
        if missing:
            raise ValueError(
                f"require_full_eef_pose=True but missing NPZ masks: {missing}. "
                "Pass valid_pos and valid_open from the EEF NPZ."
            )
        frame_ok = xyz_gripper_valid_mask(
            valid_pos=valid_pos, valid_open=valid_open, n_frames=len(eef_raw), required_slots=(0, 1)
        )
        eef_idx = df["eef_pose_index"].to_numpy(dtype=np.int64)
        in_range = (eef_idx >= 0) & (eef_idx < len(frame_ok))
        row_ok = np.zeros(len(df), dtype=bool)
        row_ok[in_range] = frame_ok[eef_idx[in_range]]
        df = df[row_ok].reset_index(drop=True)
        if len(df) > 0:
            df["master_index"] = np.arange(len(df), dtype=np.int64)
        print(
            f"  filtered incomplete EEF poses: kept {len(df)}/{n_before} "
            f"(dropped {n_before - len(df)}, require xyz+gripper for both arms; orient ignored)"
        )

    _validate_index_column(df, "left_joint_index", len(left_j), label="robot sync")
    _validate_index_column(df, "right_joint_index", len(right_j), label="robot sync")
    _validate_index_column(df, "left_index", len(left_c), label="robot sync")
    _validate_index_column(df, "right_index", len(right_c), label="robot sync")
    _validate_index_column(df, "bird_index", len(bird), label="robot sync")
    _validate_index_column(df, "front_index", len(front), label="robot sync")
    _validate_index_column(df, "eef_pose_index", len(eef_raw), label="robot sync")

    keep = [
        "master_index",
        "left_joint_index",
        "right_joint_index",
        "left_index",
        "right_index",
        "bird_index",
        "front_index",
        "eef_pose_index",
    ]
    if debug:
        keep = keep + [
            c for c in df.columns if c.endswith("_time") or c == "time_diff" or c == "eef_pose_unique_index"
        ]
    df = df[keep]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote robot sync CSV with eef_pose_index -> {out_path} (rows={len(df)}, debug={debug})")
    return df


# ---------------------------------------------------------------------------
# 2) Human hands sync
# ---------------------------------------------------------------------------


def synchronize_human_hands(
    bird_ts: np.ndarray,
    front_ts: np.ndarray,
    pose_ts: np.ndarray,
    out_csv: str | Path,
    *,
    max_skew_s: float = 0.050,
    debug: bool = False,
    valid_pos: np.ndarray | None = None,
    valid_rot: np.ndarray | None = None,
    valid_open: np.ndarray | None = None,
    require_full_pose: bool = True,
) -> pd.DataFrame:
    """
    Synchronize human bird/front cameras with hand-pose NPZ timestamps.

    pose_ts may contain duplicates (bag-rate NPZs). We collapse to a unique
    increasing timeline and write pose_index into the *original* NPZ array.

    Index columns: bird_index, front_index, pose_index
    """
    del valid_rot  # orientation never gates sync validity
    bird = _assert_finite_1d("bird_cam", bird_ts)
    front = _assert_finite_1d("front_cam", front_ts)
    pose_raw = _assert_finite_1d("pose", pose_ts)
    pose_unique, pose_src_index = make_unique_increasing_timeline(pose_raw)

    tmp_csv = Path(out_csv).with_suffix(".tmp.csv")
    df = _sync_streams(
        ["bird_cam", "front_cam", "pose"],
        [bird, front, pose_unique],
        tmp_csv,
        index_columns=["bird_index", "front_index", "pose_unique_index"],
        max_skew_s=max_skew_s,
        debug=True,
        label="human triplets",
    )
    try:
        tmp_csv.unlink(missing_ok=True)
    except TypeError:
        if tmp_csv.exists():
            tmp_csv.unlink()

    n_before = len(df)
    if len(df) > 0:
        uniq_idx = df["pose_unique_index"].to_numpy(dtype=np.int64)
        df["pose_index"] = pose_src_index[uniq_idx]
    else:
        df["pose_index"] = []

    if require_full_pose and len(df) > 0:
        missing = [name for name, arr in (("valid_pos", valid_pos), ("valid_open", valid_open)) if arr is None]
        if missing:
            raise ValueError(
                f"require_full_pose=True but missing NPZ masks: {missing}. "
                "Pass valid_pos and valid_open, or set require_full_pose=False."
            )
        frame_ok = xyz_gripper_valid_mask(
            valid_pos=valid_pos, valid_open=valid_open, n_frames=len(pose_raw), required_slots=(0, 1)
        )
        pose_idx = df["pose_index"].to_numpy(dtype=np.int64)
        in_range = (pose_idx >= 0) & (pose_idx < len(frame_ok))
        row_ok = np.zeros(len(df), dtype=bool)
        row_ok[in_range] = frame_ok[pose_idx[in_range]]
        df = df[row_ok].reset_index(drop=True)
        if len(df) > 0:
            df["master_index"] = np.arange(len(df), dtype=np.int64)
        n_dropped = n_before - len(df)
        print(
            f"  filtered incomplete poses: kept {len(df)}/{n_before} "
            f"(dropped {n_dropped}, require xyz+open for both hands; orient ignored)"
        )

    _validate_index_column(df, "bird_index", len(bird), label="human sync")
    _validate_index_column(df, "front_index", len(front), label="human sync")
    _validate_index_column(df, "pose_index", len(pose_raw), label="human sync")

    keep = ["master_index", "bird_index", "front_index", "pose_index"]
    if debug:
        keep = keep + [c for c in df.columns if c.endswith("_time") or c == "time_diff" or c == "pose_unique_index"]
    df = df[keep]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote human sync CSV with pose_index remap -> {out_path} (rows={len(df)}, debug={debug})")
    return df


# ---------------------------------------------------------------------------
# 3) Mixed one-hand + one-robot-arm sync
# ---------------------------------------------------------------------------

Side = Literal["left", "right"]

# Session-name presets used under recording/sessions/
EMBODIMENT_PRESETS: dict[str, dict[str, Side]] = {
    "left_robot_right_hand": {"robot_side": "left", "hand_side": "right"},
    "right_robot_left_hand": {"robot_side": "right", "hand_side": "left"},
}

SLOT = {"left": 0, "right": 1}

MIXED_SYNC_INDEX_COLUMNS = (
    "bird_index",
    "front_index",
    "wrist_index",
    "joint_index",
    "hand_pose_index",
    "eef_pose_index",
)


def side_to_slot(side: Side) -> int:
    if side not in SLOT:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return SLOT[side]


def infer_mixed_preset(data_root: Path) -> dict[str, Side] | None:
    """Infer robot_side/hand_side from a left_robot_right_hand / right_robot_left_hand path."""
    parts = {p.lower() for p in data_root.parts}
    for name, preset in EMBODIMENT_PRESETS.items():
        if name in parts:
            return preset
    return None


def synchronize_mixed_hand_robot(
    bird_ts: np.ndarray,
    front_ts: np.ndarray,
    wrist_ts: np.ndarray,
    joint_ts: np.ndarray,
    hand_pose_ts: np.ndarray,
    eef_ts: np.ndarray,
    out_csv: str | Path,
    *,
    robot_side: Side,
    hand_side: Side,
    hand_valid_pos: np.ndarray,
    hand_valid_open: np.ndarray,
    eef_valid_pos: np.ndarray,
    eef_valid_open: np.ndarray,
    max_skew_s: float = 0.050,
    debug: bool = False,
    require_valid_active_slots: bool = True,
) -> pd.DataFrame:
    """
    Synchronize mixed one-hand + one-arm demos.

    Index columns:
      bird_index, front_index, wrist_index, joint_index,
      hand_pose_index, eef_pose_index

    hand_pose_index / eef_pose_index index the *original* NPZ timelines
    (duplicate timestamps are collapsed then remapped).
    """
    robot_slot = side_to_slot(robot_side)
    hand_slot = side_to_slot(hand_side)

    bird = _assert_finite_1d("bird_cam", bird_ts)
    front = _assert_finite_1d("front_cam", front_ts)
    wrist = _assert_finite_1d("wrist_cam", wrist_ts)
    joint = _assert_finite_1d("joint", joint_ts)
    hand_raw = _assert_finite_1d("hand_pose", hand_pose_ts)
    eef_raw = _assert_finite_1d("eef_pose", eef_ts)

    hand_unique, hand_src = make_unique_increasing_timeline(hand_raw)
    eef_unique, eef_src = make_unique_increasing_timeline(eef_raw)

    tmp_csv = Path(out_csv).with_suffix(".tmp.csv")
    df = _sync_streams(
        ["bird_cam", "front_cam", "wrist_cam", "joint", "hand_pose", "eef_pose"],
        [bird, front, wrist, joint, hand_unique, eef_unique],
        tmp_csv,
        index_columns=[
            "bird_index",
            "front_index",
            "wrist_index",
            "joint_index",
            "hand_pose_unique_index",
            "eef_pose_unique_index",
        ],
        max_skew_s=max_skew_s,
        debug=True,
        label=f"mixed({robot_side}-robot/{hand_side}-hand)",
        write_csv=True,
    )
    try:
        tmp_csv.unlink(missing_ok=True)
    except TypeError:
        if tmp_csv.exists():
            tmp_csv.unlink()

    n_before = len(df)
    if len(df) > 0:
        df["hand_pose_index"] = hand_src[df["hand_pose_unique_index"].to_numpy(dtype=np.int64)]
        df["eef_pose_index"] = eef_src[df["eef_pose_unique_index"].to_numpy(dtype=np.int64)]
    else:
        df["hand_pose_index"] = []
        df["eef_pose_index"] = []

    if require_valid_active_slots and len(df) > 0:
        hand_ok = xyz_gripper_valid_mask(
            valid_pos=hand_valid_pos, valid_open=hand_valid_open, n_frames=len(hand_raw), required_slots=(hand_slot,)
        )
        eef_ok = xyz_gripper_valid_mask(
            valid_pos=eef_valid_pos, valid_open=eef_valid_open, n_frames=len(eef_raw), required_slots=(robot_slot,)
        )
        h_idx = df["hand_pose_index"].to_numpy(dtype=np.int64)
        e_idx = df["eef_pose_index"].to_numpy(dtype=np.int64)
        row_ok = (h_idx >= 0) & (h_idx < len(hand_ok)) & (e_idx >= 0) & (e_idx < len(eef_ok))
        keep = np.zeros(len(df), dtype=bool)
        keep[row_ok] = hand_ok[h_idx[row_ok]] & eef_ok[e_idx[row_ok]]
        df = df[keep].reset_index(drop=True)
        if len(df) > 0:
            df["master_index"] = np.arange(len(df), dtype=np.int64)
        print(
            f"  filtered inactive/invalid slots: kept {len(df)}/{n_before} "
            f"(hand_slot={hand_slot}, robot_slot={robot_slot}; orient ignored)"
        )

    _validate_index_column(df, "bird_index", len(bird), label="mixed sync")
    _validate_index_column(df, "front_index", len(front), label="mixed sync")
    _validate_index_column(df, "wrist_index", len(wrist), label="mixed sync")
    _validate_index_column(df, "joint_index", len(joint), label="mixed sync")
    _validate_index_column(df, "hand_pose_index", len(hand_raw), label="mixed sync")
    _validate_index_column(df, "eef_pose_index", len(eef_raw), label="mixed sync")

    keep_cols = ["master_index", *MIXED_SYNC_INDEX_COLUMNS]
    if debug:
        keep_cols = keep_cols + [
            c for c in df.columns if c.endswith("_time") or c == "time_diff" or c.endswith("_unique_index")
        ]
    if "master_index" not in df.columns and len(df) > 0:
        df["master_index"] = np.arange(len(df), dtype=np.int64)
    elif "master_index" not in df.columns:
        df["master_index"] = []

    df = df[[c for c in keep_cols if c in df.columns]]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(
        f"Wrote mixed sync CSV -> {out_path} "
        f"(rows={len(df)}, robot={robot_side}, hand={hand_side}, debug={debug})"
    )
    return df


# ---------------------------------------------------------------------------
# 4) Directory discovery + orchestration (formerly split across
#    training_combined.py and the standalone build_sync.py module)
# ---------------------------------------------------------------------------


def resolve_robot_eef_dir(robot_root: Path, override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    preferred = (robot_root / ROBOT_EEF_RELDIR).resolve()
    if preferred.is_dir() and any(preferred.glob("*.npz")):
        return preferred
    raise FileNotFoundError(
        f"Robot/mixed EEF NPZ dir not found. Expected {preferred} under {robot_root}."
    )


def resolve_human_pose_dir(human_root: Path, override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    preferred = (human_root / HUMAN_POSE_RELDIR).resolve()
    if preferred.is_dir() and any(preferred.glob("*.npz")):
        return preferred
    fallback = (human_root / "bird-realsense-data" / "combined_npz").resolve()
    if fallback.is_dir() and any(fallback.glob("*.npz")):
        print(f"WARNING: default hand/human pose dir missing/empty ({preferred}); falling back to {fallback}")
        return fallback
    raise FileNotFoundError(
        f"Human/mixed hand pose NPZ dir not found. Expected {preferred} under {human_root}."
    )


def build_robot_sync_csvs(
    data_root: Path,
    sync_dir: Path,
    eef_dir: Path,
    max_skew_s: float,
    max_demos: int | None,
) -> None:
    bird = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "bird-realsense-data" / "npy").glob("*.npy"))}
    front = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "front-realsense-data" / "npy").glob("*.npy"))}
    left_c = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "aloha-data" / "left" / "npy").glob("*.npy"))}
    right_c = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "aloha-data" / "right" / "npy").glob("*.npy"))}
    left_j = {
        demo_id_from_joint_npy(p, prefix="joint_timestamp_"): p
        for p in sorted((data_root / "joint-data" / "left" / "time").glob("*.npy"))
    }
    right_j = {
        demo_id_from_joint_npy(p, prefix="joint_timestamp_"): p
        for p in sorted((data_root / "joint-data" / "right" / "time").glob("*.npy"))
    }
    eef = {demo_id_from_robot_eef_npz(p): p for p in sorted(eef_dir.glob("*.npz"))}

    base_ids = sorted(set(bird) & set(front) & set(left_c) & set(right_c) & set(left_j) & set(right_j))
    ids = sorted(set(base_ids) & set(eef))
    for demo_id in sorted(set(base_ids) - set(eef)):
        print(f"WARNING: skip robot {demo_id} - missing EEF pose under {eef_dir}")

    if max_demos is not None and max_demos > 0:
        ids = ids[:max_demos]
    if not ids:
        raise FileNotFoundError(
            f"No complete robot demos under {data_root} with EEF in {eef_dir}. "
            f"base_complete={len(base_ids)} eef={len(eef)}"
        )
    sync_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building robot sync for {len(ids)} demos -> {sync_dir}")
    for demo_id in ids:
        eef_npz = np.load(eef[demo_id])
        for key in ("timestamps", "pose", "valid_pos", "valid_open"):
            if key not in eef_npz.files:
                raise KeyError(f"{eef[demo_id].name} missing required key '{key}'")
        synchronize_robot_bimanual(
            np.load(left_j[demo_id]),
            np.load(right_j[demo_id]),
            np.load(left_c[demo_id]),
            np.load(right_c[demo_id]),
            np.load(bird[demo_id]),
            np.load(front[demo_id]),
            sync_dir / f"{demo_id}.csv",
            eef_ts=eef_npz["timestamps"],
            max_skew_s=max_skew_s,
            debug=False,
            valid_pos=eef_npz["valid_pos"],
            valid_open=eef_npz["valid_open"],
            require_full_eef_pose=True,
        )
    prune_orphan_sync_csvs(sync_dir, ids)


def build_human_sync_csvs(
    data_root: Path,
    sync_dir: Path,
    pose_dir: Path,
    max_skew_s: float,
    max_demos: int | None,
) -> None:
    bird = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "bird-realsense-data" / "npy").glob("*.npy"))}
    front = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "front-realsense-data" / "npy").glob("*.npy"))}
    pose = {demo_id_from_pose_npz(p): p for p in sorted(pose_dir.glob("*.npz"))}
    ids = sorted(set(bird) & set(front) & set(pose))
    if max_demos is not None and max_demos > 0:
        ids = ids[:max_demos]
    if not ids:
        raise FileNotFoundError(
            f"No complete human demos under {data_root}. "
            f"Need bird/front npy timestamps and pose NPZs in {pose_dir}."
        )
    sync_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building human sync for {len(ids)} demos -> {sync_dir} (pose_dir={pose_dir})")
    for demo_id in ids:
        pose_npz = np.load(pose[demo_id])
        for key in ("valid_pos", "valid_open"):
            if key not in pose_npz.files:
                raise KeyError(f"{pose[demo_id].name} missing required validity key '{key}'")
        synchronize_human_hands(
            np.load(bird[demo_id]),
            np.load(front[demo_id]),
            pose_npz["timestamps"],
            sync_dir / f"{demo_id}.csv",
            max_skew_s=max_skew_s,
            debug=False,
            valid_pos=pose_npz["valid_pos"],
            valid_open=pose_npz["valid_open"],
            require_full_pose=True,
        )
    prune_orphan_sync_csvs(sync_dir, ids)


def build_mixed_sync_csvs(
    data_root: Path,
    sync_dir: Path,
    *,
    robot_side: Side,
    hand_side: Side,
    pose_dir: Path,
    eef_dir: Path,
    max_skew_s: float,
    max_demos: int | None,
) -> list[str]:
    def _map_hash(dir_path: Path) -> dict[str, Path]:
        if not dir_path.is_dir():
            return {}
        return {demo_id_from_hash_filename(p): p for p in sorted(dir_path.glob("*.npy"))}

    bird = _map_hash(data_root / "bird-realsense-data" / "npy")
    front = _map_hash(data_root / "front-realsense-data" / "npy")
    wrist = _map_hash(data_root / "aloha-data" / robot_side / "npy")

    joint_dir = data_root / "joint-data" / robot_side / "time"
    joints: dict[str, Path] = {}
    if joint_dir.is_dir():
        for p in sorted(joint_dir.glob("*.npy")):
            joints[demo_id_from_joint_npy(p, prefix="joint_timestamp_")] = p

    hands = {demo_id_from_pose_npz(p): p for p in sorted(pose_dir.glob("*.npz"))}
    eefs = {demo_id_from_robot_eef_npz(p): p for p in sorted(eef_dir.glob("*.npz"))}

    ids = sorted(set(bird) & set(front) & set(wrist) & set(joints) & set(hands) & set(eefs))
    if max_demos is not None and max_demos > 0:
        ids = ids[: int(max_demos)]

    print(
        f"Mixed sync discovery under {data_root}\n"
        f"  robot_side={robot_side} hand_side={hand_side}\n"
        f"  bird={len(bird)} front={len(front)} wrist({robot_side})={len(wrist)} "
        f"joint({robot_side})={len(joints)} hand_npz={len(hands)} eef_npz={len(eefs)}\n"
        f"  complete demos={len(ids)} -> {sync_dir}"
    )
    if not ids:
        raise FileNotFoundError(
            f"No complete mixed demos under {data_root}. "
            f"Need bird, front, {robot_side} wrist, {robot_side} joints, "
            f"hand NPZ in {pose_dir}, EEF NPZ in {eef_dir}."
        )

    sync_dir.mkdir(parents=True, exist_ok=True)
    wrote: list[str] = []
    for demo_id in ids:
        hand_npz = np.load(hands[demo_id])
        eef_npz = np.load(eefs[demo_id])
        for key in ("timestamps", "valid_pos", "valid_open"):
            if key not in hand_npz.files:
                raise KeyError(f"{hands[demo_id].name} missing '{key}'")
            if key not in eef_npz.files:
                raise KeyError(f"{eefs[demo_id].name} missing '{key}'")

        synchronize_mixed_hand_robot(
            np.load(bird[demo_id]),
            np.load(front[demo_id]),
            np.load(wrist[demo_id]),
            np.load(joints[demo_id]),
            hand_npz["timestamps"],
            eef_npz["timestamps"],
            sync_dir / f"{demo_id}.csv",
            robot_side=robot_side,
            hand_side=hand_side,
            hand_valid_pos=hand_npz["valid_pos"],
            hand_valid_open=hand_npz["valid_open"],
            eef_valid_pos=eef_npz["valid_pos"],
            eef_valid_open=eef_npz["valid_open"],
            max_skew_s=max_skew_s,
            debug=False,
            require_valid_active_slots=True,
        )
        wrote.append(demo_id)
    prune_orphan_sync_csvs(sync_dir, wrote)
    print(f"Done. Wrote {len(wrote)} mixed sync CSVs -> {sync_dir}")
    return wrote
