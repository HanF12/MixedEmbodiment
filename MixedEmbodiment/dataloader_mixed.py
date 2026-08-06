"""Mixed one-hand + one-robot-arm episode dataset for MixedEmbodiment ACT (true 3-cam)."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from MixedEmbodiment.config import (
    DEFAULT_NUM_QUERIES,
    EMBODIMENT_MIXED,
    JOINT_DIM_PER_ARM,
    MIXED_SYNC_INDEX_COLUMNS,
    POSE_DIM,
    ROBOT_JOINT_DIM,
    ROBOT_NPY_GRIPPER_BINARIZE_THRESHOLD,
    ROBOT_NPZ_GRIPPER_BINARIZE_THRESHOLD,
    camera_mask_tensor,
    compose_mixed_flat_pose,
    joint_loss_mask_for_side,
    single_arm_joint_vector,
    stack_camera_tensors,
)
from MixedEmbodiment.data_synchronization import side_to_slot, xyz_gripper_valid_mask
from MixedEmbodiment.dataloader_utils import (
    build_image_transform,
    compute_relative_pose_stats,
    demo_id_from_hash_filename,
    demo_id_from_joint_npy,
    demo_id_from_pose_npz,
    demo_id_from_robot_eef_npz,
    frame_nbytes,
    index_paths_by_demo_id,
    load_frame,
    load_video_frames,
    normalize_future_chunk,
    relative_pose_chunk,
    resolve_path,
    store_frame,
    zero_rgb_like,
)


class MixedEpisodeDataset(Dataset):
    """
    One item = one mixed episode (random start inside episode).

    Pose [8] = hand xyz+open in hand_slot + robot EEF xyz+grip in robot_slot
               (only those slots read from bimanual [2,10] NPZs).
    Joints [14] = active arm qpos (gripper binarized); inactive arm zeros.
    Cameras [3] = bird + active wrist; inactive wrist zeroed/masked.
    Front RGB is used only in sync CSV generation (not loaded here).
    Proprio / CVAE follow the robot joint pathway (see core.py) — mixed never
    feeds pose_state into the model as an observation. No embodiment does;
    human has no proprio observation at all (see core.py's module docstring).
    """

    def __init__(
        self,
        *,
        bird_vids_dir,
        wrist_vids_dir,
        joint_data_dir,
        hand_pose_npz_dir,
        eef_pose_data_dir,
        sync_csv_dir,
        robot_side: str,
        hand_side: str,
        front_vids_dir=None,  # kept for API compat; not loaded into model
        num_queries: int = DEFAULT_NUM_QUERIES,
        transform: str = "resnet_normalization",
        max_demos: int | None = None,
        resize_factor: float = 1.0,
        max_sync_rows: int | None = None,
        jpeg_in_ram: bool = False,
        jpeg_quality: int = 90,
        npz_gripper_binarize_threshold: float = ROBOT_NPZ_GRIPPER_BINARIZE_THRESHOLD,
        npy_gripper_binarize_threshold: float = ROBOT_NPY_GRIPPER_BINARIZE_THRESHOLD,
    ) -> None:
        super().__init__()
        if robot_side not in ("left", "right") or hand_side not in ("left", "right"):
            raise ValueError("robot_side and hand_side must be 'left' or 'right'")
        if robot_side == hand_side:
            raise ValueError("mixed embodiment expects opposite robot_side and hand_side")
        del front_vids_dir  # sync-only; not a model camera

        self.num_queries = int(num_queries)
        self.resize_factor = float(resize_factor)
        self.max_sync_rows = int(max_sync_rows) if max_sync_rows is not None else None
        self.jpeg_in_ram = bool(jpeg_in_ram)
        self.jpeg_quality = int(jpeg_quality)
        self.npz_gripper_binarize_threshold = float(npz_gripper_binarize_threshold)
        self.npy_gripper_binarize_threshold = float(npy_gripper_binarize_threshold)
        self.robot_side = robot_side
        self.hand_side = hand_side
        self.robot_slot = side_to_slot(robot_side)  # type: ignore[arg-type]
        self.hand_slot = side_to_slot(hand_side)  # type: ignore[arg-type]
        self.image_transform = build_image_transform(transform)
        self.joint_loss_mask = joint_loss_mask_for_side(robot_side)

        bird_vids = sorted(resolve_path(bird_vids_dir).glob("*.mp4"))
        wrist_vids = sorted(resolve_path(wrist_vids_dir).glob("*.mp4"))
        joints = sorted(resolve_path(joint_data_dir).glob("*.npy"))
        hand_files = sorted(resolve_path(hand_pose_npz_dir).glob("*.npz"))
        eef_files = sorted(resolve_path(eef_pose_data_dir).glob("*.npz"))
        sync_csvs = sorted(resolve_path(sync_csv_dir).glob("*.csv"))
        if max_demos is not None and max_demos > 0:
            sync_csvs = sync_csvs[:max_demos]
        if not sync_csvs:
            raise FileNotFoundError(f"No mixed sync CSVs in {sync_csv_dir}")

        bird_by = index_paths_by_demo_id(bird_vids, demo_id_from_hash_filename)
        wrist_by = index_paths_by_demo_id(wrist_vids, demo_id_from_hash_filename)
        joint_by = index_paths_by_demo_id(joints, demo_id_from_joint_npy)
        hand_by = index_paths_by_demo_id(hand_files, demo_id_from_pose_npz)
        eef_by = index_paths_by_demo_id(eef_files, demo_id_from_robot_eef_npz)

        print(
            f"Mixed dataset: 3-cam slots [bird, left_wrist, right_wrist] "
            f"(robot={robot_side}, hand={hand_side}); front for sync only"
        )
        if self.jpeg_in_ram:
            print(f"Mixed dataset: JPEG-in-RAM enabled (quality={self.jpeg_quality})")

        self.bird_frames: list = []
        self.wrist_frames: list = []
        self.joint_data: List[torch.Tensor] = []
        self.pose_data: List[torch.Tensor] = []
        self.demo_start_idx: List[int] = []
        self.demo_lengths: List[int] = []
        self.sample_demo_idx: List[int] = []
        self.num_samples = 0
        self.num_demos = 0

        for sync_csv in sync_csvs:
            rec_id = sync_csv.stem
            needed = {"bird": bird_by, "wrist": wrist_by, "joint": joint_by, "hand": hand_by, "eef": eef_by}
            missing = [k for k, m in needed.items() if rec_id not in m]
            if missing:
                print(f"WARNING: skip mixed {rec_id} - missing {missing}")
                continue

            df = pd.read_csv(sync_csv)
            for col in MIXED_SYNC_INDEX_COLUMNS:
                if col not in df.columns:
                    raise KeyError(f"{sync_csv.name} missing column {col}")

            hand_npz = np.load(hand_by[rec_id])
            eef_npz = np.load(eef_by[rec_id])
            for key in ("pose", "valid_pos", "valid_open"):
                if key not in hand_npz.files:
                    raise KeyError(f"{hand_by[rec_id].name} missing '{key}'")
                if key not in eef_npz.files:
                    raise KeyError(f"{eef_by[rec_id].name} missing '{key}'")
            hand_pose = np.asarray(hand_npz["pose"], dtype=np.float32)
            eef_pose = np.asarray(eef_npz["pose"], dtype=np.float32)
            # Same as robot/human: re-check validity in the dataset (sync already filtered,
            # but this catches rows that become inconsistent after caps).
            hand_ok = xyz_gripper_valid_mask(
                valid_pos=hand_npz["valid_pos"],
                valid_open=hand_npz["valid_open"],
                n_frames=len(hand_pose),
                required_slots=(self.hand_slot,),
            )
            eef_ok = xyz_gripper_valid_mask(
                valid_pos=eef_npz["valid_pos"],
                valid_open=eef_npz["valid_open"],
                n_frames=len(eef_pose),
                required_slots=(self.robot_slot,),
            )

            if len(df) > 0:
                keep = []
                for i in range(len(df)):
                    hidx = int(df.iloc[i]["hand_pose_index"])
                    eidx = int(df.iloc[i]["eef_pose_index"])
                    ok = (
                        0 <= hidx < len(hand_ok)
                        and 0 <= eidx < len(eef_ok)
                        and bool(hand_ok[hidx])
                        and bool(eef_ok[eidx])
                    )
                    keep.append(ok)
                df = df.loc[np.asarray(keep, dtype=bool)].copy()

            if self.max_sync_rows is not None:
                df = df.iloc[: self.max_sync_rows].copy()
            df = df.reset_index(drop=True)
            if df.empty:
                print(f"WARNING: skip mixed {rec_id} - no valid rows after filters")
                continue

            bird_f = load_video_frames(bird_by[rec_id], resize_factor=self.resize_factor, label=f"bird({rec_id})")
            wrist_f = load_video_frames(wrist_by[rec_id], resize_factor=self.resize_factor, label=f"wrist({rec_id})")
            joint_arr = np.load(joint_by[rec_id])

            demo_idx = self.num_demos
            self.demo_start_idx.append(self.num_samples)
            n_i = len(df)
            self.demo_lengths.append(n_i)

            for i in range(n_i):
                self.bird_frames.append(
                    store_frame(bird_f[int(df.loc[i, "bird_index"])], jpeg_in_ram=self.jpeg_in_ram, jpeg_quality=self.jpeg_quality)
                )
                self.wrist_frames.append(
                    store_frame(wrist_f[int(df.loc[i, "wrist_index"])], jpeg_in_ram=self.jpeg_in_ram, jpeg_quality=self.jpeg_quality)
                )
                self.joint_data.append(
                    single_arm_joint_vector(
                        joint_arr[int(df.loc[i, "joint_index"])],
                        robot_side=self.robot_side,
                        rec_id=rec_id,
                        binarize_grippers=True,
                        gripper_threshold=self.npy_gripper_binarize_threshold,
                    )
                )
                hidx = int(df.loc[i, "hand_pose_index"])
                eidx = int(df.loc[i, "eef_pose_index"])
                self.pose_data.append(
                    compose_mixed_flat_pose(
                        hand_pose[hidx],
                        eef_pose[eidx],
                        hand_slot=self.hand_slot,
                        robot_slot=self.robot_slot,
                        binarize_eef_gripper=True,
                        eef_gripper_threshold=self.npz_gripper_binarize_threshold,
                        rec_id=rec_id,
                    )
                )
                self.sample_demo_idx.append(demo_idx)

            self.num_samples += n_i
            self.num_demos += 1
            print(f"    -> mixed {rec_id}: {n_i} samples (robot={self.robot_side}, hand={self.hand_side})")

        if self.num_demos == 0:
            raise FileNotFoundError("No complete mixed demos found for MixedEmbodiment.")

        all_q = torch.stack(self.joint_data, dim=0)  # [N, 14]
        all_p = torch.stack(self.pose_data, dim=0)  # [N, 8]
        self.joint_mean = torch.zeros(ROBOT_JOINT_DIM, dtype=torch.float32)
        self.joint_std = torch.ones(ROBOT_JOINT_DIM, dtype=torch.float32)
        active = slice(0, JOINT_DIM_PER_ARM) if self.robot_side == "left" else slice(JOINT_DIM_PER_ARM, ROBOT_JOINT_DIM)
        self.joint_mean[active] = all_q[:, active].mean(dim=0)
        self.joint_std[active] = all_q[:, active].std(dim=0).clamp(min=1e-2)

        self.pose_abs_mean = all_p.mean(dim=0)
        self.pose_abs_std = all_p.std(dim=0).clamp(min=1e-2)
        self.pose_mean, self.pose_std = compute_relative_pose_stats(
            self.pose_data,
            demo_start_idx=self.demo_start_idx,
            demo_lengths=self.demo_lengths,
            num_queries=self.num_queries,
        )
        self.pose_rel_mean = self.pose_mean
        self.pose_rel_std = self.pose_std
        self.state_mean = self.joint_mean
        self.state_std = self.joint_std

        frame_bytes = sum(frame_nbytes(f) for f in self.bird_frames) + sum(frame_nbytes(f) for f in self.wrist_frames)
        print(
            f"Mixed dataset ready: demos={self.num_demos} samples={self.num_samples} "
            f"robot_side={self.robot_side} hand_side={self.hand_side} "
            f"(frame_storage={'jpeg' if self.jpeg_in_ram else 'raw'} "
            f"~{frame_bytes / (1024**3):.2f} GiB)"
        )

    def __len__(self) -> int:
        return self.num_demos

    def __getitem__(self, idx: int) -> dict:
        episode_idx = int(idx)
        ep_start = self.demo_start_idx[episode_idx]
        ep_len = self.demo_lengths[episode_idx]
        start_in_ep = int(np.random.randint(0, ep_len))
        sample_idx = ep_start + start_in_ep
        demo_end = ep_start + ep_len

        bird_np = load_frame(self.bird_frames[sample_idx])
        bird_t = self.image_transform(bird_np)
        wrist_t = self.image_transform(load_frame(self.wrist_frames[sample_idx]))
        inactive_t = self.image_transform(zero_rgb_like(bird_np))
        if self.robot_side == "left":
            left_t, right_t = wrist_t, inactive_t
        else:
            left_t, right_t = inactive_t, wrist_t
        images = stack_camera_tensors(bird_t, left_t, right_t)

        pose_raw = self.pose_data[sample_idx]
        joint_raw = self.joint_data[sample_idx]
        pose_state = (pose_raw - self.pose_abs_mean) / self.pose_abs_std
        joint_state = (joint_raw - self.joint_mean) / self.joint_std

        slice_end = min(demo_end, sample_idx + self.num_queries)
        pose_future_abs = list(self.pose_data[sample_idx:slice_end])
        joint_future = list(self.joint_data[sample_idx:slice_end])
        pose_future_rel = relative_pose_chunk(pose_future_abs, anchor=pose_raw)
        pose_actions, is_pad = normalize_future_chunk(
            pose_future_rel, mean=self.pose_mean, std=self.pose_std, num_queries=self.num_queries
        )
        joint_actions, is_pad_j = normalize_future_chunk(
            joint_future, mean=self.joint_mean, std=self.joint_std, num_queries=self.num_queries
        )
        if not torch.equal(is_pad, is_pad_j):
            raise RuntimeError("Pose/joint pad masks diverged in mixed dataset")

        return {
            "embodiment": EMBODIMENT_MIXED,
            "images": images,
            "camera_mask": camera_mask_tensor(EMBODIMENT_MIXED, robot_side=self.robot_side),
            "pose_state": pose_state,
            "pose_actions": pose_actions,
            "joint_state": joint_state,
            "joint_actions": joint_actions,
            "joint_loss_mask": self.joint_loss_mask.clone(),
            "has_joint_target": True,
            "is_pad": is_pad,
        }


class ConcatMixedEpisodeDataset(ConcatDataset):
    """
    One mixed modality spanning both session types (LR + RR).

    Each child keeps its own norm stats / side masks; __getitem__ routes correctly.
    Pooled stats are exposed for checkpointing convenience.
    """

    def __init__(self, datasets: list[MixedEpisodeDataset]) -> None:
        if not datasets:
            raise ValueError("ConcatMixedEpisodeDataset requires at least one dataset")
        super().__init__(datasets)
        self.datasets: list[MixedEpisodeDataset] = list(datasets)  # type: ignore[assignment]
        self.num_demos = sum(int(d.num_demos) for d in self.datasets)
        self.num_samples = sum(int(d.num_samples) for d in self.datasets)
        self._pool_stats()

    def _pool_stats(self) -> None:
        total = float(sum(max(1, int(d.num_samples)) for d in self.datasets))
        joint_mean = torch.zeros(ROBOT_JOINT_DIM, dtype=torch.float32)
        joint_var = torch.zeros(ROBOT_JOINT_DIM, dtype=torch.float32)
        pose_abs_mean = torch.zeros(POSE_DIM, dtype=torch.float32)
        pose_abs_var = torch.zeros(POSE_DIM, dtype=torch.float32)
        pose_rel_mean = torch.zeros(POSE_DIM, dtype=torch.float32)
        pose_rel_var = torch.zeros(POSE_DIM, dtype=torch.float32)
        for d in self.datasets:
            w = float(d.num_samples) / total
            joint_mean += w * d.joint_mean
            joint_var += w * (d.joint_std ** 2)
            pose_abs_mean += w * d.pose_abs_mean
            pose_abs_var += w * (d.pose_abs_std ** 2)
            pose_rel_mean += w * d.pose_mean
            pose_rel_var += w * (d.pose_std ** 2)
        self.joint_mean = joint_mean
        self.joint_std = joint_var.clamp(min=1e-4).sqrt().clamp(min=1e-2)
        self.pose_abs_mean = pose_abs_mean
        self.pose_abs_std = pose_abs_var.clamp(min=1e-4).sqrt().clamp(min=1e-2)
        self.pose_mean = pose_rel_mean
        self.pose_std = pose_rel_var.clamp(min=1e-4).sqrt().clamp(min=1e-2)
        self.pose_rel_mean = self.pose_mean
        self.pose_rel_std = self.pose_std
        self.state_mean = self.joint_mean
        self.state_std = self.joint_std
        print(
            f"Concat mixed modality: children={len(self.datasets)} "
            f"demos={self.num_demos} samples={self.num_samples}"
        )
