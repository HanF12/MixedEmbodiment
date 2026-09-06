#!/usr/bin/env python
"""
t-SNE of ResNet backbone embeddings for Combined_relative_3cam / MixedEmbodiment.

For each sampled frame, runs the **bird-view** image through the trained ResNet
backbone (never wrist cams), applies global average pooling, and collects one
embedding per frame. Optionally also extracts:
  - transformer **encoder memory** (mean over latent+proprio+spatial tokens)
  - ACT **decoder** query features ``hs`` (mean over K, before action heads)
Both get the same PCA / t-SNE / UMAP treatment.

Omar-style *no-pose-proprio* checkpoints (``human_input_const`` /
``human_cvae_state_const`` instead of pose Linear projs) are auto-detected
per checkpoint; override with ``--no_pose_proprio auto|on|off|none``.

Also supports *no-human-proprio* checkpoints that removed human state adapters
entirely (no pose projs and no learned constants).

Scatter plots are colored by dataset type:
  - Combined: human vs robot (teleop)
  - MixedEmbodiment: teleop, human, left robot + right hand, right robot + left hand

Backbone extractors are pluggable via the BackboneFeatureExtractor protocol so
other encoders can be swapped in without changing the sampling / t-SNE pipeline.

Example:
  python plot_resnet_tsne.py \\
    --sessions_root sessions_fold --embodiments robot,human,mixed \\
    --checkpoints MixedEmbodiment/weights/20260826_145417/mixed_act_epoch_20000.pth \\
    --frame_stride 20 --max_demos 2
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MixedEmbodiment.config import (  # noqa: E402
    DEFAULT_NUM_QUERIES,
    DEFAULT_SESSIONS_ROOT,
    EMBODIMENT_HUMAN,
    EMBODIMENT_MIXED,
    EMBODIMENT_NAMES,
    EMBODIMENT_ROBOT,
    MODEL_CAMERA_NAMES,
    ROBOT_JOINT_DIM,
    camera_mask_tensor,
    discover_sessions_roots,
    stack_camera_tensors,
)
from MixedEmbodiment.data_synchronization import (  # noqa: E402
    build_human_sync_csvs,
    build_mixed_sync_csvs,
    build_robot_sync_csvs,
    infer_mixed_preset,
    resolve_human_pose_dir,
    resolve_robot_eef_dir,
)
from MixedEmbodiment.dataloader_human import HumanEpisodeDataset  # noqa: E402
from MixedEmbodiment.dataloader_mixed import MixedEpisodeDataset  # noqa: E402
from MixedEmbodiment.dataloader_robot import RobotEpisodeDataset  # noqa: E402
from MixedEmbodiment.dataloader_utils import (  # noqa: E402
    demo_id_from_hash_filename,
    demo_id_from_robot_eef_npz,
    load_frame,
    zero_rgb_like,
)

_TS_RE = re.compile(r"(\d{14})$")


def _timestamp_suffix(demo_id: str) -> str | None:
    m = _TS_RE.search(demo_id)
    return m.group(1) if m else None


def remap_eef_dir_to_session_ids(
    mixed_root: Path,
    eef_dir: Path,
    staging_root: Path,
) -> Path:
    """
    Mixed sessions sometimes store EEF NPZs under teleop_bimanual_<ts>_... while
    videos / sync IDs use <session_kind>_<ts>. Symlink NPZs into a staging dir
    renamed so demo_id_from_robot_eef_npz matches the bird demo IDs.
    """
    bird_npy = mixed_root / "bird-realsense-data" / "npy"
    bird_ids = [demo_id_from_hash_filename(p) for p in sorted(bird_npy.glob("*.npy"))]
    eef_by_id = {demo_id_from_robot_eef_npz(p): p for p in sorted(eef_dir.glob("*.npz"))}

    # Already aligned — no remapping needed.
    if set(bird_ids) & set(eef_by_id):
        return eef_dir

    eef_by_ts = {_timestamp_suffix(k): p for k, p in eef_by_id.items() if _timestamp_suffix(k)}
    staging = staging_root / f"{mixed_root.parent.name}_{mixed_root.name}_eef_remapped"
    if staging.exists():
        for p in staging.glob("*.npz"):
            p.unlink()
    else:
        staging.mkdir(parents=True, exist_ok=True)

    n_linked = 0
    for bird_id in bird_ids:
        ts = _timestamp_suffix(bird_id)
        if ts is None or ts not in eef_by_ts:
            continue
        src = eef_by_ts[ts]
        # Preserve a suffix that demo_id_from_robot_eef_npz strips correctly.
        dst = staging / f"{bird_id}_arm_fk_pose_targetframe_commonframe.npz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        n_linked += 1

    if n_linked == 0:
        raise FileNotFoundError(
            f"Could not remap EEF NPZs under {eef_dir} to bird demo IDs in {mixed_root}"
        )
    print(f"Remapped {n_linked} EEF NPZs -> {staging} (timestamp match; original prefix differed)")
    return staging

# ---------------------------------------------------------------------------
# Dataset labels / colors
# ---------------------------------------------------------------------------

DATASET_TYPES_COMBINED = ("teleop", "human")
DATASET_TYPES_MIXED = (
    "teleop",
    "human",
    "left_robot_right_hand",
    "right_robot_left_hand",
)
DATASET_COLORS = {
    "teleop": "#1f77b4",
    "human": "#2ca02c",
    "left_robot_right_hand": "#ff7f0e",
    "right_robot_left_hand": "#d62728",
}
DATASET_LABELS = {
    "teleop": "robot (teleop)",
    "human": "human",
    "left_robot_right_hand": "left robot + right hand",
    "right_robot_left_hand": "right robot + left hand",
}
EMBODIMENT_TO_DATASET_TYPES = {
    "robot": ("teleop",),
    "human": ("human",),
    "mixed": ("left_robot_right_hand", "right_robot_left_hand"),
}


@dataclass
class FrameRef:
    """Lightweight pointer into an episode dataset (no image tensors)."""

    dataset_type: str
    embodiment: int
    demo_idx: int
    start_in_ep: int
    progress: float
    # Which loader + builder to use; resolved at extract time.
    source: str  # teleop | human | left_robot_right_hand | right_robot_left_hand
    dataset: Any = None  # episode dataset that owns this frame


class BuildArgs:
    """Minimal args object matching ACT build() expectations."""

    def __init__(self, num_queries: int, backbone: str = "resnet18", camera_names: Sequence[str] | None = None):
        self.num_queries = int(num_queries)
        self.camera_names = list(camera_names or MODEL_CAMERA_NAMES)
        self.hidden_dim = 512
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 3200
        self.enc_layers = 4
        self.dec_layers = 7
        self.pre_norm = False
        self.position_embedding = "sine"
        self.backbone = backbone
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False


# ---------------------------------------------------------------------------
# Pluggable backbone extractors
# ---------------------------------------------------------------------------


class BackboneFeatureExtractor(ABC):
    """Interface for turning images into a single embedding vector per frame."""

    name: str = "base"

    @abstractmethod
    @torch.no_grad()
    def extract(self, images: torch.Tensor, camera_mask: torch.Tensor | None = None) -> np.ndarray:
        """
        Args:
            images: [num_cams, 3, H, W] or [1, num_cams, 3, H, W]
            camera_mask: optional [num_cams] or [1, num_cams] (1=active)
        Returns:
            float32 vector [D]
        """


BIRD_CAM_ID = 0  # MODEL_CAMERA_NAMES[0] == "bird"; wrists are never used for embeddings


class ResNetGAPExtractor(BackboneFeatureExtractor):
    """
    Global-average-pool the final ResNet feature map from the bird-view backbone
    only (model.backbones[0]). Wrist cameras are ignored.
    """

    name = "resnet_gap"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def extract(self, images: torch.Tensor, camera_mask: torch.Tensor | None = None) -> np.ndarray:
        del camera_mask  # bird-only; mask unused
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.to(self.device, non_blocking=False)
        if images.shape[0] != 1:
            raise ValueError(f"ResNetGAPExtractor expects batch size 1, got {images.shape[0]}")

        features, _ = self.model.backbones[BIRD_CAM_ID](images[:, BIRD_CAM_ID])
        gap = features[0].mean(dim=(2, 3))[0]  # [C]
        out = gap.detach().float().cpu().numpy().astype(np.float32)
        del images, features, gap
        return out


class InputProjGAPExtractor(BackboneFeatureExtractor):
    """Bird-view backbone + input_proj, then GAP. Wrist cameras ignored."""

    name = "input_proj_gap"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def extract(self, images: torch.Tensor, camera_mask: torch.Tensor | None = None) -> np.ndarray:
        del camera_mask
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.to(self.device, non_blocking=False)
        if images.shape[0] != 1:
            raise ValueError(f"InputProjGAPExtractor expects batch size 1, got {images.shape[0]}")

        features, _ = self.model.backbones[BIRD_CAM_ID](images[:, BIRD_CAM_ID])
        proj = self.model.input_proj(features[0])
        gap = proj.mean(dim=(2, 3))[0]
        out = gap.detach().float().cpu().numpy().astype(np.float32)
        del images, features, proj, gap
        return out


class ActTrunkExtractor:
    """
    Shared prep for ACT vision+proprio → transformer.

    - encoder_memory: mean-pool transformer encoder output over all tokens
      (latent + proprio + spatial), shape [H]
    - decoder_hs: mean-pool decoder query tokens before action heads, shape [H]

    Inference-style: zero CVAE latent; bird-only camera mask (wrists off).
    """

    def __init__(self, model: torch.nn.Module, *, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def extract(
        self,
        *,
        images: torch.Tensor,
        pose_state: torch.Tensor,
        joint_state: torch.Tensor,
        embodiment: int,
        camera_mask: torch.Tensor,
        supports_mixed: bool,
        want_encoder: bool = False,
        want_decoder: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not want_encoder and not want_decoder:
            return None, None

        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.to(self.device, non_blocking=False)
        pose_state = pose_state.unsqueeze(0).to(self.device) if pose_state.ndim == 1 else pose_state.to(self.device)
        joint_state = joint_state.unsqueeze(0).to(self.device) if joint_state.ndim == 1 else joint_state.to(self.device)
        camera_mask = camera_mask.unsqueeze(0).to(self.device) if camera_mask.ndim == 1 else camera_mask.to(self.device)

        emb_id = int(embodiment)
        if emb_id == EMBODIMENT_MIXED and not supports_mixed:
            emb_id = EMBODIMENT_ROBOT

        bs = images.shape[0]
        latent_sample = torch.zeros(bs, self.model.latent_dim, dtype=torch.float32, device=self.device)
        latent_input = self.model.latent_out_proj(latent_sample)

        all_cam_features = []
        all_cam_pos = []
        for cam_id in range(images.shape[1]):
            features, pos = self.model.backbones[cam_id](images[:, cam_id])
            features = features[0]
            pos = pos[0]
            feat = self.model.input_proj(features)
            m = camera_mask[:, cam_id].view(bs, 1, 1, 1)
            feat = feat * m
            pos = pos * camera_mask[0, cam_id].view(1, 1, 1, 1)
            all_cam_features.append(feat)
            all_cam_pos.append(pos)
        src = torch.cat(all_cam_features, dim=3)
        pos = torch.cat(all_cam_pos, dim=3)

        # This repo's human path omits the proprio token (no human_input_proj).
        # Older pose-proj / Omar-const checkpoints still prepend a human token.
        human_proj = getattr(self.model, "human_input_proj", None)
        if emb_id in (EMBODIMENT_ROBOT, EMBODIMENT_MIXED):
            additional_inputs = [latent_input, self.model.robot_input_proj(joint_state)]
            additional_pos_w = self.model.additional_pos_embed.weight
        elif isinstance(human_proj, torch.nn.Linear) or hasattr(self.model, "human_input_const"):
            additional_inputs = [latent_input, human_proj(pose_state)]
            additional_pos_w = self.model.additional_pos_embed.weight
        else:
            additional_inputs = [latent_input]
            additional_pos_w = self.model.additional_pos_embed.weight[:1]

        # Mirror Transformer.forward through the encoder.
        tf = self.model.transformer
        bs2, _c, _h, _w = src.shape
        src_tok = src.flatten(2).permute(2, 0, 1)  # [HW,B,H]
        pos_tok = pos.flatten(2).permute(2, 0, 1).repeat(1, bs2, 1)
        query_embed = self.model.query_embed.weight.unsqueeze(1).repeat(1, bs2, 1)
        additional_pos = additional_pos_w.unsqueeze(1).repeat(1, bs2, 1)
        pos_full = torch.cat([additional_pos, pos_tok], dim=0)
        addition_input = torch.stack(additional_inputs, dim=0)
        src_full = torch.cat([addition_input, src_tok], dim=0)
        memory = tf.encoder(src_full, src_key_padding_mask=None, pos=pos_full)  # [S,B,H]

        enc_out = None
        if want_encoder:
            enc_out = memory.mean(dim=0)[0].detach().float().cpu().numpy().astype(np.float32)

        dec_out = None
        if want_decoder:
            # Decoder returns [L, K, B, H]; Transformer.forward does transpose(1,2) → [L, B, K, H]
            tgt = torch.zeros_like(query_embed)
            hs = tf.decoder(
                tgt,
                memory,
                memory_key_padding_mask=None,
                pos=pos_full,
                query_pos=query_embed,
            )
            hs = hs.transpose(1, 2)[0]  # [B, K, H] — same indexing as core.forward
            dec_out = hs.mean(dim=1)[0].detach().float().cpu().numpy().astype(np.float32)
            del hs

        del images, memory, src, pos, all_cam_features, all_cam_pos
        return enc_out, dec_out


EXTRACTOR_REGISTRY: dict[str, type[BackboneFeatureExtractor]] = {
    "resnet_gap": ResNetGAPExtractor,
    "input_proj_gap": InputProjGAPExtractor,
}


def build_extractor(
    kind: str,
    model: torch.nn.Module,
    *,
    device: torch.device | None = None,
) -> BackboneFeatureExtractor:
    if kind not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor '{kind}'. Choose from {sorted(EXTRACTOR_REGISTRY)}")
    return EXTRACTOR_REGISTRY[kind](model, device=device)


# ---------------------------------------------------------------------------
# Path / model helpers
# ---------------------------------------------------------------------------


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def checkpoint_human_proprio_kind(state: dict) -> str:
    """
    Infer checkpoint human-proprio layout from keys.

    Returns:
      - 'pose_proj' : has pose Linear projs (human_input_proj / human_cvae_state_proj)
      - 'const'     : Omar-style learned constants (human_input_const / human_cvae_state_const)
      - 'none'      : no human proprio adapters at all
    """
    if "human_input_const" in state or "human_cvae_state_const" in state:
        return "const"
    if "human_input_proj.weight" in state or "human_cvae_state_proj.weight" in state:
        return "pose_proj"
    return "none"


def checkpoint_is_no_pose_proprio(state: dict) -> bool:
    """Omar-style: human proprio/CVAE-state are learned constants, not pose Linears."""
    return checkpoint_human_proprio_kind(state) == "const"


def checkpoint_is_no_human_proprio(state: dict) -> bool:
    """No human proprio adapters at all (no pose projs, no learned constants)."""
    return checkpoint_human_proprio_kind(state) == "none"


def convert_model_to_no_pose_proprio(model: torch.nn.Module) -> torch.nn.Module:
    """Replace human_input_proj / human_cvae_state_proj with human_*_const parameters."""
    hidden_dim = int(model.hidden_dim)
    ref = next(model.parameters())
    device, dtype = ref.device, ref.dtype
    if hasattr(model, "human_input_proj"):
        del model.human_input_proj
    if hasattr(model, "human_cvae_state_proj"):
        del model.human_cvae_state_proj
    if not hasattr(model, "human_input_const"):
        model.register_parameter(
            "human_input_const",
            torch.nn.Parameter(torch.zeros(hidden_dim, device=device, dtype=dtype)),
        )
    if not hasattr(model, "human_cvae_state_const"):
        model.register_parameter(
            "human_cvae_state_const",
            torch.nn.Parameter(torch.zeros(hidden_dim, device=device, dtype=dtype)),
        )
    return model


def install_no_pose_proprio_callables(model: torch.nn.Module) -> torch.nn.Module:
    """Keep human_*_proj call sites working (expand learned constants; ignore pose_state)."""

    def human_input_proj(pose_state: torch.Tensor) -> torch.Tensor:
        return model.human_input_const.unsqueeze(0).expand(pose_state.shape[0], -1)

    def human_cvae_state_proj(pose_state: torch.Tensor) -> torch.Tensor:
        return model.human_cvae_state_const.unsqueeze(0).expand(pose_state.shape[0], -1)

    model.human_input_proj = human_input_proj  # type: ignore[assignment]
    model.human_cvae_state_proj = human_cvae_state_proj  # type: ignore[assignment]
    return model


def convert_model_to_no_human_proprio(model: torch.nn.Module) -> torch.nn.Module:
    """Remove all human state adapters so strict load matches no-human-proprio checkpoints."""
    for name in (
        "human_input_proj",
        "human_cvae_state_proj",
        "human_input_const",
        "human_cvae_state_const",
    ):
        if hasattr(model, name):
            delattr(model, name)
    return model


def install_no_human_proprio_callables(model: torch.nn.Module) -> torch.nn.Module:
    """Provide zero-proprio stubs so trunk extraction can run on human frames."""
    hidden_dim = int(model.hidden_dim)

    def human_input_proj(pose_state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (pose_state.shape[0], hidden_dim),
            device=pose_state.device,
            dtype=pose_state.dtype,
        )

    def human_cvae_state_proj(pose_state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (pose_state.shape[0], hidden_dim),
            device=pose_state.device,
            dtype=pose_state.dtype,
        )

    model.human_input_proj = human_input_proj  # type: ignore[assignment]
    model.human_cvae_state_proj = human_cvae_state_proj  # type: ignore[assignment]
    return model


def load_model_weights(
    model: torch.nn.Module,
    state: dict,
    *,
    no_pose_mode: str = "auto",
) -> str:
    """
    Load checkpoint into model.

    Returns the resolved human proprio layout: 'pose_proj' | 'const' | 'none'

    no_pose_mode:
      - 'auto' : infer from checkpoint keys
      - 'on'   : force Omar-style 'const'
      - 'off'  : force 'pose_proj'
      - 'none' : force 'none'
    """
    if no_pose_mode == "auto":
        kind = checkpoint_human_proprio_kind(state)
    elif no_pose_mode == "on":
        kind = "const"
    elif no_pose_mode == "off":
        kind = "pose_proj"
    elif no_pose_mode == "none":
        kind = "none"
    else:
        raise ValueError(f"Unknown no_pose_mode={no_pose_mode!r}")

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
        kind = checkpoint_human_proprio_kind(state) if no_pose_mode == "auto" else kind

    if kind == "const":
        convert_model_to_no_pose_proprio(model)
        model.load_state_dict(state, strict=True)
        install_no_pose_proprio_callables(model)
    elif kind == "none":
        convert_model_to_no_human_proprio(model)
        model.load_state_dict(state, strict=True)
        install_no_human_proprio_callables(model)
    else:
        model.load_state_dict(state, strict=True)
    return kind


def infer_model_pkg(checkpoint: Path) -> str:
    parts = checkpoint.resolve().parts
    for name in (
        "MixedEmbodiment_4cam",
        "MixedEmbodiment_dual",
        "MixedEmbodiment",
        "Combined_relative_3cam",
        "Combined_relative_xyz_3cam",
        "Combined_relative",
    ):
        if name in parts:
            return name
    return "MixedEmbodiment"


def load_build_fn(model_pkg: str):
    mod = importlib.import_module(f"{model_pkg}.core")
    return mod.build


def dataset_types_for_pkg(model_pkg: str, *, include_mixed: bool) -> tuple[str, ...]:
    if include_mixed and model_pkg.startswith("MixedEmbodiment"):
        return DATASET_TYPES_MIXED
    return DATASET_TYPES_COMBINED


# ---------------------------------------------------------------------------
# Lightweight frame refs + on-demand image decode
# ---------------------------------------------------------------------------


def _progress(start: int, ep_len: int) -> float:
    return float(start) / float(max(1, ep_len - 1)) if ep_len > 1 else 0.0


def collect_strided_refs(
    *,
    robot_ds: RobotEpisodeDataset | None,
    human_ds: HumanEpisodeDataset | None,
    mixed_lr: MixedEpisodeDataset | None,
    mixed_rl: MixedEpisodeDataset | None,
    frame_stride: int,
    max_frames_per_type: int | None,
    include_mixed: bool,
    mixed_datasets: list[tuple[Any, str]] | None = None,
    skip_first_s: float = 0.0,
    frame_hz: float = 15.0,
) -> list[FrameRef]:
    """Index every Nth frame — metadata only, no image tensors."""
    stride = max(1, int(frame_stride))
    skip_n = int(round(max(0.0, float(skip_first_s)) * float(frame_hz))) if float(skip_first_s) > 0 else 0

    def from_episode_ds(ds, dataset_type: str, embodiment: int) -> list[FrameRef]:
        out: list[FrameRef] = []
        skipped_eps = 0
        for ep in range(int(ds.num_demos)):
            ep_len = int(ds.demo_lengths[ep])
            if skip_n >= ep_len:
                skipped_eps += 1
                continue
            for start in range(skip_n, ep_len, stride):
                out.append(
                    FrameRef(
                        dataset_type=dataset_type,
                        embodiment=embodiment,
                        demo_idx=ep,
                        start_in_ep=start,
                        progress=_progress(start, ep_len),
                        source=dataset_type,
                        dataset=ds,
                    )
                )
                if max_frames_per_type is not None and len(out) >= max_frames_per_type:
                    return out
        if skip_n > 0:
            print(
                f"  {dataset_type}: skip_first_s={skip_first_s:g}s (~{skip_n} frames @ {frame_hz:g} Hz); "
                f"dropped {skipped_eps}/{int(ds.num_demos)} demos shorter than the skip",
                flush=True,
            )
        return out

    refs: list[FrameRef] = []
    if robot_ds is not None:
        refs += from_episode_ds(robot_ds, "teleop", EMBODIMENT_ROBOT)
    if human_ds is not None:
        refs += from_episode_ds(human_ds, "human", EMBODIMENT_HUMAN)
    if include_mixed:
        pairs = list(mixed_datasets or [])
        if not pairs:
            if mixed_lr is not None:
                pairs.append((mixed_lr, "left_robot_right_hand"))
            if mixed_rl is not None:
                pairs.append((mixed_rl, "right_robot_left_hand"))
        if not pairs:
            raise ValueError("Mixed datasets required when include_mixed=True")
        for ds, label in pairs:
            refs += from_episode_ds(ds, label, EMBODIMENT_MIXED)
    return refs


def _resolve_ds_for_ref(
    ref: FrameRef,
    *,
    robot_ds: RobotEpisodeDataset | None,
    human_ds: HumanEpisodeDataset | None,
    mixed_lr: MixedEpisodeDataset | None,
    mixed_rl: MixedEpisodeDataset | None,
):
    if ref.dataset is not None:
        return ref.dataset
    if ref.source == "teleop":
        return robot_ds
    if ref.source == "human":
        return human_ds
    if ref.source == "left_robot_right_hand":
        return mixed_lr
    if ref.source == "right_robot_left_hand":
        return mixed_rl
    raise ValueError(f"Unknown source {ref.source}")


def load_obs_for_ref(
    ref: FrameRef,
    *,
    robot_ds: RobotEpisodeDataset | None,
    human_ds: HumanEpisodeDataset | None,
    mixed_lr: MixedEpisodeDataset | None,
    mixed_rl: MixedEpisodeDataset | None,
    n_cameras: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Decode bird RGB + proprio for one frame.
    Returns (images [C,3,H,W], camera_mask [C] wrists-off, pose_state [8], joint_state [14], embodiment).
    Wrist RGB is always zeroed. Proprio is normalized with that dataset's stats.
    3-cam: [bird, 0, 0]. 4-cam: [bird, 0, 0, front-or-0].
    """
    ds = _resolve_ds_for_ref(
        ref, robot_ds=robot_ds, human_ds=human_ds, mixed_lr=mixed_lr, mixed_rl=mixed_rl
    )
    assert ds is not None
    sample_idx = ds.demo_start_idx[ref.demo_idx] + ref.start_in_ep
    bird_np = load_frame(ds.bird_frames[sample_idx])
    zero_np = zero_rgb_like(bird_np)
    bird_t = ds.image_transform(bird_np)
    zero_t = ds.image_transform(zero_np)
    if int(n_cameras) <= 3:
        images = stack_camera_tensors(bird_t, zero_t, zero_t)
        cam_mask = camera_mask_tensor(EMBODIMENT_HUMAN)
    else:
        front_frames = getattr(ds, "front_frames", None)
        front_t = (
            ds.image_transform(load_frame(front_frames[sample_idx]))
            if front_frames is not None
            else zero_t
        )
        slots = [bird_t, zero_t, zero_t, front_t]
        while len(slots) < int(n_cameras):
            slots.append(zero_t)
        images = torch.stack(slots[: int(n_cameras)], dim=0)
        cam_mask = torch.zeros(int(n_cameras), dtype=torch.float32)
        cam_mask[0] = 1.0
        if front_frames is not None and int(n_cameras) >= 4:
            cam_mask[3] = 1.0

    if ref.source == "teleop":
        pose_raw = ds.eef_pose_data[sample_idx]
        joint_raw = ds.joint_data[sample_idx]
        pose_state = (pose_raw - ds.eef_abs_mean) / ds.eef_abs_std.clamp_min(1e-2)
        joint_state = (joint_raw - ds.joint_mean) / ds.joint_std.clamp_min(1e-2)
        embodiment = EMBODIMENT_ROBOT
    elif ref.source == "human":
        pose_raw = ds.pose_data[sample_idx]
        pose_state = (pose_raw - ds.pose_abs_mean) / ds.pose_abs_std.clamp_min(1e-2)
        joint_state = torch.zeros(ROBOT_JOINT_DIM, dtype=torch.float32)
        embodiment = EMBODIMENT_HUMAN
    else:
        pose_raw = ds.pose_data[sample_idx]
        joint_raw = ds.joint_data[sample_idx]
        pose_state = (pose_raw - ds.pose_abs_mean) / ds.pose_abs_std.clamp_min(1e-2)
        joint_state = (joint_raw - ds.joint_mean) / ds.joint_std.clamp_min(1e-2)
        embodiment = EMBODIMENT_MIXED

    return (
        images,
        cam_mask,
        pose_state.float(),
        joint_state.float(),
        int(embodiment),
    )


@torch.no_grad()
def extract_embeddings_streaming(
    refs: list[FrameRef],
    *,
    resnet_extractor: BackboneFeatureExtractor,
    trunk_extractor: ActTrunkExtractor | None,
    want_encoder: bool,
    want_decoder: bool,
    robot_ds: RobotEpisodeDataset | None,
    human_ds: HumanEpisodeDataset | None,
    mixed_lr: MixedEpisodeDataset | None,
    mixed_rl: MixedEpisodeDataset | None,
    supports_mixed: bool,
    progress_every: int = 25,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Decode one bird frame at a time.
    Always returns ResNet GAP; optionally encoder memory and/or decoder hs.
    """
    resnet_embs: list[np.ndarray] = []
    encoder_embs: list[np.ndarray] | None = [] if want_encoder else None
    decoder_embs: list[np.ndarray] | None = [] if want_decoder else None
    n = len(refs)
    n_cameras = len(getattr(resnet_extractor, "model").backbones)
    for i, ref in enumerate(refs):
        images, cam_mask, pose_state, joint_state, embodiment = load_obs_for_ref(
            ref,
            robot_ds=robot_ds,
            human_ds=human_ds,
            mixed_lr=mixed_lr,
            mixed_rl=mixed_rl,
            n_cameras=n_cameras,
        )
        resnet_embs.append(resnet_extractor.extract(images, cam_mask))
        if trunk_extractor is not None and (want_encoder or want_decoder):
            enc_v, dec_v = trunk_extractor.extract(
                images=images,
                pose_state=pose_state,
                joint_state=joint_state,
                embodiment=embodiment,
                camera_mask=cam_mask,
                supports_mixed=supports_mixed,
                want_encoder=want_encoder,
                want_decoder=want_decoder,
            )
            if encoder_embs is not None and enc_v is not None:
                encoder_embs.append(enc_v)
            if decoder_embs is not None and dec_v is not None:
                decoder_embs.append(dec_v)
        del images
        if (i + 1) % progress_every == 0 or i == 0 or i + 1 == n:
            parts = ["bird ResNet"]
            if want_encoder:
                parts.append("encoder")
            if want_decoder:
                parts.append("decoder")
            print(f"  extracted {i + 1}/{n} ({'+'.join(parts)})", flush=True)
        if (i + 1) % 100 == 0:
            gc.collect()
    resnet_arr = np.stack(resnet_embs, axis=0)
    encoder_arr = np.stack(encoder_embs, axis=0) if encoder_embs is not None else None
    decoder_arr = np.stack(decoder_embs, axis=0) if decoder_embs is not None else None
    return resnet_arr, encoder_arr, decoder_arr


def reduce_and_plot(
    feat_arr: np.ndarray,
    *,
    types: list[str],
    dataset_order: Sequence[str],
    ckpt_stem: str,
    feat_name: str,
    ckpt_dir: Path,
    progress: np.ndarray,
    demo_idx: np.ndarray,
    start_in_ep: np.ndarray,
    seed: int,
    perplexity: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
) -> dict[str, np.ndarray]:
    """Fit PCA/t-SNE/UMAP on feat_arr and write embeddings + scatter plots."""
    print(f"  [{feat_name}] feature matrix: {feat_arr.shape}", flush=True)
    pca_xy = run_pca(feat_arr, random_state=seed)
    tsne_xy = run_tsne(feat_arr, perplexity=perplexity, random_state=seed)
    umap_xy = run_umap(
        feat_arr,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=seed,
    )
    np.savez_compressed(
        ckpt_dir / f"{feat_name}_embeddings_2d.npz",
        pca=pca_xy,
        tsne=tsne_xy,
        umap=umap_xy,
        dataset_type=np.asarray(types),
        progress=progress,
        demo_idx=demo_idx,
        start_in_ep=start_in_ep,
        perplexity=np.float32(perplexity),
        umap_n_neighbors=np.int32(umap_n_neighbors),
        umap_min_dist=np.float32(umap_min_dist),
        random_state=np.int32(seed),
    )
    for method, xy, xlab, ylab in (
        ("pca", pca_xy, "PC1", "PC2"),
        ("tsne", tsne_xy, "t-SNE 1", "t-SNE 2"),
        ("umap", umap_xy, "UMAP 1", "UMAP 2"),
    ):
        fig_path = ckpt_dir / f"{feat_name}_{method}_by_dataset.png"
        scatter_by_dataset(
            xy,
            types,
            dataset_order=dataset_order,
            title=f"{ckpt_stem} | {feat_name} {method.upper()} (by dataset)",
            out_path=fig_path,
            xlabel=xlab,
            ylabel=ylab,
        )
        print(f"  wrote {fig_path}", flush=True)
    return {"pca": pca_xy, "tsne": tsne_xy, "umap": umap_xy}


# ---------------------------------------------------------------------------
# 2D reducers + plotting
# ---------------------------------------------------------------------------


def run_tsne(
    features: np.ndarray,
    *,
    perplexity: float = 30.0,
    random_state: int = 0,
    n_components: int = 2,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be [N,D], got {x.shape}")
    n = x.shape[0]
    perp = min(float(perplexity), max(5.0, (n - 1) / 3.0))
    if n < 4:
        raise RuntimeError(f"Need at least 4 samples for t-SNE, got {n}")
    print(f"  TSNE: N={n} D={x.shape[1]} perplexity={perp} random_state={random_state}", flush=True)
    reducer = TSNE(
        n_components=int(n_components),
        perplexity=perp,
        random_state=int(random_state),
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(x).astype(np.float32)


def run_pca(
    features: np.ndarray,
    *,
    random_state: int = 0,
    n_components: int = 2,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be [N,D], got {x.shape}")
    n = x.shape[0]
    n_comp = min(int(n_components), n, x.shape[1])
    if n_comp < 2:
        raise RuntimeError(f"Need at least 2 samples/dims for PCA, got N={n} D={x.shape[1]}")
    print(f"  PCA: N={n} D={x.shape[1]} n_components={n_comp} random_state={random_state}", flush=True)
    reducer = PCA(n_components=n_comp, random_state=int(random_state))
    return reducer.fit_transform(x).astype(np.float32)


def run_umap(
    features: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
    n_components: int = 2,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be [N,D], got {x.shape}")
    n = x.shape[0]
    if n < 4:
        raise RuntimeError(f"Need at least 4 samples for UMAP, got {n}")
    # n_neighbors must be < N
    neighbors = max(2, min(int(n_neighbors), n - 1))
    print(
        f"  UMAP: N={n} D={x.shape[1]} n_neighbors={neighbors} min_dist={min_dist} "
        f"random_state={random_state}",
        flush=True,
    )
    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=neighbors,
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state),
    )
    return reducer.fit_transform(x).astype(np.float32)


def _scatter_by_dataset_ax(
    ax: plt.Axes,
    xy: np.ndarray,
    types: Sequence[str],
    *,
    dataset_order: Sequence[str],
    title: str,
    xlabel: str = "dim 1",
    ylabel: str = "dim 2",
    show_legend: bool = True,
) -> None:
    for dt in dataset_order:
        mask = np.asarray([t == dt for t in types], dtype=bool)
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=22,
            c=DATASET_COLORS[dt],
            label=DATASET_LABELS[dt],
            alpha=0.8,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)


def scatter_by_dataset(
    xy: np.ndarray,
    types: Sequence[str],
    *,
    dataset_order: Sequence[str],
    title: str,
    out_path: Path,
    xlabel: str = "dim 1",
    ylabel: str = "dim 2",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    _scatter_by_dataset_ax(
        ax,
        xy,
        types,
        dataset_order=dataset_order,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show_legend=True,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _union_dataset_order(type_lists: Sequence[Sequence[str]]) -> tuple[str, ...]:
    present = set()
    for types in type_lists:
        present |= set(types)
    out: list[str] = []
    # Prefer mixed order (superset), then any extras in sorted order.
    for dt in DATASET_TYPES_MIXED:
        if dt in present:
            out.append(dt)
    extras = sorted(present - set(out))
    out += extras
    return tuple(out)


def save_side_by_side_grid(
    *,
    out_dir: Path,
    feat_name: str,
    ckpt_titles: list[str],
    type_lists: list[list[str]],
    xy_by_ckpt: list[dict[str, np.ndarray]],
    seed: int,
    perplexity: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
) -> Path:
    """
    Save a 3xN grid (rows: PCA/t-SNE/UMAP, cols: checkpoints) for one feature.
    """
    n_ckpts = len(ckpt_titles)
    if n_ckpts < 2:
        raise ValueError("Need >=2 checkpoints for side-by-side grid")
    methods = ("pca", "tsne", "umap")
    row_labels = ("PCA", "t-SNE", "UMAP")
    axis_labels = (("PC1", "PC2"), ("t-SNE 1", "t-SNE 2"), ("UMAP 1", "UMAP 2"))
    dataset_order = _union_dataset_order(type_lists)

    fig_w = 5.2 * n_ckpts
    fig_h = 3.9 * len(methods)
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=n_ckpts,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        squeeze=False,
    )
    for col, title in enumerate(ckpt_titles):
        axes[0, col].set_title(title)
    for row, (method, row_lab, (xlab, ylab)) in enumerate(zip(methods, row_labels, axis_labels)):
        for col in range(n_ckpts):
            ax = axes[row, col]
            xy = xy_by_ckpt[col][method]
            _scatter_by_dataset_ax(
                ax,
                xy,
                type_lists[col],
                dataset_order=dataset_order,
                title="",  # column titles handled above
                xlabel=xlab if row == len(methods) - 1 else "",
                ylabel=ylab if col == 0 else "",
                show_legend=False,
            )
        axes[row, 0].set_ylabel(f"{row_lab} ({axis_labels[row][1]})")

    handles = []
    for dt in dataset_order:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=DATASET_COLORS[dt],
                markeredgecolor="none",
                label=DATASET_LABELS[dt],
                alpha=0.9,
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=9)
    fig.suptitle(
        f"{feat_name} side-by-side (seed={seed}, perplexity={perplexity:g}, "
        f"umap_n_neighbors={umap_n_neighbors}, umap_min_dist={umap_min_dist:g})",
        fontsize=12,
    )
    out_path = out_dir / f"side_by_side_{feat_name}_{n_ckpts}ckpts.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE of ResNet GAP backbone embeddings across embodiment datasets"
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more .pth files (raw state_dict or training-state with model_state_dict)",
    )
    p.add_argument(
        "--model_pkgs",
        nargs="*",
        default=None,
        help="Optional override, parallel to --checkpoints",
    )
    p.add_argument(
        "--no_pose_proprio",
        choices=("auto", "on", "off", "none"),
        default="auto",
        help=(
            "Human proprio layout: auto-detect pose-proj vs Omar-const vs none, "
            "or force 'on'(const) / 'off'(pose_proj) / 'none' for all --checkpoints"
        ),
    )
    p.add_argument(
        "--sessions_root",
        type=str,
        default=str(DEFAULT_SESSIONS_ROOT),
        help="sessions/<kind>/<date> tree (same as training). Used when per-modality roots are omitted.",
    )
    p.add_argument(
        "--embodiments",
        type=str,
        default="auto",
        help=(
            "auto (default): load the union of each checkpoint's run_metadata.json "
            "embodiments and plot each ckpt with only what it was trained on. "
            "Or pass the same list for every checkpoint, e.g. robot,human or robot,human,mixed."
        ),
    )
    p.add_argument("--teleop_root", type=str, default=None, help="Override robot date root")
    p.add_argument("--human_root", type=str, default=None, help="Override human date root")
    p.add_argument(
        "--mixed_lr_root",
        type=str,
        default=None,
        help="Override left-robot + right-hand date root (repeatable via sessions_root discovery)",
    )
    p.add_argument(
        "--mixed_rl_root",
        type=str,
        default=None,
        help="Override right-robot + left-hand date root",
    )
    p.add_argument(
        "--frame_stride",
        type=int,
        default=20,
        help="Sample every Nth synced frame within each demo (default: 20, light on RAM)",
    )
    p.add_argument(
        "--max_frames_per_type",
        type=int,
        default= 0,
        help="Cap samples per dataset type (default 60; 0 = no cap)",
    )
    p.add_argument(
        "--max_demos",
        type=int,
        default=2,
        help="Per dataset type (default 2; 0=all). Keep small on ~16GB machines.",
    )
    p.add_argument(
        "--skip_first_s",
        type=float,
        default=0.0,
        help=(
            "Drop the first N seconds of each demo before sampling "
            "(synced rows are ~15 Hz bird; use --frame_hz to override). "
            "Demos shorter than this contribute no points."
        ),
    )
    p.add_argument(
        "--frame_hz",
        type=float,
        default=15.0,
        help="Hz used to convert --skip_first_s into a frame offset (default 15, the recorded bird rate)",
    )
    p.add_argument(
        "--shared_frames_only",
        action="store_true",
        help=(
            "Only use pure teleop + pure human demos (skip mixed). "
            "All checkpoints then see the same frame set for fairer comparison."
        ),
    )
    p.add_argument("--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
    p.add_argument("--max_skew_s", type=float, default=0.05)
    p.add_argument(
        "--jpeg_in_ram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store synced frames as JPEG bytes (much smaller than raw RGB)",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--cpu_threads",
        type=int,
        default=2,
        help="Cap torch/OMP threads to avoid CPU thrash (default 2)",
    )
    p.add_argument(
        "--extractor",
        type=str,
        default="resnet_gap",
        choices=sorted(EXTRACTOR_REGISTRY),
        help="Which BackboneFeatureExtractor to use (always bird-view only)",
    )
    p.add_argument(
        "--include_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Extract transformer encoder memory (mean-pooled over latent+proprio+spatial "
            "tokens) and write PCA/t-SNE/UMAP for it (default: on). Use --no-include_encoder to skip."
        ),
    )
    p.add_argument(
        "--include_decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Extract ACT decoder query features (hs mean-pooled over K queries, "
            "before pose/joint action heads) and write PCA/t-SNE/UMAP for them "
            "(default: on). Use --no-include_decoder to skip."
        ),
    )
    p.add_argument(
        "--side_by_side",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When passing 2–3 checkpoints, also write a 3xN side-by-side grid "
            "per feature (rows PCA/t-SNE/UMAP, columns checkpoints)."
        ),
    )
    p.add_argument("--backbone", type=str, default="resnet18", help="Passed to model build()")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="Shared random_state for PCA / t-SNE / UMAP")
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/resnet_tsne",
    )
    return p.parse_args()


def parse_embodiments(spec: str) -> set[str]:
    requested = {tok.strip().lower() for tok in spec.split(",") if tok.strip()}
    unknown = requested - set(EMBODIMENT_NAMES)
    if unknown:
        raise ValueError(f"Unknown --embodiments entries {sorted(unknown)}; choose from {EMBODIMENT_NAMES}")
    if not requested:
        raise ValueError("--embodiments must name at least one of robot,human,mixed")
    return requested


def checkpoint_trained_embodiments(checkpoint: Path) -> set[str] | None:
    """Read embodiments from the sibling run_metadata.json written at train time."""
    meta_path = checkpoint.parent / "run_metadata.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    embs = {str(x).strip().lower() for x in (meta.get("embodiments") or []) if str(x).strip()}
    embs &= set(EMBODIMENT_NAMES)
    return embs or None


def dataset_types_for_embodiments(embs: set[str]) -> set[str]:
    out: set[str] = set()
    for name in embs:
        out.update(EMBODIMENT_TO_DATASET_TYPES.get(name, ()))
    return out


def filter_refs_for_embodiments(refs: list[FrameRef], embs: set[str]) -> list[FrameRef]:
    allowed = dataset_types_for_embodiments(embs)
    return [r for r in refs if r.dataset_type in allowed]


def _count_refs(refs: list[FrameRef]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in refs:
        out[r.dataset_type] = out.get(r.dataset_type, 0) + 1
    return out


def main() -> None:
    args = parse_args()

    # Keep CPU fan-out low so the machine stays responsive while decoding.
    n_threads = max(1, int(args.cpu_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    torch.set_num_threads(n_threads)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    max_demos = None if int(args.max_demos) <= 0 else int(args.max_demos)
    max_per_type = None if int(args.max_frames_per_type) <= 0 else int(args.max_frames_per_type)
    k = int(args.num_queries)

    ckpts = [resolve_path(c) for c in args.checkpoints]
    ckpt_trained_embs = [checkpoint_trained_embodiments(c) for c in ckpts]
    if str(args.embodiments).strip().lower() == "auto":
        inferred = [e for e in ckpt_trained_embs if e]
        active = set().union(*inferred) if inferred else set(EMBODIMENT_NAMES)
    else:
        active = parse_embodiments(args.embodiments)
    sessions_root = resolve_path(args.sessions_root)
    discovered = discover_sessions_roots(sessions_root)
    teleop_root = resolve_path(args.teleop_root) if args.teleop_root else discovered["robot_data_root"]
    human_root = resolve_path(args.human_root) if args.human_root else discovered["human_data_root"]
    mixed_roots: list[Path] = []
    if args.mixed_lr_root:
        mixed_roots.append(resolve_path(args.mixed_lr_root))
    if args.mixed_rl_root:
        mixed_roots.append(resolve_path(args.mixed_rl_root))
    if not mixed_roots:
        mixed_roots = list(discovered["mixed_data_roots"] or [])

    if "robot" not in active:
        teleop_root = None
    elif teleop_root is None:
        print(f"WARNING: --embodiments requested 'robot' but no teleop_bimanual/<date> under {sessions_root}; skipping.")
    if "human" not in active:
        human_root = None
    elif human_root is None:
        print(f"WARNING: --embodiments requested 'human' but no human_hands_bimanual_raw/<date> under {sessions_root}; skipping.")
    if "mixed" not in active:
        mixed_roots = []
    elif not mixed_roots:
        print(f"WARNING: --embodiments requested 'mixed' but no left/right_robot_*_hand/<date> under {sessions_root}; skipping.")

    sync_root = REPO_ROOT / "m-synced-csvs" / "resnet_tsne"

    print(f"device={device} cpu_threads={n_threads}", flush=True)
    print(
        f"frame_stride={args.frame_stride} max_demos={max_demos} "
        f"max_frames_per_type={max_per_type} skip_first_s={float(args.skip_first_s):g} "
        f"frame_hz={float(args.frame_hz):g} camera=bird (wrists unused) "
        f"embodiments={sorted(active)} (cli={args.embodiments}) "
        f"shared_frames_only={bool(args.shared_frames_only)}",
        flush=True,
    )
    print(f"extractor={args.extractor} sessions_root={sessions_root}", flush=True)

    robot_ds: RobotEpisodeDataset | None = None
    human_ds: HumanEpisodeDataset | None = None
    if teleop_root is not None:
        robot_eef = resolve_robot_eef_dir(teleop_root, None)
        robot_sync = sync_root / f"{teleop_root.name}_robot_tsne"
        build_robot_sync_csvs(teleop_root, robot_sync, robot_eef, float(args.max_skew_s), max_demos)
        robot_ds = RobotEpisodeDataset(
            bird_vids_dir=teleop_root / "bird-realsense-data" / "mp4",
            front_vids_dir=teleop_root / "front-realsense-data" / "mp4",
            left_arm_vids_dir=teleop_root / "aloha-data" / "left" / "mp4",
            right_arm_vids_dir=teleop_root / "aloha-data" / "right" / "mp4",
            left_joint_data_dir=teleop_root / "joint-data" / "left" / "position",
            right_joint_data_dir=teleop_root / "joint-data" / "right" / "position",
            eef_pose_data_dir=robot_eef,
            sync_csv_dir=robot_sync,
            num_queries=k,
            max_demos=max_demos,
            jpeg_in_ram=bool(args.jpeg_in_ram),
        )
    if human_root is not None:
        human_pose = resolve_human_pose_dir(human_root, None)
        human_sync = sync_root / f"{human_root.name}_human_tsne"
        build_human_sync_csvs(human_root, human_sync, human_pose, float(args.max_skew_s), max_demos)
        human_ds = HumanEpisodeDataset(
            bird_vids_dir=human_root / "bird-realsense-data" / "mp4",
            front_vids_dir=human_root / "front-realsense-data" / "mp4",
            pose_npz_dir=human_pose,
            sync_csv_dir=human_sync,
            num_queries=k,
            max_demos=max_demos,
            jpeg_in_ram=bool(args.jpeg_in_ram),
        )

    mixed_lr_ds: MixedEpisodeDataset | None = None
    mixed_rl_ds: MixedEpisodeDataset | None = None
    mixed_pairs: list[tuple[Any, str]] = []
    model_pkgs = list(args.model_pkgs) if args.model_pkgs else [infer_model_pkg(c) for c in ckpts]
    if len(model_pkgs) != len(ckpts):
        raise ValueError("--model_pkgs length must match --checkpoints")

    load_mixed = ("mixed" in active) and (not bool(args.shared_frames_only)) and bool(mixed_roots)

    if load_mixed:
        eef_staging_root = sync_root / "_eef_remapped"
        for mixed_root in mixed_roots:
            preset = infer_mixed_preset(mixed_root)
            if preset is None:
                raise ValueError(f"Cannot infer robot/hand sides from {mixed_root}")
            robot_side = preset["robot_side"]
            hand_side = preset["hand_side"]
            label = "left_robot_right_hand" if robot_side == "left" else "right_robot_left_hand"
            mixed_sync = sync_root / f"{mixed_root.parent.name}_{mixed_root.name}_mixed_tsne"
            pose_dir = resolve_human_pose_dir(mixed_root, None)
            eef_dir = remap_eef_dir_to_session_ids(
                mixed_root,
                resolve_robot_eef_dir(mixed_root, None),
                eef_staging_root,
            )
            print(f"Mixed {label}: robot_side={robot_side} hand_side={hand_side} root={mixed_root}", flush=True)
            build_mixed_sync_csvs(
                mixed_root,
                mixed_sync,
                robot_side=robot_side,
                hand_side=hand_side,
                pose_dir=pose_dir,
                eef_dir=eef_dir,
                max_skew_s=float(args.max_skew_s),
                max_demos=max_demos,
            )
            child = MixedEpisodeDataset(
                bird_vids_dir=mixed_root / "bird-realsense-data" / "mp4",
                wrist_vids_dir=mixed_root / "aloha-data" / robot_side / "mp4",
                front_vids_dir=mixed_root / "front-realsense-data" / "mp4",
                joint_data_dir=mixed_root / "joint-data" / robot_side / "position",
                hand_pose_npz_dir=pose_dir,
                eef_pose_data_dir=eef_dir,
                sync_csv_dir=mixed_sync,
                robot_side=robot_side,
                hand_side=hand_side,
                num_queries=k,
                max_demos=max_demos,
                jpeg_in_ram=bool(args.jpeg_in_ram),
            )
            mixed_pairs.append((child, label))
            if robot_side == "left" and mixed_lr_ds is None:
                mixed_lr_ds = child
            if robot_side == "right" and mixed_rl_ds is None:
                mixed_rl_ds = child
    elif args.shared_frames_only:
        print("shared_frames_only: skipping mixed demos (teleop + human only)", flush=True)

    if robot_ds is None and human_ds is None and not mixed_pairs:
        raise ValueError(
            f"No datasets loaded for --embodiments={sorted(active)} under {sessions_root}. "
            "Expected teleop_bimanual/<date>, human_hands_bimanual_raw/<date>, and/or "
            "left_robot_right_hand|right_robot_left_hand/<date>."
        )

    refs_shared = collect_strided_refs(
        robot_ds=robot_ds,
        human_ds=human_ds,
        mixed_lr=mixed_lr_ds if load_mixed else None,
        mixed_rl=mixed_rl_ds if load_mixed else None,
        frame_stride=int(args.frame_stride),
        max_frames_per_type=max_per_type,
        include_mixed=load_mixed,
        mixed_datasets=mixed_pairs if load_mixed else None,
        skip_first_s=float(args.skip_first_s),
        frame_hz=float(args.frame_hz),
    )
    refs_combined = [r for r in refs_shared if r.dataset_type in DATASET_TYPES_COMBINED]
    refs_for_mixed_pkg: list[FrameRef] = refs_shared

    print(f"teleop+human refs: {len(refs_combined)} {_count_refs(refs_combined)}", flush=True)
    print(f"all refs: {len(refs_shared)} {_count_refs(refs_shared)}", flush=True)
    print("Streaming extract (1 frame in RAM at a time)...", flush=True)

    summary: dict[str, Any] = {
        "frame_stride": int(args.frame_stride),
        "max_demos": max_demos,
        "max_frames_per_type": max_per_type,
        "skip_first_s": float(args.skip_first_s),
        "frame_hz": float(args.frame_hz),
        "shared_frames_only": bool(args.shared_frames_only),
        "embodiments_cli": str(args.embodiments),
        "embodiments_loaded": sorted(active),
        "include_encoder": bool(args.include_encoder),
        "include_decoder": bool(args.include_decoder),
        "extractor": args.extractor,
        "cameras": "bird",
        "perplexity": float(args.perplexity),
        "umap_n_neighbors": int(args.umap_n_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "seed": int(args.seed),
        "checkpoints": [],
    }

    # Accumulate per-ckpt 2D reductions so we can optionally save side-by-side grids.
    grids: dict[str, dict[str, Any]] = {}
    side_by_side_dir: Path | None = None

    for ckpt, pkg, trained in zip(ckpts, model_pkgs, ckpt_trained_embs):
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        embodiments_are_auto = str(args.embodiments).strip().lower() == "auto"
        if embodiments_are_auto and trained is not None:
            plot_embs = trained & active
        else:
            plot_embs = set(active)
        if pkg.startswith("Combined"):
            plot_embs &= {"robot", "human"}
            refs = refs_combined
        else:
            refs = refs_shared
        refs = filter_refs_for_embodiments(refs, plot_embs)
        use_mixed_types = "mixed" in plot_embs and load_mixed
        if not refs:
            raise ValueError(
                f"No frames for {pkg} :: {ckpt.name} "
                f"(trained={None if trained is None else sorted(trained)}, plotting={sorted(plot_embs)})"
            )
        dataset_order = tuple(
            dt for dt in DATASET_TYPES_MIXED if any(r.dataset_type == dt for r in refs)
        ) or dataset_types_for_pkg(pkg, include_mixed=use_mixed_types)
        types = [r.dataset_type for r in refs]
        progress = np.asarray([r.progress for r in refs], dtype=np.float32)
        demo_idx = np.asarray([r.demo_idx for r in refs], dtype=np.int32)
        start_in_ep = np.asarray([r.start_in_ep for r in refs], dtype=np.int32)
        seed = int(args.seed)
        supports_mixed = pkg.startswith("MixedEmbodiment")
        want_encoder = bool(args.include_encoder)
        want_decoder = bool(args.include_decoder)

        print(f"\n=== {pkg} :: {ckpt.name} ===", flush=True)
        print(
            f"  trained_embodiments={None if trained is None else sorted(trained)} "
            f"plotting={sorted(plot_embs)} frames={len(refs)} types={_count_refs(refs)}",
            flush=True,
        )

        build = load_build_fn(pkg)
        pkg_cfg = importlib.import_module(f"{pkg}.config")
        model = build(BuildArgs(k, backbone=args.backbone, camera_names=pkg_cfg.MODEL_CAMERA_NAMES)).to(device)
        try:
            state = torch.load(str(ckpt), map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(str(ckpt), map_location=device)
        human_proprio_kind = load_model_weights(model, state, no_pose_mode=str(args.no_pose_proprio))
        used_no_pose = human_proprio_kind == "const"
        model.eval()
        del state
        print(
            f"  human_proprio={human_proprio_kind} (mode={args.no_pose_proprio})",
            flush=True,
        )

        resnet_extractor = build_extractor(args.extractor, model, device=device)
        trunk_extractor = (
            ActTrunkExtractor(model, device=device) if (want_encoder or want_decoder) else None
        )

        feat_arr, encoder_arr, decoder_arr = extract_embeddings_streaming(
            refs,
            resnet_extractor=resnet_extractor,
            trunk_extractor=trunk_extractor,
            want_encoder=want_encoder,
            want_decoder=want_decoder,
            robot_ds=robot_ds,
            human_ds=human_ds,
            mixed_lr=mixed_lr_ds,
            mixed_rl=mixed_rl_ds,
            supports_mixed=supports_mixed,
        )

        ckpt_dir = out_dir / f"{pkg}_{ckpt.stem}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if side_by_side_dir is None:
            side_by_side_dir = ckpt_dir

        save_kwargs = dict(
            dataset_type=np.asarray(types),
            progress=progress,
            demo_idx=demo_idx,
            start_in_ep=start_in_ep,
            checkpoint=str(ckpt),
            model_pkg=pkg,
            cameras="bird",
            human_proprio_kind=np.asarray(human_proprio_kind),
            no_pose_proprio=np.asarray(used_no_pose),
        )
        np.savez_compressed(
            ckpt_dir / "features.npz",
            features=feat_arr,
            extractor=args.extractor,
            **save_kwargs,
        )
        if encoder_arr is not None:
            np.savez_compressed(
                ckpt_dir / "encoder_memory_features.npz",
                features=encoder_arr,
                extractor="encoder_memory",
                **save_kwargs,
            )
        if decoder_arr is not None:
            np.savez_compressed(
                ckpt_dir / "decoder_hs_features.npz",
                features=decoder_arr,
                extractor="decoder_hs",
                **save_kwargs,
            )

        plot_kw = dict(
            types=types,
            dataset_order=dataset_order,
            ckpt_stem=ckpt.stem,
            ckpt_dir=ckpt_dir,
            progress=progress,
            demo_idx=demo_idx,
            start_in_ep=start_in_ep,
            seed=seed,
            perplexity=float(args.perplexity),
            umap_n_neighbors=int(args.umap_n_neighbors),
            umap_min_dist=float(args.umap_min_dist),
        )
        feat_xy = reduce_and_plot(feat_arr, feat_name=args.extractor, **plot_kw)
        if encoder_arr is not None:
            enc_xy = reduce_and_plot(encoder_arr, feat_name="encoder_memory", **plot_kw)
        else:
            enc_xy = None
        if decoder_arr is not None:
            dec_xy = reduce_and_plot(decoder_arr, feat_name="decoder_hs", **plot_kw)
        else:
            dec_xy = None

        if bool(args.side_by_side) and 2 <= len(ckpts) <= 3:
            ckpt_title = f"{pkg}:{ckpt.stem}"
            for feat_name, xy in (
                (str(args.extractor), feat_xy),
                ("encoder_memory", enc_xy),
                ("decoder_hs", dec_xy),
            ):
                if xy is None:
                    continue
                grids.setdefault(feat_name, {"titles": [], "types": [], "xy": []})
                grids[feat_name]["titles"].append(ckpt_title)
                grids[feat_name]["types"].append(list(types))
                grids[feat_name]["xy"].append(xy)

        summary["checkpoints"].append(
            {
                "checkpoint": str(ckpt),
                "model_pkg": pkg,
                "human_proprio_kind": str(human_proprio_kind),
                "no_pose_proprio": bool(used_no_pose),
                "out_dir": str(ckpt_dir),
                "n_frames": int(feat_arr.shape[0]),
                "feat_dim": int(feat_arr.shape[1]),
                "encoder_dim": int(encoder_arr.shape[1]) if encoder_arr is not None else None,
                "decoder_dim": int(decoder_arr.shape[1]) if decoder_arr is not None else None,
                "counts": _count_refs(refs),
                "trained_embodiments": None if trained is None else sorted(trained),
                "plotting_embodiments": sorted(plot_embs),
            }
        )

        del model, resnet_extractor, trunk_extractor, feat_arr, encoder_arr, decoder_arr
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote PCA / t-SNE / UMAP outputs to {out_dir}", flush=True)

    if bool(args.side_by_side) and 2 <= len(ckpts) <= 3:
        if side_by_side_dir is None:
            raise RuntimeError("Internal error: side_by_side_dir was not set")
        wrote = {}
        for feat_name, blob in grids.items():
            out_path = save_side_by_side_grid(
                out_dir=side_by_side_dir,
                feat_name=feat_name,
                ckpt_titles=blob["titles"],
                type_lists=blob["types"],
                xy_by_ckpt=blob["xy"],
                seed=int(args.seed),
                perplexity=float(args.perplexity),
                umap_n_neighbors=int(args.umap_n_neighbors),
                umap_min_dist=float(args.umap_min_dist),
            )
            wrote[feat_name] = str(out_path)
            print(f"  wrote {out_path}", flush=True)
        summary["side_by_side"] = wrote
        summary["side_by_side_out_dir"] = str(side_by_side_dir)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
