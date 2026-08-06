"""
MixedEmbodiment DETR-VAE / ACT model (EgoMimic sharing + mixed modality).

Pose action head predicts relative pose deltas vs the chunk's first observation.
Joint head predicts absolute joints for robot and mixed.

Shared:
  - camera backbones + input_proj
  - ACT transformer + CVAE encoder trunk
  - pose_action_head: H -> 8

Not shared (modality routing):
  - robot_input_proj / robot CVAE (robot + mixed)
  - human CVAE action proj (human has no proprio adapter — no observation)
  - joint_action_head (robot + mixed; mixed masks inactive arm in loss)

No embodiment embedding token.

Pose-as-observation (item 8 of the unification request)
---------------------------------------------------------
Robot and mixed NEVER receive pose_state (EEF/hand pose) as an observation —
their proprio (both the transformer's additional token and the CVAE encoder's
state channel) is always joint_state. This was already true in
Combined_relative_3cam_gripweight / MixedEmbodiment_gripweight and is
unchanged here.

Human is the only embodiment whose proprio *could be* pose_state, because a
human demo has no joint encoder. It never is: the human pathway has no
proprio observation at all, structurally, not just a masked-out one.

  - No human_input_proj / human_cvae_state_proj: there is no
    Linear(POSE_DIM, hidden_dim) adapter that reads pose_state as an
    observation.
  - No placeholder in their place either. An earlier version of this model
    stood a learned constant vector (human_input_const / human_cvae_state_const)
    in for "the current human proprio observation" when the real adapter was
    disabled. That constant is the same value every step regardless of
    pose_state, so it carries no information a bias term inside the
    surrounding heads couldn't already supply — it was a token that existed
    only to be a token. Removing it means the human embodiment simply omits
    a proprio token everywhere one would have gone: one fewer token in the
    transformer's extra-token slot (additional_inputs, see model.py) and one
    fewer token in the CVAE encoder's [CLS, state, action] sequence (which
    becomes [CLS, action] for human).
  - pose_state is still accepted by forward() and still populated by
    HumanEpisodeDataset — the dataset needs it as the anchor for computing
    the relative pose_actions target — but the model never reads it as an
    observation. The human pathway predicts the relative pose chunk purely
    from images + the CVAE latent (at train time) or images + z~N(0,I) (at
    inference).

An even earlier version zeroed pose_state and fed it through a Linear layer
that was always constructed. That left a Linear(8, hidden_dim) whose weight
matrix could never receive gradient (0 input -> 0 gradient w.r.t. weight,
every step) for the entire run. The learned-constant version above fixed the
dead-gradient problem but still spent a token and a parameter vector on
carrying zero information. Dropping the token entirely removes both.

Note pose_actions (the *target* the CVAE encoder's action channel sees during
training) are unaffected — that is the standard ACT/CVAE recognition-network
input, discarded at inference when z is sampled from the prior, not an
"observation" in the sense item 8 is about. human_cvae_action_proj (which
encodes pose_actions) is always created.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from MixedEmbodiment.config import (
    EMBODIMENT_HUMAN,
    EMBODIMENT_MIXED,
    EMBODIMENT_ROBOT,
    NUM_CAMERAS,
    POSE_DIM,
    ROBOT_JOINT_DIM,
    validate_camera_names,
)

from MixedEmbodiment.model import (  # noqa: E402
    TransformerEncoder,
    TransformerEncoderLayer,
    build_backbone,
    build_transformer,
)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class MixedDETRVAE(nn.Module):
    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        num_queries,
        camera_names,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.camera_names = list(camera_names)
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.backbones = nn.ModuleList(backbones)

        # Modality-specific proprio adapters (EgoMimic embodiment cue).
        # Robot/mixed always observe joint_state. Human has no proprio
        # observation at all (see module docstring) — no adapter, no
        # placeholder constant; the human forward path below simply omits a
        # proprio token everywhere one would otherwise go.
        self.robot_input_proj = nn.Linear(ROBOT_JOINT_DIM, hidden_dim)  # [B,14] -> [B,H]

        # Shared pose head (human primary + robot aux) and robot-only joint head
        self.pose_action_head = nn.Linear(hidden_dim, POSE_DIM)  # [B,K,H] -> [B,K,8]
        self.joint_action_head = nn.Linear(hidden_dim, ROBOT_JOINT_DIM)  # [B,K,H] -> [B,K,14]
        self.is_pad_head = nn.Linear(hidden_dim, 1)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # CVAE encoder — separate state/action projs per modality (EgoMimic).
        # Human's CVAE encoder sequence has no state token (see _encode_latent).
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.human_cvae_action_proj = nn.Linear(POSE_DIM, hidden_dim)  # always: encodes pose_actions (target), not an observation
        self.robot_cvae_state_proj = nn.Linear(ROBOT_JOINT_DIM, hidden_dim)
        self.robot_cvae_action_proj = nn.Linear(ROBOT_JOINT_DIM, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        # Sized for the longest CVAE encoder sequence (robot/mixed: CLS + state
        # + K action steps). Human's shorter sequence (CLS + K action steps,
        # no state) reuses position 0 (CLS) and positions [2, 2+K) (action
        # steps) from this same table — see _encode_latent — so the shared
        # self.encoder sees the same position id for "first action step" in
        # both cases, and only position 1 ("state") goes unused for human.
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim),
        )
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        # Row 0 = latent token's position; row 1 = proprio token's position
        # (robot/mixed only). Human uses only row 0.
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def _encode_latent(
        self,
        *,
        pose_actions: torch.Tensor,
        joint_state: torch.Tensor,
        joint_actions: torch.Tensor,
        is_pad: torch.Tensor,
        embodiment: int,
    ):
        """
        CVAE prior over the encoder input sequence.

        Robot / mixed path (EgoMimic): [CLS, joint_state, joint_action_chunk].
        Human path: [CLS, pose_action_chunk] — no state token at all (human has
        no proprio observation; see module docstring), so the sequence is one
        token shorter and reuses this module's pos_table at positions {0} u
        {2, ..., 2+K-1}, skipping position 1 (the state slot robot/mixed use).
        """
        bs = is_pad.shape[0]
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        if embodiment in (EMBODIMENT_ROBOT, EMBODIMENT_MIXED):
            if joint_state.shape[-1] != ROBOT_JOINT_DIM:
                raise ValueError(f"joint_state must be [B,{ROBOT_JOINT_DIM}], got {tuple(joint_state.shape)}")
            if joint_actions.shape[-1] != ROBOT_JOINT_DIM:
                raise ValueError(f"joint_actions must be [B,K,{ROBOT_JOINT_DIM}], got {tuple(joint_actions.shape)}")
            state_embed = self.robot_cvae_state_proj(joint_state).unsqueeze(1)
            action_embed = self.robot_cvae_action_proj(joint_actions)
            device = joint_state.device
            encoder_input = torch.cat([cls_embed, state_embed, action_embed], dim=1).permute(1, 0, 2)
            is_pad_full = torch.cat([torch.full((bs, 2), False, device=device), is_pad], dim=1)
            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
        else:
            if pose_actions.shape[-1] != POSE_DIM:
                raise ValueError(f"pose_actions must be [B,K,{POSE_DIM}], got {tuple(pose_actions.shape)}")
            action_embed = self.human_cvae_action_proj(pose_actions)
            device = pose_actions.device
            encoder_input = torch.cat([cls_embed, action_embed], dim=1).permute(1, 0, 2)
            is_pad_full = torch.cat([torch.full((bs, 1), False, device=device), is_pad], dim=1)
            k = pose_actions.shape[1]
            human_pos_table = torch.cat([self.pos_table[:, :1], self.pos_table[:, 2 : 2 + k]], dim=1)
            pos_embed = human_pos_table.clone().detach().permute(1, 0, 2)

        encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_full)
        encoder_output = encoder_output[0]
        latent_info = self.latent_proj(encoder_output)
        mu = latent_info[:, : self.latent_dim]
        logvar = latent_info[:, self.latent_dim :]
        latent_sample = reparametrize(mu, logvar)
        latent_input = self.latent_out_proj(latent_sample)
        return latent_input, mu, logvar

    def forward(
        self,
        *,
        pose_state: torch.Tensor,
        images: torch.Tensor,
        embodiment,
        joint_state: torch.Tensor | None = None,
        camera_mask=None,
        pose_actions: torch.Tensor | None = None,
        joint_actions: torch.Tensor | None = None,
        has_joint_target: bool | None = None,
        is_pad: torch.Tensor | None = None,
    ):
        """
        Returns dict:
          pose_pred [B,K,8], joint_pred [B,K,14] or None for human,
          is_pad_pred [B,K,1], mu, logvar

        pose_state is accepted for interface uniformity with the dataset batch
        schema (and because HumanEpisodeDataset needs to compute it as the
        pose_actions anchor) but is never read here — no embodiment feeds
        pose_state into the model as an observation. See module docstring.
        """
        del pose_state
        is_training = pose_actions is not None or joint_actions is not None
        bs = images.shape[0]

        if isinstance(embodiment, torch.Tensor):
            emb_ids = embodiment.to(dtype=torch.long, device=images.device).reshape(-1)
            if emb_ids.numel() == 1:
                emb_ids = emb_ids.repeat(bs)
            if not torch.all(emb_ids == emb_ids[0]):
                raise ValueError("MixedDETRVAE expects a homogeneous embodiment per batch")
            embodiment_id = int(emb_ids[0].item())
        else:
            embodiment_id = int(embodiment)

        if images.shape[1] != NUM_CAMERAS:
            raise ValueError(f"images must be [B,{NUM_CAMERAS},3,H,W], got {tuple(images.shape)}")

        if joint_state is None:
            joint_state = torch.zeros(bs, ROBOT_JOINT_DIM, device=images.device, dtype=images.dtype)
        if has_joint_target is None:
            has_joint_target = embodiment_id in (EMBODIMENT_ROBOT, EMBODIMENT_MIXED)

        if is_training:
            if is_pad is None:
                raise ValueError("Training requires is_pad")
            if pose_actions is None:
                pose_actions = torch.zeros(bs, self.num_queries, POSE_DIM, device=images.device, dtype=images.dtype)
            if joint_actions is None:
                joint_actions = torch.zeros(
                    bs, self.num_queries, ROBOT_JOINT_DIM, device=images.device, dtype=images.dtype
                )
            latent_input, mu, logvar = self._encode_latent(
                pose_actions=pose_actions,
                joint_state=joint_state,
                joint_actions=joint_actions,
                is_pad=is_pad,
                embodiment=embodiment_id,
            )
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=images.device)
            latent_input = self.latent_out_proj(latent_sample)

        if camera_mask is None:
            camera_mask = torch.ones(bs, NUM_CAMERAS, device=images.device, dtype=torch.float32)
        else:
            camera_mask = camera_mask.to(device=images.device, dtype=torch.float32)
            if camera_mask.ndim == 1:
                camera_mask = camera_mask.unsqueeze(0).expand(bs, -1)
            if camera_mask.shape != (bs, NUM_CAMERAS):
                raise ValueError(f"camera_mask must be [B,{NUM_CAMERAS}], got {tuple(camera_mask.shape)}")

        all_cam_features = []
        all_cam_pos = []
        for cam_id in range(NUM_CAMERAS):
            features, pos = self.backbones[cam_id](images[:, cam_id])
            features = features[0]
            pos = pos[0]
            feat = self.input_proj(features)
            m = camera_mask[:, cam_id].view(bs, 1, 1, 1)
            feat = feat * m
            # Keep pos batchless [1,C,h,w]; scale with homogeneous mask scalar.
            pos = pos * camera_mask[0, cam_id].view(1, 1, 1, 1)
            all_cam_features.append(feat)
            all_cam_pos.append(pos)

        src = torch.cat(all_cam_features, dim=3)
        pos = torch.cat(all_cam_pos, dim=3)

        # Modality routing (no embedding token). Robot/mixed get a real proprio
        # token (joint_state); human has no proprio observation at all, so it
        # contributes only the latent token — one fewer extra token than
        # robot/mixed, not a placeholder standing in for a missing one.
        if embodiment_id in (EMBODIMENT_ROBOT, EMBODIMENT_MIXED):
            if joint_state.shape[-1] != ROBOT_JOINT_DIM:
                raise ValueError(f"joint_state must be [B,{ROBOT_JOINT_DIM}], got {tuple(joint_state.shape)}")
            proprio_input = self.robot_input_proj(joint_state)
            additional_inputs = [latent_input, proprio_input]
            additional_pos_embed = self.additional_pos_embed.weight  # rows [latent, proprio]
        elif embodiment_id == EMBODIMENT_HUMAN:
            additional_inputs = [latent_input]
            additional_pos_embed = self.additional_pos_embed.weight[:1]  # row [latent] only
        else:
            raise ValueError(f"Unknown embodiment id {embodiment_id}")

        hs = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            additional_inputs,
            additional_pos_embed,
        )[0]  # [B, K, H]

        # Shared pose head always; joint head for robot + mixed
        pose_pred = self.pose_action_head(hs)
        joint_pred = (
            self.joint_action_head(hs)
            if embodiment_id in (EMBODIMENT_ROBOT, EMBODIMENT_MIXED)
            else None
        )
        is_pad_pred = self.is_pad_head(hs)
        return {
            "pose_pred": pose_pred,
            "joint_pred": joint_pred,
            "is_pad_pred": is_pad_pred,
            "mu": mu,
            "logvar": logvar,
        }


def build_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        args.hidden_dim,
        args.nheads,
        args.dim_feedforward,
        args.dropout,
        "relu",
        args.pre_norm,
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    return TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)


def build(args):
    validate_camera_names(args.camera_names)
    backbones = [build_backbone(args) for _ in range(NUM_CAMERAS)]
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    model = MixedDETRVAE(
        backbones,
        transformer,
        encoder,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model


def kl_divergence(mu, logvar):
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld
