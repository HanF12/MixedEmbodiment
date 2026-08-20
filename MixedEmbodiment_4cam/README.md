# MixedEmbodiment ACT training

One CLI, three CLI-selectable modalities (robot / human / mixed hand+arm), sharing a
single ACT/DETR-VAE model. See `config.py` / `core.py` docstrings for the architecture;
this file is just "how do I run it."

## Can I copy just `sessions/` + `MixedEmbodiment/` into a new repo?

**Yes.** `MixedEmbodiment/` is fully self-contained — it used to import the shared
backbone/transformer code from a sibling `ALOHA-mimic/` folder, but that code
(`model.py`, `position_encoding.py` — the ResNet backbone, position encodings, and DETR
transformer) and the ROS joint listener (`joint_lisener.py`, used only by
`inference_combined.py`) are now vendored directly inside this folder. Verified by
actually renaming `ALOHA-mimic/` out of the way and running a full robot+human+mixed
training step — no code in `MixedEmbodiment/` reaches outside it anymore. So:

```
<new_repo_root>/
  MixedEmbodiment/     <- this folder, nothing else required
  sessions/            <- your data
```

You still need a Python environment with torch/torchvision/opencv/pandas/numpy/tqdm
(`wandb` only if you pass `--wandb`) — that's a *runtime* dependency, not a *code*
dependency, so any environment with those packages works; it doesn't need to be the
`ALOHA-mimic/.venv` specifically (that's just the one already set up on this machine,
which is why the examples below invoke it directly). To build a fresh one elsewhere:

```bash
uv venv && source .venv/bin/activate
uv pip install --index-url "https://download.pytorch.org/whl/cu124" torch torchvision  # or the CPU wheel
uv pip install opencv-python-headless pandas numpy tqdm wandb
```

One more gotcha: `--sessions_root` defaults to the **relative** path `sessions`, resolved
against your current working directory, not the script's location. Either run commands
from `<new_repo_root>`, or pass `--sessions_root /absolute/path/to/sessions`.

`MixedEmbodiment/compare_legacy.py` is the exception — it intentionally imports the old
`Combined_relative_3cam_gripweight` / `MixedEmbodiment_gripweight` packages for
regression testing and will not run outside this repo. It's not needed for training;
leave it behind (or delete it) when copying to a new repo.

## Expected `sessions/` layout

```
sessions/
  teleop_bimanual/<date>/                       # robot
  human_hands_bimanual_raw/<date>/              # human
  left_robot_right_hand/<date>/                 # mixed (left arm + right hand)
  right_robot_left_hand/<date>/                 # mixed (right arm + left hand)
```

`--sessions_root` discovers the **latest** date folder under `teleop_bimanual/` and
`human_hands_bimanual_raw/`, and **all** date folders under both mixed kinds (pooled
into one modality). This is the only supported input shape — there's no flag to point
at an arbitrary custom path per modality.

## The three trainings

`--embodiments` selects exactly one of the three regimes below — these are the **only**
three accepted values; anything else (`human` alone, `mixed` alone, `human,mixed`
without robot, etc.) is rejected at startup. robot is always the anchor, and mixed
requires human, matching how the three predecessor folders actually worked (robot alone
used a different, simpler model in `Bimanual-3cam`;
`Combined_relative_3cam_gripweight` always required robot+human together;
`MixedEmbodiment_gripweight` always required robot+human, with mixed as the optional
addition on top). Run from `<repo_root>` with any Python env that has the packages
listed above (the examples below use `ALOHA-mimic/.venv` since it's already set up on
this machine — swap in your own venv's python elsewhere).

**Robot only** (nearest equivalent to `Bimanual-3cam`, but note this still runs the full
shared `MixedDETRVAE` architecture, not `Bimanual-3cam`'s simpler model — it is not a
numerically-verified stand-in the way the other two are):
```bash
ALOHA-mimic/.venv/bin/python -m MixedEmbodiment.training_combined \
  --sessions_root sessions --embodiments robot
```

**Robot + human** (matches `Combined_relative_3cam_gripweight` — numerically verified
identical via `compare_legacy.py`):
```bash
ALOHA-mimic/.venv/bin/python -m MixedEmbodiment.training_combined \
  --sessions_root sessions --embodiments robot,human
```

**Robot + human + mixed** (matches `MixedEmbodiment_gripweight` — numerically verified
identical via `compare_legacy.py`; this is the default for `--embodiments` and the main
EgoMimic-style co-training run):
```bash
ALOHA-mimic/.venv/bin/python -m MixedEmbodiment.training_combined \
  --sessions_root sessions --embodiments robot,human,mixed \
  --wandb --wandb_project mixed-embodiment-3cam-act --run_name my_run
```

**Fast sanity check before a real run** (tiny frames, short chunk, one batch/modality,
exits after one step — use this to catch data/path problems in seconds instead of
minutes):
```bash
ALOHA-mimic/.venv/bin/python -m MixedEmbodiment.training_combined \
  --sessions_root sessions --embodiments robot,human,mixed \
  --robot_max_demos 1 --human_max_demos 1 --mixed_max_demos 1 \
  --resize_factor 0.25 --num_queries 8 --batch 1 --num_workers 0 --cpu \
  --max_sync_rows 16 --dry_run
```

Checkpoints and `run_metadata.json` land in `MixedEmbodiment/weights/<run_name>/`
(`--run_name` defaults to a timestamp); sync CSVs land in
`MixedEmbodiment/m-synced-csvs/<run_name>/` and are safe to delete after a run.

## CLI flags

### Data / modality selection
| Flag | Default | Notes |
|---|---|---|
| `--sessions_root` | `sessions` | Only supported data-input shape; see layout above. |
| `--embodiments` | `robot,human,mixed` | Exactly one of `robot` / `robot,human` / `robot,human,mixed` — see "The three trainings" above. Any other combination is rejected at startup. A requested modality that's genuinely missing on disk is skipped with a warning; fails only if none end up active. |
| `--robot_max_demos` | `None` (all) | First N robot demos, sorted by demo ID. |
| `--human_max_demos` | `None` (all) | First N human demos. |
| `--mixed_max_demos` | `None` (all) | First N mixed demos **per mixed session root** (i.e. applies separately to `left_robot_right_hand` and `right_robot_left_hand`). |

### Model / schedule
| Flag | Default | Notes |
|---|---|---|
| `-e`, `--epochs` | `10000` | One epoch = one full pass over the longest active-modality loader; shorter ones are recycled. |
| `-b`, `--batch` | `8` | |
| `-q`, `--num_queries` | `45` | Action-chunk horizon K. |
| `-g`, `--gpu_number` | `0` | Ignored if `--cpu`. |
| `--cpu` | off | Force CPU even if a GPU is available. |
| `--lr` | `1e-5` | |
| `--weight_decay` | `1e-4` | AdamW. |
| `--num_workers` | `2` | DataLoader workers. |
| `--resize_factor` | `1.0` | Scale frames before encoding. |
| `--jpeg_in_ram` | off | Store synced frames as JPEG bytes in RAM instead of raw arrays (much less host memory). |
| `--jpeg_quality` | `90` | Only relevant with `--jpeg_in_ram`. |
| `--max_sync_rows` | `None` (no cap) | Cap synced rows per demo — debug/smoke use. |
| `--max_skew_s` | `0.050` | Sync tolerance in seconds across camera/joint/pose streams. |
| `--save_every_epochs` | `1000` | Also always saves `mixed_act_best.pth` on every new best avg loss. |

### Loss
| Flag | Default | Notes |
|---|---|---|
| `--pose_loss_weight` | `1.0` | Weight on the shared pose-head recon loss. |
| `--joint_loss_weight` | `1.0` | Weight on the robot/mixed joint-head recon loss. |
| `--gripper_loss_weight` | `5.0` | **Gripweight.** Multiplies per-element recon error on gripper dims (pose indices 3,7; joint indices 6,13). Use `1.0` for an unweighted baseline. |
| `--kl_weight` | `10.0` | CVAE KL term weight. |
| `--hand_lambda` | `1.0` | Scales the whole human loss (pose recon + KL). |
| `--mixed_lambda` | auto | Default: `len(mixed_loader)/steps_per_epoch` (equiv. to seeing each mixed batch once). Pass a float to override. |
| `--reconstruction_loss` | `l1` | `l1` or `mse`. |
| `--gripper_binarize_threshold` | `0.5` | Binarizes robot/mixed gripper channels (both EEF/hand-pose NPZ and joint NPY) at data-load time. Human hand poses are never re-binarized. |
| `--joint_modality_update` / `--no-joint_modality_update` | on | Average all active-modality losses into one optimizer step per training step (default) vs. alternating single-modality steps. |
| human proprio | **hard-removed** | No adapter and no placeholder token; human predicts relative pose from video + CVAE latent only. Robot/mixed proprio is always `joint_state`. |

### Logging / misc
| Flag | Default | Notes |
|---|---|---|
| `--output_dir` | `/data/hfang09/MixedEmbodiment/weights` | Explicit root overrides the default. |
| `--weights_on_home` | off | Save under `MixedEmbodiment/weights_home/` on `/home` (real dir, not the `weights/` → `/data` symlink). |
| `--run_name` | timestamp | |
| `--wandb` | off | |
| `--wandb_project` | `mixed-embodiment-3cam-act` | |
| `--wandb_entity` / `--wandb_run_name` / `--wandb_mode` | `None` / `None` / `online` | |
| `--dry_run` | off | Sync, load one batch per active modality, run one train step each, then exit — no checkpoint saved. |

## Inference

`inference_combined.py` is a ROS + RealSense control loop for the **robot** embodiment
only (it drives `joint_pred`, absolute joint targets). It needs `rospy` and
`pyrealsense2`, which aren't part of the training environment — run it on the robot
control machine, not the training machine. Unaffected by human proprio hard-removal
(that only changes the human pathway, which this script never touches).
