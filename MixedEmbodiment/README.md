# MixedEmbodiment ACT training

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


**Robot only** (nearest equivalent to `Bimanual-3cam`, but note this still runs the full
shared `MixedDETRVAE` architecture, not `Bimanual-3cam`'s simpler model — it is not a
numerically-verified stand-in the way the other two are):
```bash
ALOHA-mimic/.venv/bin/python -m MixedEmbodiment.training_combined \
  --sessions_root sessions --embodiments robot
```

**Robot + human**:
```bash
ALOHA-mimic/.venv/bin/python -m MixedEmbodiment.training_combined \
  --sessions_root sessions --embodiments robot,human
```

**Robot + human + mixed**:
```bash
ALOHA-mimic/.venv/bin/python -m MixedEmbodiment.training_combined \
  --sessions_root sessions --embodiments robot,human,mixed \
  --wandb --wandb_project mixed-embodiment-3cam-act --run_name my_run
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
| `-b`, `--batch` | `16` | |
| `-q`, `--num_queries` | `45` | Action-chunk horizon K. |
| `-g`, `--gpu_number` | `0` | Ignored if `--cpu`. |
| `--cpu` | off | Force CPU even if a GPU is available. |
| `--lr` | `2e-5` | Linearly scaled with the default batch size (was `1e-5` @ batch 8). |
| `--weight_decay` | `1e-4` | AdamW. |
| `--num_workers` | `2` | DataLoader workers, per active modality loader. **Careful raising this without `--jpeg_in_ram`**: each worker is forked from a process holding the full raw-frame dataset in RAM as plain Python/numpy objects, and copy-on-write duplication across workers is easy to trigger — this is a shared machine. Safe to raise once `--jpeg_in_ram` is on (much smaller per-worker footprint). |
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
| `--mixed_lambda` | `1.0` | Scales the whole mixed loss (pose + joint recon + KL). |
| `--reconstruction_loss` | `l1` | `l1` or `mse`. |
| `--gripper_binarize_threshold` | `0.5` | Binarizes robot/mixed gripper channels (both EEF/hand-pose NPZ and joint NPY) at data-load time. Human hand poses are never re-binarized. |
| `--joint_modality_update` / `--no-joint_modality_update` | on | Average all active-modality losses into one optimizer step per training step (default) vs. alternating single-modality steps. |
| `--pose_observation` / `--no-pose_observation` | **off** | Off (default): the human embodiment's proprio/CVAE-state adapters aren't even constructed — a learned constant stands in, so the model must predict the relative pose chunk from video alone. On: restores the legacy behavior (real `Linear` adapter fed the true absolute hand pose). Robot/mixed are unaffected either way — their proprio is always `joint_state`. |

### Logging / misc
| Flag | Default | Notes |
|---|---|---|
| `--output_dir` | `MixedEmbodiment/weights` | |
| `--run_name` | timestamp | |
| `--wandb` | off | |
| `--wandb_project` | `mixed-embodiment-3cam-act` | |
| `--wandb_entity` / `--wandb_run_name` / `--wandb_mode` | `None` / `None` / `online` | |
| `--dry_run` | off | Sync, load one batch per active modality, run one train step each, then exit — no checkpoint saved. |

## Inference

`inference_combined.py` is a ROS + RealSense control loop for the **robot** embodiment
only (it drives `joint_pred`, absolute joint targets). It needs `rospy` and
`pyrealsense2`, which aren't part of the training environment — run it on the robot
control machine, not the training machine. Not affected by `--pose_observation` (that
flag only changes the human pathway, which this script never touches).
