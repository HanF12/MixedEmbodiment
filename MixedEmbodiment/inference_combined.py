#!/usr/bin/env python
"""
MixedEmbodiment ACT inference (robot / joint pathway only).

I/O closely follows Bimanual-3cam/inference_bimanual.py (RealSense + ROS joints),
but drives MixedEmbodiment.MixedDETRVAE with:
  - embodiment = robot
  - proprio = joint_state [14]  (robot_input_proj)
  - camera slots [bird, left_wrist, right_wrist] (true 3-cam; no front)
  - shared pose head is xyz+gripper (8D); control from joint_action_head only
  - MixedEmbodiment training also has human/mixed modalities; this script uses
    the robot joint path only (human proprio hard-removed in this package; robot path unaffected)

Relative vs absolute
--------------------
Training uses *relative* pose actions for the shared EEF/hand head:
  pose_actions[k] = pose[t+k] - pose[t]
Joint actions remain *absolute*:
  joint_actions[k] = joints[t+k]

This script publishes absolute joint targets from joint_pred, so the relative
pose training convention does **not** change the robot control loop. Human /
EEF relative norms are optional (for logging / future pose control) and unused
when commanding joints.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import torch
from sensor_msgs.msg import JointState

_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PKG_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_PKG_DIR))

from MixedEmbodiment.config import (  # noqa: E402
    CAMERA_ORDER,
    DEFAULT_NUM_QUERIES,
    EMBODIMENT_ROBOT,
    GRIPPER_INDICES,
    LEFT_ARM_SLICE,
    MODEL_CAMERA_NAMES,
    POSE_DIM,
    RIGHT_ARM_SLICE,
    ROBOT_JOINT_DIM,
    camera_mask_tensor,
    concat_bimanual_joints,
    load_run_metadata,
    stack_camera_tensors,
    validate_run_metadata,
)
from MixedEmbodiment.core import build  # noqa: E402
from MixedEmbodiment.joint_lisener import (  # noqa: E402
    get_current_slave_left_positions,
    get_current_slave_right_positions,
    joint_state_listener,
)


RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
RESNET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

DEFAULT_CHECKPOINT = "mixed_act_best.pth"
DEFAULT_ROBOT_NORM = "normalization_stats_robot.npz"
DEFAULT_HUMAN_NORM = "normalization_stats_human.npz"


def resolve_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return path


def serial_matches(requested: str, device_serial: str) -> bool:
    return requested == device_serial or requested in device_serial or device_serial.endswith(requested)


def load_camera_map() -> dict[str, str]:
    try:
        from recording.hand_pose_track import CAMERA_MAP  # type: ignore

        return dict(CAMERA_MAP)
    except Exception:
        return {}


def connected_realsense_devices():
    ctx = rs.context()
    return list(ctx.query_devices())


def pick_realsense_serial(*, role: str | None, serial_arg: str | None) -> str:
    devices = connected_realsense_devices()
    if not devices:
        raise RuntimeError("No RealSense devices found.")

    if serial_arg:
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            if serial_matches(serial_arg, serial):
                return serial
        raise RuntimeError(f"Requested RealSense serial {serial_arg!r} not found.")

    camera_map = load_camera_map()
    if role:
        for mapped_serial, mapped_role in camera_map.items():
            if mapped_role != role:
                continue
            for device in devices:
                serial = device.get_info(rs.camera_info.serial_number)
                if serial_matches(mapped_serial, serial):
                    return serial

    if len(devices) == 1:
        return devices[0].get_info(rs.camera_info.serial_number)
    connected = [device.get_info(rs.camera_info.serial_number) for device in devices]
    raise RuntimeError(f"Multiple RealSense devices detected; please pass a serial or role. Connected: {connected}")


def to_resnet_norm_rgb_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(dtype=torch.float32).div_(255.0)
    return (tensor - RESNET_MEAN) / RESNET_STD


def poll_color_bgr(pipeline: rs.pipeline, *, timeout_ms: int = 120) -> Optional[np.ndarray]:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        frames = pipeline.poll_for_frames()
        if frames:
            color_frame = frames.get_color_frame()
            if color_frame:
                return np.asanyarray(color_frame.get_data())
        time.sleep(0.002)
    return None


def maybe_resize(img_bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def annotate(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img_bgr.copy()
    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22
    return out


def stack_preview_3cam(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """bird | left_wrist on top row; right_wrist alone on bottom (letterboxed left)."""
    top = np.concatenate([a, b], axis=1)
    pad = np.zeros_like(c)
    bottom = np.concatenate([c, pad], axis=1)
    return np.concatenate([top, bottom], axis=0)


def load_joint_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(str(path), allow_pickle=True)
    if "joint_mean" in stats.files and "joint_std" in stats.files:
        mean = np.asarray(stats["joint_mean"], dtype=np.float32)
        std = np.asarray(stats["joint_std"], dtype=np.float32)
    elif "qpos_mean" in stats.files and "qpos_std" in stats.files:
        mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
        std = np.asarray(stats["qpos_std"], dtype=np.float32)
    else:
        raise KeyError(f"{path} missing joint_mean/joint_std or qpos_mean/qpos_std (keys={stats.files}).")
    if mean.shape != (ROBOT_JOINT_DIM,) or std.shape != (ROBOT_JOINT_DIM,):
        raise ValueError(f"Expected joint norm shape ({ROBOT_JOINT_DIM},), got mean={mean.shape} std={std.shape}")
    return mean, std


class ArmPublishers:
    def __init__(self, arm_topic: str, gripper_topic: str, frame_id: str = "world"):
        self.arm_pub = rospy.Publisher(arm_topic, JointState, queue_size=10)
        self.gripper_pub = None
        self.gripper_msg_type = None
        self.frame_id = frame_id
        try:
            from signal_arm.msg import gripper_position_control  # type: ignore

            self.gripper_msg_type = gripper_position_control
            self.gripper_pub = rospy.Publisher(gripper_topic, gripper_position_control, queue_size=10)
        except Exception as exc:
            rospy.logwarn(f"Gripper publisher disabled for {gripper_topic}: {exc}")

    def publish(self, positions: np.ndarray) -> None:
        arm_msg = JointState()
        arm_msg.header.stamp = rospy.Time.now()
        arm_msg.header.frame_id = self.frame_id
        arm_msg.name = [f"joint{i}" for i in range(1, 7)]
        arm_msg.position = positions[:6].tolist()
        self.arm_pub.publish(arm_msg)

        if self.gripper_pub is not None and self.gripper_msg_type is not None and len(positions) >= 7:
            grip_msg = self.gripper_msg_type()
            grip_msg.header.stamp = arm_msg.header.stamp
            grip_msg.header.frame_id = self.frame_id
            grip_msg.gripper_stroke = float(positions[6])
            self.gripper_pub.publish(grip_msg)


class Args:
    def __init__(self, num_queries: int):
        self.num_queries = int(num_queries)
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


parser = argparse.ArgumentParser(description="MixedEmbodiment ACT inference (robot/joint pathway; absolute joint targets)")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
parser.add_argument(
    "--normalization_path", type=str, default=DEFAULT_ROBOT_NORM,
    help="Robot normalization npz (joint_mean/joint_std or qpos_mean/qpos_std)",
)
parser.add_argument(
    "--human_normalization_path", type=str, default=DEFAULT_HUMAN_NORM,
    help="Human relative-pose norms (unused for joint control; validated/logged only)",
)
parser.add_argument("--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--display_scale", type=float, default=0.5)
parser.add_argument("--display_max_fps", type=float, default=15.0)
parser.add_argument("--chunking", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--inference_fps", type=float, default=15.0)
parser.add_argument("--resize_factor", type=float, default=1.0)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--aggregation_horizon", type=int, default=None)
parser.add_argument("--left_joint_topic", type=str, default="/joint_states_slave_left")
parser.add_argument("--right_joint_topic", type=str, default="/joint_states_slave_right")
parser.add_argument("--topic_arm_left", type=str, default="/arm_joint_target_position_slave_left")
parser.add_argument("--topic_gripper_left", type=str, default="/gripper_position_control_slave_left")
parser.add_argument("--topic_arm_right", type=str, default="/arm_joint_target_position_slave_right")
parser.add_argument("--topic_gripper_right", type=str, default="/gripper_position_control_slave_right")
parser.add_argument(
    "--gripper_mode", choices=("binary", "continuous"), default="binary",
    help="binary: threshold denorm preds to 0/70; continuous: scale*pred then clamp to --gripper_max",
)
parser.add_argument("--gripper_threshold_left", type=float, default=0.5, help="[binary] Left gripper: denormalized pred < threshold -> 0, else 70")
parser.add_argument("--gripper_threshold_right", type=float, default=0.5, help="[binary] Right gripper: denormalized pred < threshold -> 0, else 70")
parser.add_argument("--gripper_scale_left", type=float, default=65.0, help="[continuous] Multiply left denormalized gripper pred by this before publishing")
parser.add_argument("--gripper_scale_right", type=float, default=65.0, help="[continuous] Multiply right denormalized gripper pred by this before publishing")
parser.add_argument("--gripper_max", type=float, default=80.0, help="[continuous] Clamp scaled gripper cmd to this (set <0 to disable)")
parser.add_argument("--max_joint_speed", type=float, default=0.45)
parser.add_argument("--max_gripper_speed", type=float, default=200)
parser.add_argument("--bird_role", choices=("left", "right", "center", "front"), default="center")
parser.add_argument("--bird_serial", type=str, default=None)
parser.add_argument("--bird_color_fps", type=int, default=15)
parser.add_argument("--left_wrist_role", choices=("left", "right", "center", "front"), default="left")
parser.add_argument("--left_wrist_serial", type=str, default=None)
parser.add_argument("--left_wrist_color_fps", type=int, default=15)
parser.add_argument("--right_wrist_role", choices=("left", "right", "center", "front"), default="right")
parser.add_argument("--right_wrist_serial", type=str, default=None)
parser.add_argument("--right_wrist_color_fps", type=int, default=15)
cli = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device} cuda_available={torch.cuda.is_available()}")
print(f"safety: max_joint_speed={cli.max_joint_speed:g} rad/s, max_gripper_speed={cli.max_gripper_speed:g}, gripper_mode={cli.gripper_mode}")
print(f"cameras={list(CAMERA_ORDER)} (true 3-cam, pose_dim={POSE_DIM})")

checkpoint_path = resolve_path(cli.checkpoint)
metadata = load_run_metadata(checkpoint_path.parent)
if metadata is not None:
    try:
        validate_run_metadata(metadata, num_queries=cli.num_queries)
        print(f"run_metadata: pose_action_space={metadata.get('pose_action_space')} (joint control uses absolute joint_pred)")
    except ValueError as exc:
        print(f"Warning: skipping run_metadata validation: {exc}")

model = build(Args(cli.num_queries)).to(device)
state_dict = torch.load(str(checkpoint_path), map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded checkpoint: {checkpoint_path}")

robot_norm_path = resolve_path(cli.normalization_path)
joint_mean_np, joint_std_np = load_joint_norm_stats(robot_norm_path)
qpos_mean = torch.from_numpy(joint_mean_np.reshape(1, ROBOT_JOINT_DIM)).to(device)
qpos_std = torch.from_numpy(joint_std_np.reshape(1, ROBOT_JOINT_DIM)).to(device)
print(f"Robot joint norms: {robot_norm_path}")

human_norm_path = resolve_path(cli.human_normalization_path)
if human_norm_path.is_file():
    human_stats = np.load(str(human_norm_path), allow_pickle=True)
    space = str(human_stats["pose_action_space"]) if "pose_action_space" in human_stats.files else "unknown"
    print(f"Human pose norms (unused for joint control): {human_norm_path} keys={human_stats.files} pose_action_space={space}")
else:
    print(f"Warning: human normalization file not found: {human_norm_path}")

robot_cam_mask = camera_mask_tensor(EMBODIMENT_ROBOT).unsqueeze(0).to(device)  # [1,3]
dummy_pose_state = torch.zeros(1, POSE_DIM, dtype=torch.float32, device=device)

rospy.init_node("mixed_embodiment_act_inference_robot", anonymous=True)
joint_state_listener(topic=str(cli.left_joint_topic), side="left")
joint_state_listener(topic=str(cli.right_joint_topic), side="right")
left_publishers = ArmPublishers(str(cli.topic_arm_left), str(cli.topic_gripper_left))
right_publishers = ArmPublishers(str(cli.topic_arm_right), str(cli.topic_gripper_right))

color_width = int(cli.width)
color_height = int(cli.height)
camera_specs = [
    (CAMERA_ORDER[0], str(cli.bird_role), cli.bird_serial, int(cli.bird_color_fps)),
    (CAMERA_ORDER[1], str(cli.left_wrist_role), cli.left_wrist_serial, int(cli.left_wrist_color_fps)),
    (CAMERA_ORDER[2], str(cli.right_wrist_role), cli.right_wrist_serial, int(cli.right_wrist_color_fps)),
]

pipelines: list[tuple[str, str, rs.pipeline]] = []
for label, role, serial_arg, fps in camera_specs:
    serial = pick_realsense_serial(role=role, serial_arg=serial_arg)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
    pipeline.start(config)
    pipelines.append((label, serial, pipeline))
    cam_id = list(CAMERA_ORDER).index(label)
    print(f"Started {label} -> model cam{cam_id}/backbone{cam_id} on serial {serial} (role={role})")

prediction_horizon = int(cli.num_queries)
aggregation_horizon = int(cli.aggregation_horizon) if cli.aggregation_horizon is not None else prediction_horizon
decay = 0.075
past_predictions_buffer = collections.deque(maxlen=aggregation_horizon)
loop_rate = rospy.Rate(float(cli.inference_fps))
last_cmd: Optional[np.ndarray] = None
last_cmd_t: Optional[float] = None
last_preview_t = 0.0

try:
    while not rospy.is_shutdown():
        frames = [poll_color_bgr(pipeline, timeout_ms=90) for _, _, pipeline in pipelines]
        left_state = get_current_slave_left_positions()
        right_state = get_current_slave_right_positions()
        if any(frame is None for frame in frames) or left_state is None or right_state is None:
            loop_rate.sleep()
            continue

        qpos_np = concat_bimanual_joints(
            np.asarray(list(left_state[0]), dtype=np.float32),
            np.asarray(list(right_state[0]), dtype=np.float32),
            rec_id="live_inference",
        ).cpu().numpy()
        joint_state = torch.from_numpy(qpos_np).unsqueeze(0).to(device)
        joint_state = (joint_state - qpos_mean) / qpos_std

        if float(cli.resize_factor) != 1.0:
            frames = [maybe_resize(frame, float(cli.resize_factor)) for frame in frames]

        # Stack by label (not list index) so order always matches training:
        # CAMERA_ORDER = (bird, left_wrist, right_wrist) -> cam0/cam1/cam2.
        frame_by_label = {label: frame for (label, _, _), frame in zip(pipelines, frames)}
        stacked_images = stack_camera_tensors(
            to_resnet_norm_rgb_tensor(frame_by_label["bird"]),
            to_resnet_norm_rgb_tensor(frame_by_label["left_wrist"]),
            to_resnet_norm_rgb_tensor(frame_by_label["right_wrist"]),
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(
                pose_state=dummy_pose_state,
                images=stacked_images,
                embodiment=EMBODIMENT_ROBOT,
                joint_state=joint_state,
                camera_mask=robot_cam_mask,
            )
            pred = out["joint_pred"]
            if pred is None:
                raise RuntimeError("joint_pred is None for robot embodiment")

        predicted_trajectory = pred[0] * qpos_std.squeeze(0) + qpos_mean.squeeze(0)
        predicted_trajectory_np = predicted_trajectory.cpu().numpy()
        past_predictions_buffer.append(predicted_trajectory_np)

        positions_to_publish = np.zeros((ROBOT_JOINT_DIM,), dtype=np.float32)
        if bool(cli.chunking):
            wsum = 0.0
            for i in range(len(past_predictions_buffer)):
                buffered_pred = past_predictions_buffer[-(i + 1)]
                if i < buffered_pred.shape[0]:
                    weight = float(np.exp(-decay * i))
                    positions_to_publish += buffered_pred[i] * weight
                    wsum += weight
            if wsum > 0:
                positions_to_publish /= wsum
            else:
                positions_to_publish = predicted_trajectory_np[0].astype(np.float32)
        else:
            positions_to_publish = predicted_trajectory_np[0].astype(np.float32)

        raw_grip_l = float(positions_to_publish[GRIPPER_INDICES[0]])
        raw_grip_r = float(positions_to_publish[GRIPPER_INDICES[1]])
        thr_l = float(cli.gripper_threshold_left)
        thr_r = float(cli.gripper_threshold_right)
        if cli.gripper_mode == "binary":
            cmd_l = 70.0 if raw_grip_l >= thr_l else 0.0
            cmd_r = 70.0 if raw_grip_r >= thr_r else 0.0
        else:
            cmd_l = raw_grip_l * float(cli.gripper_scale_left)
            cmd_r = raw_grip_r * float(cli.gripper_scale_right)
            if float(cli.gripper_max) >= 0:
                cmd_l = min(cmd_l, float(cli.gripper_max))
                cmd_r = min(cmd_r, float(cli.gripper_max))
        positions_to_publish[GRIPPER_INDICES[0]] = cmd_l
        positions_to_publish[GRIPPER_INDICES[1]] = cmd_r

        desired = positions_to_publish.astype(np.float32)
        now_t = time.monotonic()
        if last_cmd is None or last_cmd_t is None:
            last_cmd = qpos_np.astype(np.float32).copy()
            if cli.gripper_mode == "binary":
                last_cmd[GRIPPER_INDICES[0]] = 70.0 if float(last_cmd[GRIPPER_INDICES[0]]) >= thr_l else 0.0
                last_cmd[GRIPPER_INDICES[1]] = 70.0 if float(last_cmd[GRIPPER_INDICES[1]]) >= thr_r else 0.0
            else:
                last_cmd[GRIPPER_INDICES[0]] *= float(cli.gripper_scale_left)
                last_cmd[GRIPPER_INDICES[1]] *= float(cli.gripper_scale_right)
                if float(cli.gripper_max) >= 0:
                    last_cmd[GRIPPER_INDICES[0]] = min(float(last_cmd[GRIPPER_INDICES[0]]), float(cli.gripper_max))
                    last_cmd[GRIPPER_INDICES[1]] = min(float(last_cmd[GRIPPER_INDICES[1]]), float(cli.gripper_max))
            last_cmd_t = now_t
        else:
            dt_nom = 1.0 / max(1e-3, float(cli.inference_fps))
            dt = min(max(1e-3, float(now_t - last_cmd_t)), dt_nom)
            max_dq = float(cli.max_joint_speed) * dt
            max_dg = float(cli.max_gripper_speed) * dt
            cmd = last_cmd.copy()

            for offset in (0, 7):
                dq = desired[offset : offset + 6] - cmd[offset : offset + 6]
                dq = np.clip(dq, -max_dq, max_dq)
                cmd[offset : offset + 6] += dq
                dg = float(np.clip(desired[offset + 6] - cmd[offset + 6], -max_dg, max_dg))
                cmd[offset + 6] += dg

            last_cmd = cmd.astype(np.float32)
            last_cmd_t = now_t

        left_publishers.publish(last_cmd[LEFT_ARM_SLICE])
        right_publishers.publish(last_cmd[RIGHT_ARM_SLICE])

        if bool(cli.display):
            wall = time.time()
            min_dt = 1.0 / max(1e-3, float(cli.display_max_fps))
            if wall - last_preview_t >= min_dt:
                last_preview_t = wall
                if cli.gripper_mode == "binary":
                    mode_note = f"binary thr L/R={thr_l:g}/{thr_r:g}"
                else:
                    mode_note = f"continuous scale L/R={cli.gripper_scale_left:g}/{cli.gripper_scale_right:g} max={cli.gripper_max:g}"
                grip_overlay = (
                    f"raw L/R={raw_grip_l:.3f}/{raw_grip_r:.3f} "
                    f"-> cmd {desired[GRIPPER_INDICES[0]]:.1f}/{desired[GRIPPER_INDICES[1]]:.1f} ({mode_note})"
                )
                print(f"gripper {grip_overlay}", flush=True)
                preview_order = ("bird", "left_wrist", "right_wrist")
                serial_by_label = {label: serial for label, serial, _ in pipelines}
                shown = []
                for cam_id, label in enumerate(preview_order):
                    shown.append(annotate(frame_by_label[label], [f"cam{cam_id}={label} serial={serial_by_label[label]}", grip_overlay]))
                shown = [maybe_resize(frame, float(cli.display_scale)) for frame in shown]
                h = min(frame.shape[0] for frame in shown)
                w = min(frame.shape[1] for frame in shown)
                preview = stack_preview_3cam(shown[0][:h, :w], shown[1][:h, :w], shown[2][:h, :w])
                cv2.imshow("MixedEmbodiment ACT Inference (robot)", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        loop_rate.sleep()
except KeyboardInterrupt:
    print("Inference stopped by user.")
finally:
    for _, _, pipeline in pipelines:
        try:
            pipeline.stop()
        except Exception:
            pass
    cv2.destroyAllWindows()
