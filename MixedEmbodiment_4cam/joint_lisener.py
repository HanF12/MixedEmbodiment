#!/usr/bin/env python
# This line tells the OS to execute the script using the python interpreter
"""Vendored from ALOHA-mimic/joint_lisener.py; unchanged."""

import rospy
# Import the JointState message type
from sensor_msgs.msg import JointState
import numpy as np

sampling_interval = 30 # Currently at 30Hz

# Last received JointState per side.
last_left_joint_state_msg = None
last_right_joint_state_msg = None

def slave_left_joint_states_callback(msg):
    """
    Callback function for receiving robot arm joint state feedback for slave_left.
    Samples the data and stores it in an HDF5 file.
    """
    global last_left_joint_state_msg
    last_left_joint_state_msg = msg # Store the entire message

    current_time = msg.header.stamp.to_sec() # Get the timestamp of the message in seconds
    # You can process msg.position and msg.velocity here if needed
    # msg.position
    # msg.velocity


def slave_right_joint_states_callback(msg):
    """
    Callback function for receiving robot arm joint state feedback for slave_right.
    """
    global last_right_joint_state_msg
    last_right_joint_state_msg = msg


def get_current_slave_left_positions():
    """
    Returns the joint positions from the last received JointState message for slave_left.
    Returns None if no message has been received yet.
    """
    global last_left_joint_state_msg
    if last_left_joint_state_msg is not None:
        return (last_left_joint_state_msg.position, last_left_joint_state_msg.velocity)
    else:
        rospy.logwarn("No slave_left joint state message received yet.")
        return None


def get_current_slave_right_positions():
    """
    Returns the joint positions from the last received JointState message for slave_right.
    Returns None if no message has been received yet.
    """
    global last_right_joint_state_msg
    if last_right_joint_state_msg is not None:
        return (last_right_joint_state_msg.position, last_right_joint_state_msg.velocity)
    else:
        rospy.logwarn("No slave_right joint state message received yet.")
        return None


def joint_state_listener(topic: str = "/joint_states_slave_left", side: str = "left"):
    """
    Register the subscriber used by inference.

    Note: this does NOT call `rospy.init_node()` and does NOT block/spin.
    """
    cb = slave_left_joint_states_callback if side == "left" else slave_right_joint_states_callback
    rospy.Subscriber(topic, JointState, cb)
    rospy.loginfo(f"Subscriber created for topic: {topic} (side={side})")
