#!/usr/bin/env python

"""
CS 6341 Homework 3 Programming
Planning scene, FK and IK
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from pymoveit2 import MoveIt2
from pymoveit2.robots import fetch as robot

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat


def ros_quat(tf_quat): #wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


# Convert quaternion and translation to a 4x4 tranformation matrix
# See Appendix B.3 in Lynch and Park, Modern Robotics for the definition of quaternion
def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


# Convert a ROS pose message to a 4x4 tranformation matrix
def ros_pose_to_rt(pose):
    qarray = [0, 0, 0, 0]
    qarray[0] = pose.orientation.x
    qarray[1] = pose.orientation.y
    qarray[2] = pose.orientation.z
    qarray[3] = pose.orientation.w

    t = [0, 0, 0]
    t[0] = pose.position.x
    t[1] = pose.position.y
    t[2] = pose.position.z

    return ros_qt_to_rt(qarray, t)


def get_message_once(node, topic_name, msg_type, timeout=5.0):
    future = Future()

    def callback(msg):
        if not future.done():
            future.set_result(msg)
            node.destroy_subscription(sub)

    sub = node.create_subscription(msg_type, topic_name, callback, 10)

    # Spin this node until a message is received or timeout
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)

    return future.result() if future.done() else None


MODEL = 'cube'
ROBOT = 'fetch'

# Query pose of frames from the Gazebo environment
def get_pose_gazebo(node):

    pose_model = get_message_once(node, f'/model/{MODEL}/pose', Pose, timeout=2.0)
    # convert the cube pose in world frame T_wo
    T_wo = ros_pose_to_rt(pose_model)
    print('T_wo', T_wo)

    pose_robot = get_message_once(node, f'/model/{ROBOT}/pose', Pose, timeout=2.0)
    # convert the robot pose in world frame T_wb
    T_wb = ros_pose_to_rt(pose_robot)
    print('T_wb', T_wb)

    ################ TO DO ##########################
    # compute the object pose in robot base link T_bo: 4x4 transformation matrix

    ################ TO DO ##########################
    return T_bo


def get_current_joint_states(node):
    """Return one JointState message from /joint_states, or None if timeout."""
    msg = get_message_once(node, '/joint_states', JointState, timeout=2.0)
    print(msg)

    # get the joint names in the Fetch robot group
    names = robot.joint_names()
    joint_positions = extract_specific_joints(msg, names)
    return joint_positions


# extract the joint positions from the msg
def extract_specific_joints(msg, names):
    joint_positions = np.zeros((len(names), ), np.float32)
    for (i, name) in enumerate(names):
        joint_positions[i] = msg.position[msg.name.index(name)]
        print(name, joint_positions[i])
    return joint_positions    

    
if __name__ == "__main__":
    """
    Main function to run the code
    """

    rclpy.init()

    # Create node for this example
    node = Node("cs6341_homework3")    

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    # Create MoveIt 2 interface
    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=robot.base_link_name(),
        end_effector_name=robot.end_effector_name(),
        group_name=robot.MOVE_GROUP_ARM,
        callback_group=callback_group,
    )    
    
    # query the pose of the cube
    T_bo = get_pose_gazebo(node)
    print('T_bo', T_bo)

    ################ TO DO ##########################
    # add a collision box in moveit scene for the cube
    # follow this example: https://github.com/IRVLUTD/pymoveit2/blob/main/examples/ex_collision_primitive.py
    # use the function moveit2.add_collision_box() to add a collision box for the cube
    object_id = 'cube'
    dimensions = [0.06, 0.06, 0.06]

    ################ TO DO ##########################


    # get the current robot joints
    joint_positions = get_current_joint_states(node)
    print(joint_positions)

    ################ TO DO ##########################
    # forward kinematics
    # follow this example: https://github.com/IRVLUTD/pymoveit2/blob/main/examples/ex_fk.py
    # use the function moveit2.compute_fk() for FK, return the results to retval
    retval = None

    ################ TO DO ##########################
    if retval is None:
        print("Forward kinematics failed.")
    else:
        print("Forward kinematics succeeded. Result: " + str(retval))
        print("---------------------------")
        # extract the pose for the result
        p = retval.pose.position
        q = retval.pose.orientation
        print(f"{robot.end_effector_name()} in {retval.header.frame_id}: "
              f"pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}), "
              f"quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})")             

        print("---------------------------")

        ################ TO DO ##########################
        # inverse kinematics using the above end-effector pose (p, q)
        # follow this example: https://github.com/IRVLUTD/pymoveit2/blob/main/examples/ex_ik.py
        # use the function moveit2.compute_ik() for IK, return the results to retval_ik
        retval_ik = None

        ################ TO DO ##########################
        if retval_ik is None:
            print("Inverse kinematics failed.")
        else:
            print("Inverse kinematics succeeded. Result: " + str(retval_ik))

            # extract the arm joints
            print("---------------------------")
            names = robot.joint_names()
            joint_positions_ik = extract_specific_joints(retval_ik, names)

            # compute the distances between joints
            distance = np.linalg.norm(joint_positions - joint_positions_ik)
            print('joint distance:', distance)