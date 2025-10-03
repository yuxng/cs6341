#!/usr/bin/env python

"""
CS 6301 Homework 3 Programming
Planning scene and IK
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future
from geometry_msgs.msg import Pose

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
    
    # compute the object pose in robot base link T_bo
    T_bo = np.matmul(np.linalg.inv(T_wb), T_wo)
    return T_bo

    
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
    print(T_bo)

    # add a collision box in moveit scene for the cube
    object_id = 'cube'
    position = T_bo[:3, 3]
    dimensions = [0.06, 0.06, 0.06]
    quat_xyzw = ros_quat(mat2quat(T_bo[:3, :3]))
    moveit2.add_collision_box(
        id=object_id, position=position, quat_xyzw=quat_xyzw, size=dimensions
    )