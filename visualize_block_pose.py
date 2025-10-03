#!/usr/bin/env python3

"""
CS 6341 Homework 2 Programming
Transformation
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
import tf2_ros
import message_filters

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat


# convert quaternion
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


MODEL = 'cube'
ROBOT = 'fetch'

class PoseToTF(Node):
    def __init__(self):
        super().__init__('pose_to_tf')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        
        sub_a = message_filters.Subscriber(self, Pose, f'/model/{MODEL}/pose')
        sub_b = message_filters.Subscriber(self, Pose, f'/model/{ROBOT}/pose')

        # Approximate sync by arrival time (Pose has no header)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_a, sub_b], queue_size=10, slop=0.1, allow_headerless=True
        )
        self.sync.registerCallback(self.cb)        


    # callback function to handle the cube pose and robot pose
    def cb(self, pose_model: Pose, pose_robot: Pose):

        # convert the cube pose in world frame T_wo
        T_wo = ros_pose_to_rt(pose_model)
        print('T_wo', T_wo)

        # convert the robot pose in world frame T_wb
        T_wb = ros_pose_to_rt(pose_robot)
        print('T_wb', T_wb)
    
        ################ TO DO ##########################
        # compute the object pose in robot base link T_bo: 4x4 transformation matrix

        ################ TO DO ########################## 
        
        # get the position and quaternion from T_bo
        position = T_bo[:3, 3]
        quaternion = mat2quat(T_bo[:3, :3])
        q = ros_quat(quaternion)

        # publish the tf
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = MODEL
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.br.sendTransform(t)
        print('sending pose in base link: ' + MODEL)


def main():
    rclpy.init()
    node = PoseToTF()
    # Spin continuously; otherwise you’ll process one callback and quit
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()