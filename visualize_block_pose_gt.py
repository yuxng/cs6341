#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
import tf2_ros
import time

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


MODEL = 'cube'
WORLD = 'world'

class PoseToTF(Node):
    def __init__(self):
        super().__init__('pose_to_tf')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.create_subscription(Pose, f'/model/{MODEL}/pose', self.cb, 10)


    def wait_for_tf(self, target='world', source='cube', timeout=1.0):
        """
        Block until the transform (source->target) is available, or return None after timeout.
        """
        end_time = time.time() + timeout
        while rclpy.ok() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                t = self.tf_buffer.lookup_transform(
                    target,
                    source,
                    rclpy.time.Time())
                return t   # got it
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                # Not ready yet, keep looping
                continue
        return None


    def cb(self, msg: Pose):

        # msg contains the cube pose in world frame T_wo
        T_wo = ros_pose_to_rt(msg)

        # query fetch base link pose in Gazebo world T_wb
        t = self.wait_for_tf(
                'world',          # target_frame
                'base_link')      # source_frame
        p = Pose()
        p.position.x = t.transform.translation.x       
        p.position.y = t.transform.translation.y
        p.position.z = t.transform.translation.z
        p.orientation = t.transform.rotation
        T_wb = ros_pose_to_rt(p)
    
        # compute the object pose in robot base link T_bo
        T_bo = np.matmul(np.linalg.inv(T_wb), T_wo)
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


if __name__ == "__main__":
    rclpy.init()
    rclpy.spin(PoseToTF())
    rclpy.shutdown()