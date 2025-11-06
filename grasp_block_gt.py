#!/usr/bin/env python

"""
CS 6341 Homework 4 Programming
Grasping
"""

from threading import Thread
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from pymoveit2 import MoveIt2
from pymoveit2 import GripperInterface
from pymoveit2.robots import fetch as robot

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX


def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


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


def get_current_joint_states(node):
    """Return one JointState message from /joint_states, or None if timeout."""
    while 1:
        msg = get_message_once(node, '/joint_states', JointState, timeout=2.0)
        print(msg)

        # get the joint names in the Fetch robot group
        names = robot.joint_names()
        joint_positions = extract_specific_joints(msg, names)
        if joint_positions is not None:
            break
    return joint_positions


# extract the joint positions from the msg
def extract_specific_joints(msg, names):
    joint_positions = np.zeros((len(names), ), np.float64)
    for (i, name) in enumerate(names):
        joint_positions[i] = msg.position[msg.name.index(name)]
        print(name, joint_positions[i])
    return joint_positions


class FrameBroadcaster(Node):
    def __init__(self, p=(0.5,0,0.2), q=(0,0,0,1)):
        super().__init__('frame_broadcaster')
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.timer_cb)
        self.p = p
        self.q = q

    def timer_cb(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'my_frame'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = self.p
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = self.q
        self.br.sendTransform(t)

    
if __name__ == "__main__":
    """
    Main function to run the code
    """

    rclpy.init()

    # Create node for this example
    node_moveit = Node("cs6341_homework4")    

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    # Create MoveIt 2 interface
    moveit2 = MoveIt2(
        node=node_moveit,
        joint_names=robot.joint_names(),
        base_link_name=robot.base_link_name(),
        end_effector_name=robot.end_effector_name(),
        group_name=robot.MOVE_GROUP_ARM,
        callback_group=callback_group,
    )    

    # Create gripper interface
    node_gripper = Node("fetch_gripper")
    gripper_interface = GripperInterface(
        node=node_gripper,
        gripper_joint_names=robot.gripper_joint_names(),
        open_gripper_joint_positions=robot.OPEN_GRIPPER_JOINT_POSITIONS,
        closed_gripper_joint_positions=robot.CLOSED_GRIPPER_JOINT_POSITIONS,
        gripper_group_name=robot.MOVE_GROUP_GRIPPER,
        callback_group=callback_group,
        gripper_command_action_name="gripper_action_controller/gripper_cmd",
    )

    gripper_interface.toggle()
    gripper_interface.wait_until_executed()
    gripper_interface.open()
    gripper_interface.wait_until_executed()

    # query the pose of the cube
    while 1:
        T_bo = get_pose_gazebo(node_moveit)
        if T_bo is not None:
            break
    print('T_bo', T_bo)

    # get the current robot joints
    joint_positions = get_current_joint_states(node_moveit)
    print(joint_positions)

    # try to figure out the end-effector pose for grapsing the block
    # orientation
    R = rotY(np.pi / 2) @ rotX(np.pi)
    print(R.shape)
    q_xyzw = ros_quat(mat2quat(R[:3, :3]))
    # position from the cube position
    position = T_bo[:3, 3]
    p = [position[0], position[1], position[2] + 0.2 + 0.06]   

    # publish the frame for debugging
    # Start executor in a separate thread (non-blocking)
    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node_moveit)
    executor.add_node(node_gripper)
    node_tf = FrameBroadcaster(p, q_xyzw)
    executor.add_node(node_tf)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()    

    # Sleep a while in order to get the first joint state
    node_moveit.create_rate(3.0).sleep()
    node_gripper.create_rate(3.0).sleep()        

    # compute IK using the end-effector pose
    print("---------------------------")
    retval_ik = None
    retval_ik = moveit2.compute_ik(p, q_xyzw)
    if retval_ik is None:
        print("Inverse kinematics failed.")
    else:
        print("Inverse kinematics succeeded. Result: " + str(retval_ik))

        # extract the arm joints
        print("---------------------------")
        names = robot.joint_names()
        joint_positions_ik = extract_specific_joints(retval_ik, names)
        print("---------------------------")

        # compute the distances between joints
        distance = np.linalg.norm(joint_positions - joint_positions_ik)
        print('joint distance:', distance)

        # use moveit2 to move to the joint position from IK
        print(moveit2.joint_state)
        input('move?')
        moveit2.move_to_configuration(joint_positions_ik)
        moveit2.wait_until_executed()

        print(moveit2.joint_state)
        input('move to grasping pose?')

        # move the grasping pose
        p[2] -= 0.05
        moveit2.move_to_pose(
            position=p,
            quat_xyzw=q_xyzw,
            cartesian=False,
            cartesian_max_step=0.0025,
            cartesian_fraction_threshold=0.0,
        )
        moveit2.wait_until_executed()

        # close gripper
        print(moveit2.joint_state)
        input('close the gripper?')
        gripper_interface.close()
        gripper_interface.wait_until_executed()

        # move to the initial configuration
        print(moveit2.joint_state)
        input('move back?')
        moveit2.move_to_configuration(joint_positions)
        moveit2.wait_until_executed()

    # Clean shutdown
    print('finished grasping the cube')
    executor.shutdown()
    executor_thread.join()
    node_gripper.destroy_node()
    node_tf.destroy_node()
    rclpy.shutdown()   