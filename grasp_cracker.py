#!/usr/bin/env python

"""
CS 6341 Homework 5 Programming
Grasp a Cracker Box
"""

import sys, time
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
import trimesh
from transforms3d.quaternions import mat2quat, quat2mat
from parse_grasps import parse_grasps, extract_grasps


# joint limits of the Fetch robot
joint_limits = {'bellows_joint': [0.0, 0.4],
    'elbow_flex_joint': [-2.251, 2.251],
    'forearm_roll_joint': [-3.14, 3.14],
    'head_pan_joint': [-1.57, 1.57],
    'head_tilt_joint': [-0.76, 1.45],
    'l_gripper_finger_joint': [0.0, 0.05],
    'l_wheel_joint': [-3.14, 3.14],
    'r_gripper_finger_joint':  [0.0, 0.05],
    'r_wheel_joint': [-3.14, 3.14],
    'shoulder_lift_joint': [-1.221, 1.518],
    'shoulder_pan_joint': [-1.6056, 1.6056],
    'torso_lift_joint': [0.371, 0.38615],
    'upperarm_roll_joint': [-3.14, 3.14],
    'wrist_flex_joint': [-2.16, 2.16],
    'wrist_roll_joint': [-3.14, 3.14],
}

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


# convert a 4x4 RT matrix to ros pose
def rt_to_ros_pose(pose, rt):

    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = rt[:3, 3]

    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]

    pose.position.x = trans[0]
    pose.position.y = trans[1]
    pose.position.z = trans[2]

    return pose


# convert a 4x4 RT matrix to quaternion and translation
def rt_to_ros_qt(rt):
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = rt[:3, 3]

    return quat, trans


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


# query ROS2 message once
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


# obtain the current joint state of the robot
def get_current_joint_states(node, is_gripper=False):
    """Return one JointState message from /joint_states, or None if timeout."""
    if is_gripper:
        names = robot.gripper_joint_names()
    else:
        names = robot.joint_names()
    while 1:
        msg = get_message_once(node, '/joint_states', JointState, timeout=2.0)
        # get the joint names in the Fetch robot group
        joint_positions = extract_specific_joints(msg, names)
        if joint_positions is not None:
            break

    # clip joint positions using limits
    for i in range(len(names)):
        pos = joint_positions[i]
        bounds = joint_limits[names[i]]
        if pos < bounds[0]:
            pos = bounds[0] + 0.0001
        if pos > bounds[1]:
            pos = bounds[1] - 0.0001
        joint_positions[i] = pos
        print(names[i], joint_positions[i])
    return joint_positions


# extract the joint positions from the msg
def extract_specific_joints(msg, names):
    joint_positions = np.zeros((len(names), ), np.float64)
    for (i, name) in enumerate(names):
        joint_positions[i] = msg.position[msg.name.index(name)]
    return joint_positions
        

'''
RT_grasps_base is with shape (50, 4, 4): 50 grasps in the robot base frame
The plan_grasp function tries to plan a trajectory to each grasp. It stops when a plan is found.
A standoff is a gripper pose with a short distance along x-axis of the gripper frame before grasping the object.
'''       
def plan_grasp(moveit2, RT_grasps_base, grasp_index):
    
    n = RT_grasps_base.shape[0]
    pose_standoff = np.eye(4)
    pose_standoff[0, 3] = -0.1
    
    # for each grasp    
    for i in range(n):
        RT_grasp = RT_grasps_base[i]
        grasp_idx = grasp_index[i]
    
        standoff_grasp_global = np.matmul(RT_grasp, pose_standoff)

        ################ TO DO: plan to standoff ##########################
        # use pymoveit2 to check if there is a plan to the standoff grasping pose
        # use the function moveit2.plan_to_pose:
        # https://github.com/IRVLUTD/pymoveit2/blob/d84d986586e4e64dfcb6c8c3ee43d8b2329d1c1f/pymoveit2/moveit2.py#L443
        # save the result to trajectory
        trajectory = None


        
        ################ TO DO: plan to standoff ##########################   
    
        if trajectory:
            print('find a plan for grasp')
            print(RT_grasp)
            print('grasp idx', grasp_idx)
            print('grasp index', grasp_index)
            break
        else:
            print('no plan for grasp %d with index %d' % (i, grasp_idx))

    if not trajectory:
        print('no plan found')
        return None, -1
            
    return RT_grasp, grasp_idx    


# first plan to the standoff pose, then move the the grasping pose
def grasp(node, moveit2, gripper_interface, RT_grasp, joint_positions_initial):

    # plan to standoff pose
    pose_standoff = np.eye(4)
    pose_standoff[0, 3] = -0.1    
    standoff_grasp_global = np.matmul(RT_grasp, pose_standoff)
    
    # move toe standoff pose
    input('move to standoff pose?')
    q_xyzw, p = rt_to_ros_qt(standoff_grasp_global)  # xyzw for quat
    joint_positions = get_current_joint_states(node)  
    moveit2.move_to_pose(
        position=p,
        quat_xyzw=q_xyzw,
        cartesian=False,
        cartesian_max_step=0.0025,
        cartesian_fraction_threshold=0.0,
        start_joint_state = joint_positions,
    )
    moveit2.wait_until_executed()
    time.sleep(2.0)
    
    # move to grasp pose
    moveit2.remove_collision_object(id='cracker_box')
    input('move to grasp pose?')
    pose_standoff[0, 3] = 0.02    
    grasp_global = np.matmul(RT_grasp, pose_standoff)
    q_xyzw, p = rt_to_ros_qt(grasp_global)  # xyzw for quat 
    joint_positions = get_current_joint_states(node)   
    moveit2.move_to_pose(
        position=p,
        quat_xyzw=q_xyzw,
        cartesian=True,
        cartesian_max_step=0.0025,
        cartesian_fraction_threshold=0.0,
        start_joint_state = joint_positions,
    )
    moveit2.wait_until_executed()
    time.sleep(2.0)
    
    # close gripper
    input('close the gripper?')
    joint_positions = get_current_joint_states(node, is_gripper=True)  
    gripper_interface.set_start_joint_state(joint_positions)
    gripper_interface.close()
    gripper_interface.wait_until_executed()
    time.sleep(2.0)

    # lift object
    input('lift object?')
    RT_lift = RT_grasp.copy()
    RT_lift[2, 3] += 0.10
    q_xyzw, p = rt_to_ros_qt(RT_lift)  # xyzw for quat
    joint_positions = get_current_joint_states(node)    
    moveit2.move_to_pose(
        position=p,
        quat_xyzw=q_xyzw,
        cartesian=True,
        cartesian_max_step=0.0025,
        cartesian_fraction_threshold=0.0,
        start_joint_state = joint_positions,
    )
    moveit2.wait_until_executed()

    # move back
    input('move back?')
    joint_positions = get_current_joint_states(node) 
    moveit2.move_to_configuration(joint_positions_initial, start_joint_state=joint_positions)
    moveit2.wait_until_executed()    

    
ROBOT = 'fetch'
# Query pose of frames from the Gazebo environment
def get_pose_gazebo(node, name):

    pose_model = get_message_once(node, f'/model/{name}/pose', Pose, timeout=2.0)
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


# sort grasps according to distances to gripper
def sort_grasps(RT_obj, RT_gripper, RT_grasps):
    # translate all RT graspits grasps using the object mean
    # transform grasps to robot base
    n = RT_grasps.shape[0]
    RT_grasps_base = np.zeros_like(RT_grasps)
    distances = np.zeros((n, ), dtype=np.float32)
    for i in range(n):
        RT_g = RT_grasps[i]
        # transform grasp to robot base
        RT = RT_obj @ RT_g
        RT_grasps_base[i] = RT
        d = np.linalg.norm(RT_gripper[:3, 3] - RT[:3, 3])
        distances[i] = d
    
    index = np.argsort(distances)
    RT_grasps_base = RT_grasps_base[index]
    print(distances)
    print(index)
    return RT_grasps_base, index


# publish a pose as a TF frame for visualization
class FrameBroadcaster(Node):
    def __init__(self, p=(0.5,0,0.2), q=(0,0,0,1)):
        super().__init__('frame_broadcaster')
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.timer_cb)
        self.p = [float(x) for x in p]
        self.q = [float(x) for x in q]

    def timer_cb(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'RT_grasp'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = self.p
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = self.q
        self.br.sendTransform(t)


if __name__ == "__main__":
    """
    Main function to run the code
    """
    
    rclpy.init()

    # Create node for this example
    node_moveit = Node("cs6341_homework5")    

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

    joint_positions = get_current_joint_states(node_moveit, is_gripper=True)  
    gripper_interface.set_start_joint_state(joint_positions)
    gripper_interface.toggle()
    gripper_interface.wait_until_executed()

    joint_positions = get_current_joint_states(node_moveit, is_gripper=True)  
    gripper_interface.set_start_joint_state(joint_positions)
    gripper_interface.open()
    gripper_interface.wait_until_executed()

    # add a collision box in moveit scene for the table
    object_id = 'table'
    position = [0.9, 0, 0.3]
    dimensions = [1, 5, 1]
    quat_xyzw = [0, 0, 0, 1]
    moveit2.add_collision_box(
        id=object_id, position=position, quat_xyzw=quat_xyzw, size=dimensions
    )

    # get object pose
    name = 'cracker_box'
    # query the pose of the cracker box
    while 1:
        RT_obj = get_pose_gazebo(node_moveit, name)
        if RT_obj is not None:
            break
    print('RT_obj', RT_obj)

    # load the cracker box mesh
    filename = '003_cracker_box.ply'
    q_xyzw, p = rt_to_ros_qt(RT_obj)  # xyzw for quat  
    moveit2.add_collision_mesh(
        filepath=filename,
        id=name,
        position=p,
        quat_xyzw=q_xyzw,
        scale=1.0
    )         

    input('next?')    
            
    # load grasps
    filename = 'refined_003_cracker_box_google_16k_textured_scale_1000-fetch_gripper.json'
    RT_grasps = parse_grasps(filename)
        
    # current gripper pose from forward kinematics  
    joint_positions = get_current_joint_states(node_moveit)
    print(joint_positions)
    retval = None
    retval = moveit2.compute_fk(joint_positions)
    if retval is None:
        print("Forward kinematics failed.")
        sys.exit(0)
    else:
        print("Forward kinematics succeeded. Result: " + str(retval))
        print("---------------------------")
        # extract the pose for the result
        p = retval.pose.position
        q = retval.pose.orientation
        print(f"{robot.end_effector_name()} in {retval.header.frame_id}: "
              f"pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}), "
              f"quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})")     

        RT_gripper = ros_pose_to_rt(retval.pose)
        print('gripper pose:', RT_gripper)
        
    # sort grasps according to distances to gripper
    RT_grasps_base, grasp_index = sort_grasps(RT_obj, RT_gripper, RT_grasps)
        
    # grasp planning
    RT_grasp, grasp_num = plan_grasp(moveit2, RT_grasps_base, grasp_index)

    # publish the frame for debugging
    # Start executor in a separate thread (non-blocking)
    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node_moveit)
    executor.add_node(node_gripper)
    q_xyzw, p = rt_to_ros_qt(RT_grasp)  # xyzw for quat 
    node_tf = FrameBroadcaster(p, q_xyzw)
    executor.add_node(node_tf)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
        
    # grasp object
    grasp(node_moveit, moveit2, gripper_interface, RT_grasp, joint_positions)

    # Clean shutdown
    print('finished grasping the cracker box')
    executor.shutdown()
    executor_thread.join()
    node_moveit.destroy_node()
    node_gripper.destroy_node()
    node_tf.destroy_node()
    rclpy.shutdown()   