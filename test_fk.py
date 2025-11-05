import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState

def test_fk(node):
    cli = node.create_client(GetPositionFK, '/compute_fk')
    cli.wait_for_service(5.0)

    req = GetPositionFK.Request()
    req.header.frame_id = 'base_link'  # <-- your planning frame
    req.fk_link_names = ['wrist_roll_link']      # <-- one valid link name

    # Fill *all* joints in the chain with a valid configuration
    req.robot_state.joint_state = JointState(
        name=["torso_lift_joint",
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "upperarm_roll_joint",
      "elbow_flex_joint",
      "forearm_roll_joint",
      "wrist_flex_joint",
      "wrist_roll_joint"],  # <-- your joint names
        position=[0, 0, 0, 0, 0, 0, 0, 0]
    )

    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    res = fut.result()
    print('FK error code:', res.error_code.val)
    for p in res.pose_stamped:
        print(p.header.frame_id, '->', p.pose)

if __name__ == '__main__':
    rclpy.init()
    n = rclpy.create_node('fk_quickcheck')
    test_fk(n)
    n.destroy_node()
    rclpy.shutdown()