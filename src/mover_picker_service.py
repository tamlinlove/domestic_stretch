#!/usr/bin/env python3

from typing import Tuple

import tf2_ros as tf
import tf2_geometry_msgs
import rospy
import ikpy
import numpy as np
import genpy
import actionlib
import hello_helpers.hello_misc as hm  # type: ignore
from manipulation.ik import StretchDexIK
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from geometry_msgs.msg import PointStamped, PoseStamped, Pose
from visualization_msgs.msg import Marker
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from stretch_body.hello_utils import deg_to_rad  # type: ignore
from sensor_msgs.msg import JointState
from stretch_manipulation.srv import (
    GoTo,
    GoToRequest,
    GoToResponse,
    LookAt,
    LookAtRequest,
    LookAtResponse,
    GraspPosition,
    GraspPositionRequest,
    GraspPositionResponse,
)

# Constant variables
NODE_NAME = "robot_manager"
GRASP_ERROR_THRESHOLD = 0.13
CONSTANT_ERROR_TRANSLATE = 0.02
CONSTANT_ERROR_EXTEND = 0.06
CONSTANT_ERROR_LIFT = -0.015
OPEN_GRIPPER = 0.08
CLOSE_GRIPPER = -0.06
MAX_MOVE_TRIES = 20

# Constant topics
JOINTS_STATE_TOPIC = "/stretch/joint_states"
LAST_GRASP_TRAJECTORY_TOPIC = f"/{NODE_NAME}/last_grasp_trajectory"

# Constant frames
BASE_TARGET_FRAME = "base_link"
OBJECTS_TARGET_FRAME = "odom"
GRIPPER_GRASP_CENTER_FRAME = "link_grasp_center"

# Constant actions
MOVE_BASE_ACTION = "move_base"

MOVE_BASE_GOAL_STATUS = [
    "PENDING",
    "ACTIVE",
    "PREEMPTED",
    "SUCCEEDED",
    "ABORTED",
    "REJECTED",
    "PREEMPTING",
    "RECALLING",
    "RECALLED",
    "LOST"
]

# Constant services
GO_TO_SERVICE = f"{NODE_NAME}/go_to_service"
HEAD_SCAN_SERVICE = f"{NODE_NAME}/head_scan_service"
LOOK_AT_SERVICE = f"{NODE_NAME}/look_at_service"
MOVE_TO_GRASPING_POSITION_SERVICE = f"/{NODE_NAME}/move_to_grasping_position_service"
GRASP_POSITION_SERVICE = f"{NODE_NAME}/grasp_position_service"
RETRIEVE_THE_ROBOT_SERVICE = f"{NODE_NAME}/retrieve_the_robot_service"
HANDOVER_OBJECT_SERVICE = f"{NODE_NAME}/handover_object_service"
OPEN_GRIPPER_SERVICE = f"{NODE_NAME}/open_gripper_service"
LOOK_AT_GRIPPER_SERVICE = f"{NODE_NAME}/look_at_gripper_service"

# Constant service proxies
SWITCH_TO_POSITION_MODE_SERVICE = "/switch_to_position_mode"
SWITCH_TO_NAVIGATION_MODE_SERVICE = "/switch_to_navigation_mode"


class RobotPicker(hm.HelloNode):
    """
    Class representing a robot picker.

    This class inherits from `hm.HelloNode` and implements various methods and callbacks
    for picking objects using a robot arm.
    """

    def __init__(self):
        # Initialize the super class
        super().__init__()
        super().main(NODE_NAME, NODE_NAME, wait_for_first_pointcloud=False)

        # Variables
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.tf_buffer = tf.Buffer(rospy.Duration(100))
        self.tf_listener = tf.TransformListener(self.tf_buffer)

        # Publishers
        self.last_grasp_trajectory_pub = rospy.Publisher(
            LAST_GRASP_TRAJECTORY_TOPIC, Marker, queue_size=10
        )
        self.gripper_position_pub = rospy.Publisher(
            f"/{NODE_NAME}/gripper_position", PoseStamped, queue_size=10
        )

        # Actions
        self.move_base = actionlib.SimpleActionClient(MOVE_BASE_ACTION, MoveBaseAction)
        self.move_base.wait_for_server()

        # Subscribers
        self.joint_state_sub = rospy.Subscriber(
            JOINTS_STATE_TOPIC, JointState, self.joint_state_callback
        )
        rospy.wait_for_message(JOINTS_STATE_TOPIC, JointState)

        # Service proxies
        self.switch_to_position_mode = rospy.ServiceProxy(
            SWITCH_TO_POSITION_MODE_SERVICE, Trigger
        )
        self.switch_to_position_mode.wait_for_service()
        self.switch_to_navigation_mode = rospy.ServiceProxy(
            SWITCH_TO_NAVIGATION_MODE_SERVICE, Trigger
        )
        self.switch_to_navigation_mode.wait_for_service()
        self.switch_to_navigation_mode()

        # Services
        rospy.Service(GO_TO_SERVICE, GoTo, self.go_to_callback)
        rospy.Service(HEAD_SCAN_SERVICE, Trigger, self.head_scan_callback)
        rospy.Service(LOOK_AT_SERVICE, LookAt, self.look_at_callback)
        rospy.Service(
            MOVE_TO_GRASPING_POSITION_SERVICE,
            GraspPosition,
            self.move_to_grasping_position,
        )
        rospy.Service(
            GRASP_POSITION_SERVICE, GraspPosition, self.grasp_position_callback
        )
        rospy.Service(
            RETRIEVE_THE_ROBOT_SERVICE, Trigger, self.retrieve_the_robot_callback
        )
        rospy.Service(
            LOOK_AT_GRIPPER_SERVICE, Trigger, self.look_at_gripper_callback
        )
        rospy.Service(HANDOVER_OBJECT_SERVICE, Trigger, self.handover_object_callback)
        rospy.Service(OPEN_GRIPPER_SERVICE, Trigger, self.open_gripper_callback)

    #######################
    # CALLBACK VARIABLES
    #######################

    def joint_state_callback(self, msg: JointState):
        self.joint_positions = dict(zip(msg.name, msg.position))
        self.joint_velocities = dict(zip(msg.name, msg.velocity))
        self.joint_efforts = dict(zip(msg.name, msg.effort))

    #######################
    # DEBUG
    #######################

    def print_grasp_arrow(self, origin: PoseStamped, target: PoseStamped):
        assert origin.header.frame_id == target.header.frame_id, "Frames are different"
        marker = Marker()
        marker.header.frame_id = origin.header.frame_id
        marker.header.stamp = origin.header.stamp
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.02
        marker.scale.z = 0.03
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.points = [
            origin.pose.position,
            target.pose.position,
        ]
        self.last_grasp_trajectory_pub.publish(marker)

    #######################
    # UTILS
    #######################

    def euclidean_distance(self, point1: Pose, point2: Pose) -> float:
        return (
            (point1.position.x - point2.position.x) ** 2
            + (point1.position.y - point2.position.y) ** 2
            + (point1.position.z - point2.position.z) ** 2
        ) ** 0.5

    def transform_to_tf(
        self,
        pose: PoseStamped,
        to_frame: str,
        time: genpy.rostime.Time = genpy.rostime.Time(0),
    ):
        assert pose.header.frame_id is not None, "Pose frame_id is None"
        assert pose.header.frame_id != "", "Pose frame_id is empty"

        rospy.loginfo(f"Transforming pose from {pose.header.frame_id} to {to_frame}")
        transform = self.tf_buffer.lookup_transform(
            to_frame, pose.header.frame_id, time, rospy.Duration(1)
        )
        return tf2_geometry_msgs.do_transform_pose(pose, transform)

    def get_joint_state(self, joint_name, moving_threshold=0.001):
        # name: [joint_arm_l0, joint_arm_l1, joint_arm_l2, joint_arm_l3, joint_gripper_finger_left,
        # joint_gripper_finger_right, joint_head_pan, joint_head_tilt, joint_left_wheel, joint_lift,
        # joint_right_wheel, joint_wrist_yaw]

        joint_position = self.joint_positions[joint_name]
        joint_velocity = self.joint_velocities[joint_name]
        joint_effort = self.joint_efforts[joint_name]
        joint_is_moving = abs(joint_velocity) > moving_threshold

        return (joint_position, joint_velocity, joint_effort, joint_is_moving)

    def get_gripper_pose(self) -> PoseStamped:
        # Get the pose of the gripper
        gripper_pose = PoseStamped()
        gripper_pose.header.frame_id = GRIPPER_GRASP_CENTER_FRAME
        gripper_pose.header.stamp = rospy.Time.now()
        gripper_pose.pose.orientation.w = 1
        gripper_pose_odom = self.transform_to_tf(gripper_pose, OBJECTS_TARGET_FRAME)
        return gripper_pose_odom

    def calculate_distance_from_gripper(
        self, pose: PoseStamped
    ) -> Tuple[float, float, float]:

        # Calculate point in base frame
        pose_in_base_frame = self.transform_to_tf(
            pose, BASE_TARGET_FRAME, pose.header.stamp
        )
        rospy.loginfo(f"Point in gripper frame: {pose_in_base_frame}")

        # Calculate gripper in base frame
        gripper_pose = self.get_gripper_pose()
        gripper_in_base_frame = self.transform_to_tf(
            gripper_pose, BASE_TARGET_FRAME, gripper_pose.header.stamp
        )
        rospy.loginfo(f"Gripper in base frame: {gripper_in_base_frame}")

        self.print_grasp_arrow(gripper_in_base_frame, pose_in_base_frame)
        self.gripper_position_pub.publish(gripper_in_base_frame)

        x = pose_in_base_frame.pose.position.x - gripper_in_base_frame.pose.position.x
        y = pose_in_base_frame.pose.position.y - gripper_in_base_frame.pose.position.y
        z = pose_in_base_frame.pose.position.z - gripper_in_base_frame.pose.position.z

        rospy.loginfo(f"Calculated distances: {x, y, z}")

        return x, y, z

    def move_to_grasp_object(self, pose: PoseStamped):
        x, _, z = self.calculate_distance_from_gripper(pose)

        final_x = CONSTANT_ERROR_TRANSLATE + x
        final_z = self.joint_positions["joint_lift"] + z + CONSTANT_ERROR_LIFT
        rospy.loginfo(f"Final distances: {final_x, final_z}")

        self.move_to_pose(
            {
                "translate_mobile_base": final_x,
            }
        )

    def grasp_object(self, pose: PoseStamped):
        # ALL_JOINTS = 'joint_head_tilt' | 'joint_head_pan' | 'joint_gripper_finger_left'
        # | 'wrist_extension' | 'joint_lift' | 'joint_wrist_yaw' | "translate_mobile_base" | "rotate_mobile_base"
        # | 'gripper_aperture' | 'joint_arm_l0' | 'joint_arm_l1' | 'joint_arm_l2' | 'joint_arm_l3' | 'joint_wrist_pitch'
        # | 'joint_wrist_roll';

        # Calculate distances
        x, _, z = self.calculate_distance_from_gripper(pose)

        final_x = CONSTANT_ERROR_TRANSLATE + x
        final_z = self.joint_positions["joint_lift"] + z + CONSTANT_ERROR_LIFT
        rospy.loginfo(f"Final distances: {final_x, final_z}")

        # self.move_to_pose(
        #     {
        #         "translate_mobile_base": final_x,
        #     }
        # )

        # Move arm down
        self.move_to_pose(
            {
                "joint_lift": final_z,
            }
        )

        # Orient gripper
        self.move_to_pose(
            {
                "joint_wrist_yaw": deg_to_rad(0.0),
                "gripper_aperture": OPEN_GRIPPER,
            }
        )

        # Calculate extension after performing the yaw
        rospy.wait_for_message(JOINTS_STATE_TOPIC, JointState)
        _, y, _ = self.calculate_distance_from_gripper(pose)
        final_y = -(
            -self.joint_positions["wrist_extension"] + y - CONSTANT_ERROR_EXTEND
        )

        # Move arm forward
        self.move_to_pose(
            {
                "wrist_extension": final_y,
            }
        )

        # position_g, velocity_g, effort_g, moving_g = self.get_joint_state(
        #     "joint_gripper_finger_left"
        # )
        # rospy.loginfo(
        #     f"Position: {position_g}, velocity: {velocity_g}, effort: {effort_g}, moving: {moving_g}"
        # )

        # Close gripper
        self.move_to_pose(
            {"gripper_aperture": (0.0, 0.01, 0.01, 40)}, custom_full_goal=True
        )

        # position_g, velocity_g, effort_g, moving_g = self.get_joint_state(
        #     "joint_gripper_finger_left"
        # )
        # rospy.loginfo(
        #     f"Position: {position_g}, velocity: {velocity_g}, effort: {effort_g}, moving: {moving_g}"
        # )

        # rospy.wait_for_message(JOINTS_STATE_TOPIC, JointState)
        # rospy.wait_for_message(JOINTS_STATE_TOPIC, JointState)
        # rospy.wait_for_message(JOINTS_STATE_TOPIC, JointState)

        # Move arm up and back
        self.move_to_pose(
            {
                "joint_lift": self.joint_positions["joint_lift"] + 0.1,
                "wrist_extension": 0.0,
            }
        )

    #######################
    # CALLBACK SERVICES
    #######################

    def go_to_callback(self, request: GoToRequest) -> GoToResponse:

        status = None
        tries = 0

        while status != actionlib.GoalStatus.SUCCEEDED and tries < MAX_MOVE_TRIES:

            # Go to the specified coordinates
            move_base_action_goal = MoveBaseGoal()
            move_base_action_goal.target_pose = request.goal
            #rospy.loginfo(f"Sending goal: \n{move_base_action_goal}")

            def done_callback(state, result):
               rospy.loginfo(f"Action finished with result: {result} {state}")

            self.move_base.send_goal(move_base_action_goal, done_cb=done_callback)
            #rospy.loginfo("GOAL SENT, WAITING FOR RESULTS")
            result = self.move_base.wait_for_result(timeout=rospy.Duration(2))
            '''
            if not result:
                rospy.logerr("Move base action server failed to respond")
                return GoToResponse(success=False)
            '''
            
            # Check if the action was successful
            state = self.move_base.get_state()
            rospy.loginfo(f"Move_base state: {MOVE_BASE_GOAL_STATUS[state]}")
            if state != actionlib.GoalStatus.SUCCEEDED:
                rospy.logerr(
                    f"Move base action server failed to reach the goal. State: {state}"
                )

            status = state
            tries += 1
            rospy.loginfo(f"NUM TRIES: {tries}")

        return GoToResponse(success=status == actionlib.GoalStatus.SUCCEEDED)

    def head_scan_callback(self, *_: TriggerRequest) -> TriggerResponse:
        # Save old position
        rospy.wait_for_message(JOINTS_STATE_TOPIC, JointState)
        old_position = {
            "joint_head_tilt": self.joint_positions["joint_head_tilt"],
            "joint_head_pan": self.joint_positions["joint_head_pan"],
        }

        # Tilt the head to 45 degrees
        self.move_to_pose({"joint_head_tilt": deg_to_rad(-45)})

        for degrees in range(-180, 1, 45):
            rospy.loginfo(f"Head scan at {degrees} degrees")
            # Move head to 0 degrees
            self.move_to_pose({"joint_head_pan": deg_to_rad(degrees)})
            rospy.sleep(2)

        # Restore old position
        self.move_to_pose(old_position)
        return TriggerResponse(success=True)

    # TODO: Proper look at function: https://github.com/UTNuclearRoboticsPublic/look_at_pose/blob/kinetic/nodes/look_at_pose_server
    def look_at_callback(self, *_: LookAtRequest) -> LookAtResponse:
        """
        Given a position in space (x, y, z). It calculates the tilt and pan of the head to look at that position.
        """
        self.move_to_pose(
            {
                "joint_head_tilt": deg_to_rad(-45),
                "joint_head_pan": deg_to_rad(-90),
                "joint_wrist_pitch": deg_to_rad(-20),
            }
        )

        return LookAtResponse(success=True)
    
    def look_at_gripper_callback(self, _: TriggerRequest) -> TriggerResponse:
        self.switch_to_position_mode.call()
        self.move_to_pose(
            {
                "joint_wrist_yaw": deg_to_rad(120),
                "joint_head_tilt": deg_to_rad(-50),
                "joint_head_pan": deg_to_rad(0),
                #"joint_lift": self.joint_positions["joint_lift"] - 0.2,
            }
        )
        self.switch_to_navigation_mode.call()
        return TriggerResponse(success=True)

    def move_to_grasping_position(
        self, request: GraspPositionRequest
    ) -> GraspPositionResponse:
        rospy.loginfo(f"Moving to grasp object at {request.pose}")
        self.switch_to_position_mode.call()
        self.move_to_grasp_object(request.pose)
        self.switch_to_navigation_mode.call()
        return GraspPositionResponse(success=True)

    def grasp_position_callback(
        self, request: GraspPositionRequest
    ) -> GraspPositionResponse:
        rospy.loginfo(f"Grasping object at {request.pose}")
        self.switch_to_position_mode.call()
        self.grasp_object(request.pose)
        self.switch_to_navigation_mode.call()
        return GraspPositionResponse(success=True)

    def retrieve_the_robot_callback(self, _: TriggerRequest) -> TriggerResponse:
        self.switch_to_position_mode.call()
        self.move_to_pose(
            {
                "joint_wrist_yaw": deg_to_rad(120),
                "joint_head_tilt": deg_to_rad(0),
                "joint_head_pan": deg_to_rad(0),
                "joint_lift": self.joint_positions["joint_lift"] - 0.2,
            }
        )
        self.switch_to_navigation_mode.call()
        return TriggerResponse(success=True)

    def handover_object_callback(self, *_: TriggerRequest) -> TriggerResponse:
        self.switch_to_position_mode.call()
        self.move_to_pose(
            {
                "translate_mobile_base": 0.1,
                "joint_wrist_pitch": deg_to_rad(0),
            }
        )
        self.move_to_pose(
            {
                "gripper_aperture": OPEN_GRIPPER,
            }
        )
        self.move_to_pose(
            {
                "joint_wrist_pitch": deg_to_rad(-20),
            }
        )
        self.switch_to_navigation_mode.call()
        return TriggerResponse(success=True)

    def open_gripper_callback(self, *_: TriggerRequest) -> TriggerResponse:
        self.switch_to_position_mode.call()
        self.move_to_pose(
            {
                "gripper_aperture": OPEN_GRIPPER,
            }
        )
        self.switch_to_navigation_mode.call()
        return TriggerResponse(success=True)


if __name__ == "__main__":
    robot_manager = RobotPicker()
    rospy.loginfo("Robot picker is running...")
    rospy.spin()
