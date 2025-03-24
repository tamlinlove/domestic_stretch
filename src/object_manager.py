#!/usr/bin/env python3

import rospy
import genpy
import tf2_ros as tf
from typing import List
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
import tf2_geometry_msgs
from stretch_manipulation.srv import (
    GetDetectedObjectNames,
    GetDetectedObjectNamesResponse,
    GetObjectPosition,
    GetObjectPositionRequest,
    GetObjectPositionResponse,
)

NODE_NAME = "object_manager"
OBJECTS_TARGET_FRAME = "odom"
DETECTION_TOPIC = "/objects/marker_array"
ALL_OBJECTS_TOPIC = f"/{NODE_NAME}/objects/all"
GET_OBJECT_POSITION_SERVICE = f"/{NODE_NAME}/get_object_position"
FORCE_OBJECT_POSITION_SERVICE = f"/{NODE_NAME}/force_object_position"
CLEAR_ALL_OBJECTS_SERVICE = f"/{NODE_NAME}/clear_all_objects"
GET_DETECTED_OBJECT_NAMES_SERVICE = f"/{NODE_NAME}/get_detected_object_names"

FORCE_OBJECT_POSITION_WAIT_TIME = 5


class ObjectManager:
    def __init__(self):
        # ROS
        rospy.init_node(NODE_NAME)

        # Variables
        self.object_coordinates = {}
        self.current_objects = []
        self.tf_buffer = tf.Buffer(rospy.Duration(100))
        self.tf_listener = tf.TransformListener(self.tf_buffer)
        rospy.sleep(1)

        # Publishers
        self.all_objects_pub = rospy.Publisher(
            ALL_OBJECTS_TOPIC, MarkerArray, queue_size=10
        )

        # Subscribers
        rospy.Subscriber(DETECTION_TOPIC, MarkerArray, self.detection_callback)
        rospy.wait_for_message(DETECTION_TOPIC, MarkerArray)

        # Services
        rospy.Service(
            GET_OBJECT_POSITION_SERVICE,
            GetObjectPosition,
            self.get_object_position,
        )
        rospy.Service(
            FORCE_OBJECT_POSITION_SERVICE,
            GetObjectPosition,
            self.force_get_object_position,
        )
        rospy.Service(CLEAR_ALL_OBJECTS_SERVICE, Trigger, self.clear_all_objects)
        rospy.Service(GET_DETECTED_OBJECT_NAMES_SERVICE, GetDetectedObjectNames, self.get_detected_object_names)

    def transform_to_tf(
        self,
        pose: PoseStamped,
        to_frame: str,
        time: genpy.rostime.Time = genpy.rostime.Time(0),
    ):
        transform = self.tf_buffer.lookup_transform(
            to_frame, pose.header.frame_id, time, rospy.Duration(1)
        )
        return tf2_geometry_msgs.do_transform_pose(pose, transform)
    
    def get_detected_object_names(self, *_) -> GetDetectedObjectNamesResponse:
        return GetDetectedObjectNamesResponse(names=self.current_objects)

    def detection_callback(self, msg: MarkerArray):
        # Filter out markers that have no size (i.e. are not localized)
        if msg.markers is None:
            msg.markers = []

        markers: List[Marker] = [
            marker
            for marker in msg.markers
            if abs(marker.scale.x) >= 0.01 and abs(marker.scale.y) >= 0.01
        ]
        
        current_objects = []

        # Add the markers to the dictionary
        for marker in markers:
            marker_pose_camera = PoseStamped()
            marker_pose_camera.header.frame_id = marker.header.frame_id
            marker_pose_camera.header.stamp = marker.header.stamp
            marker_pose_camera.pose = marker.pose

            try:
                marker_pose_odom = self.transform_to_tf(
                    marker_pose_camera, OBJECTS_TARGET_FRAME, marker.header.stamp
                )

                marker.pose = marker_pose_odom.pose
                marker.header = marker_pose_odom.header
                # TODO: Marker ID should be a unique int32
                marker.id = hash(marker.text) % 2**31

                self.object_coordinates[marker.text] = marker
                current_objects.append(marker.text)
            except Exception as e:
                rospy.logerr(f"Error transforming marker pose: {marker.text} ({e})")
                
        self.current_objects = current_objects

        # Publish all the markers
        self.all_objects_pub.publish(
            MarkerArray(markers=list(self.object_coordinates.values()))
        )

    def force_get_object_position(
        self, req: GetObjectPositionRequest
    ) -> GetObjectPositionResponse:
        object_class = req.object_class
        if object_class in self.object_coordinates:
            object_position: Marker = self.object_coordinates[object_class]
            waiting_time = 0.0
            while self.object_coordinates[object_class].pose == object_position.pose and waiting_time < FORCE_OBJECT_POSITION_WAIT_TIME:
                rospy.loginfo(
                    f"Force getting object position for {waiting_time}: {object_position.pose.position}"
                )
                rospy.sleep(0.1)
                waiting_time += 0.1
            return GetObjectPositionResponse(
                success=True,
                pose=PoseStamped(
                    header=self.object_coordinates[object_class].header,
                    pose=self.object_coordinates[object_class].pose,
                ),
            )
        return GetObjectPositionResponse(success=False)

    def get_object_position(
        self, req: GetObjectPositionRequest
    ) -> GetObjectPositionResponse:
        object_class = req.object_class

        if object_class in self.object_coordinates:
            return GetObjectPositionResponse(
                success=True,
                pose=PoseStamped(
                    header=self.object_coordinates[object_class].header,
                    pose=self.object_coordinates[object_class].pose,
                ),
            )
        return GetObjectPositionResponse(success=False)

    def clear_all_objects(self, *_: TriggerRequest) -> TriggerResponse:
        self.object_coordinates = {}
        return TriggerResponse(success=True)


if __name__ == "__main__":
    object_manager = ObjectManager()
    rospy.loginfo("Object manager is running...")
    rospy.spin()
