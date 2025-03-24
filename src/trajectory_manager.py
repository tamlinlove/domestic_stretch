#!/usr/bin/env python3

import csv
import os
import genpy
import tf2_ros as tf
import tf2_geometry_msgs
from typing import Dict, List, Optional
import rospy
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from nav_msgs.msg import Path
from gensim.models import KeyedVectors
from skspatial.objects import Point as SKOPoint
from skspatial.objects import Line as SKOLine
from stretch_manipulation.srv import (
    GetCurrentLocation,
    GetCurrentLocationResponse,
    GetRouteForObject,
    GetRouteForObjectRequest,
    GetRouteForObjectResponse,
    GoTo,
    GoToRequest,
    GoToResponse,
    GoToLocation,
    GoToLocationRequest,
    GoToLocationResponse,
    AddCurrentLocation,
    AddCurrentLocationRequest,
    AddCurrentLocationResponse,
    ClosestLocationToPosition,
    ClosestLocationToPositionRequest,
    ClosestLocationToPositionResponse,
)

NODE_NAME = "trajectory_manager"
LOCATIONS_CSV = "data/example_locations.csv"
GENSIM_MODEL = "/path/to/gensim/model.bin"
LOCATIONS_FRAME = "map"
LOCATION_UPDATE_TOPIC = "/amcl_pose"
LAST_ROUTE_PUBLISHER_TOPIC = f"/{NODE_NAME}/last_route"
GET_CURRENT_LOCATION_SERVICE = f"/{NODE_NAME}/get_current_location"
GET_ROUTE_FOR_OBJECT_SERVICE = f"/{NODE_NAME}/get_route_for_object"
GO_TO_LOCATION_SERVICE = f"/{NODE_NAME}/go_to_location"
ADD_CURRENT_LOCATION_SERVICE = f"/{NODE_NAME}/add_current_location"
CLOSEST_LOCATION_TO_POSITION_SERVICE = f"/{NODE_NAME}/closest_location_to_position"

ROBOT_MANAGER_NODE_NAME = "robot_manager"
GO_TO_SERVICE = f"/{ROBOT_MANAGER_NODE_NAME}/go_to_service"


def load_csv(csv_file: str):
    assert os.path.exists(csv_file), f"File {csv_file} does not exist"
    with open(csv_file, encoding="utf-8") as f:
        locations = list(csv.DictReader(f, skipinitialspace=True))
        rospy.loginfo(f"Loaded {len(locations)} locations from {csv_file}")
        return locations


def save_csv(csv_file: str, data: List[Dict]):
    with open(csv_file, mode="w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def load_gensim_model(model_file: str):
    assert os.path.exists(model_file), f"File {model_file} does not exist"
    rospy.loginfo(f"Loading gensim model from {model_file}")
    model = KeyedVectors.load_word2vec_format(model_file, binary=True)
    rospy.loginfo(f"Loaded gensim model with {len(model.index_to_key)} words")
    #rospy.loginfo(f"Words: {sorted([w for w in model.index_to_key])}")
    return model


def pose_to_line(pose: Pose) -> SKOLine:
    point = SKOPoint([pose.position.x, pose.position.y, pose.position.z])
    vector = SKOPoint([pose.orientation.x, pose.orientation.y, pose.orientation.z])
    return SKOLine(point, vector)


def pose_to_point(pose: Pose) -> SKOPoint:
    return SKOPoint([pose.position.x, pose.position.y, pose.position.z])


class TrajectoryManager:

    def __init__(self):
        # ROS
        rospy.init_node(NODE_NAME)

        # Variables
        self.current_location: Optional[PoseWithCovarianceStamped] = None
        self.locations_csv = load_csv(LOCATIONS_CSV)
        self.gensim_model = load_gensim_model(GENSIM_MODEL)
        self.tf_buffer = tf.Buffer(rospy.Duration(100))
        self.tf_listener = tf.TransformListener(self.tf_buffer)
        rospy.sleep(1)

        # Publishers
        self.last_route_pub = rospy.Publisher(
            LAST_ROUTE_PUBLISHER_TOPIC, Path, queue_size=10
        )

        # Subscribers
        rospy.Subscriber(
            LOCATION_UPDATE_TOPIC,
            PoseWithCovarianceStamped,
            self.update_location_callback,
        )
        rospy.loginfo("Waiting for position update...")
        rospy.wait_for_message(LOCATION_UPDATE_TOPIC, PoseWithCovarianceStamped)
        rospy.loginfo("Waiting for position update... Done!")

        # Services: Robot Manager
        self.go_to_service = rospy.ServiceProxy(GO_TO_SERVICE, GoTo)

        # Services
        rospy.Service(
            GET_CURRENT_LOCATION_SERVICE, GetCurrentLocation, self.get_current_location
        )
        rospy.Service(
            GET_ROUTE_FOR_OBJECT_SERVICE, GetRouteForObject, self.get_route_for_object
        )
        rospy.Service(GO_TO_LOCATION_SERVICE, GoToLocation, self.go_to_location)
        rospy.Service(
            ADD_CURRENT_LOCATION_SERVICE, AddCurrentLocation, self.add_current_location
        )
        rospy.Service(
            CLOSEST_LOCATION_TO_POSITION_SERVICE,
            ClosestLocationToPosition,
            self.closest_location_to_position,
        )

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


    def add_current_location(
        self, req: AddCurrentLocationRequest
    ) -> AddCurrentLocationResponse:
        # Make sure current location has been updated at least one
        assert self.current_location is not None, "Current location is not available"

        # Update locations csv
        l = req.location_name
        x = self.current_location.pose.pose.position.x
        y = self.current_location.pose.pose.position.y
        t = self.current_location.pose.pose.orientation.z
        w = self.current_location.pose.pose.orientation.w
        self.locations_csv.append({"name": l, "x": x, "y": y, "theta": t, "w": w})
        rospy.loginfo(f"Added location {l} at ({x}, {y}, {t}, {w}) to locations CSV")

        # Save locations csv
        save_csv(LOCATIONS_CSV, self.locations_csv)
        rospy.loginfo(
            f"Locations CSV saved to {LOCATIONS_CSV} ({len(self.locations_csv)} locations)"
        )

        return AddCurrentLocationResponse(success=True, pose=self.current_location)

    def update_location_callback(self, pose: PoseWithCovarianceStamped):
        self.current_location = pose

    def get_current_location(self, *_) -> GetCurrentLocationResponse:
        return GetCurrentLocationResponse(self.current_location)

    def get_route_for_object(
        self, req: GetRouteForObjectRequest
    ) -> GetRouteForObjectResponse:
        object_class: str = req.object_class

        semantic_similarity = self.locations_csv.copy()

        # Calculate semantic similarity object <=> location
        for location in semantic_similarity:
            try:
                location_name = location['name'].replace(" ","::")
                object_name = object_class.replace(" ","::")
                # Here we replace identified words that aren't in gensim model with some viable alternatives
                # Ideally, this should be replaced with something more robust
                if object_name == "remote":
                    # No idea why remote isn't in the model, but whatever
                    object_name = "television"
                
                # Get similarity
                similarity = self.gensim_model.similarity(
                    f"{location_name}_NOUN", f"{object_name}_NOUN"
                )
                location["semantic_similarity"] = similarity
                rospy.loginfo(f"Similarity between {location['name']} and {object_class} is {similarity}")
            except KeyError:
                rospy.logerr(f"One of these words not found in gensim model: {location_name}_NOUN, {object_name}_NOUN")
                location["semantic_similarity"] = 0.0

        # Sort locations by semantic similarity
        sorted_locations = sorted(
            semantic_similarity, key=lambda x: x["semantic_similarity"], reverse=True
        )
        names = [ sorted_location['name'] for sorted_location in sorted_locations ]
        rospy.loginfo(
            f"Sorted locations: {names}"
        )

        sorted_poses = []
        for location in sorted_locations:
            pose = PoseStamped()
            pose.header.frame_id = LOCATIONS_FRAME
            pose.pose.position.x = float(location["x"])
            pose.pose.position.y = float(location["y"])
            pose.pose.orientation.z = float(location["theta"])
            pose.pose.orientation.w = float(location["w"])
            sorted_poses.append(pose)
        path = Path(
            header=(
                sorted_poses[0].header
                if sorted_poses
                else rospy.Header(stamp=rospy.Time.now(), frame_id=LOCATIONS_FRAME)
            ),
            poses=sorted_poses,
        )

        self.last_route_pub.publish(path)

        return GetRouteForObjectResponse(path=path,names=names)

    def go_to_location(self, req: GoToLocationRequest) -> GoToLocationResponse:
        location_name: str = req.location
        for location in self.locations_csv:
            if location["name"] == location_name:
                pose = PoseStamped()
                pose.header.frame_id = LOCATIONS_FRAME
                pose.pose.position.x = float(location["x"])
                pose.pose.position.y = float(location["y"])
                pose.pose.orientation.z = float(location["theta"])
                pose.pose.orientation.w = float(location["w"])
                response: GoToResponse = self.go_to_service(GoToRequest(goal=pose))
                return GoToLocationResponse(success=response.success)
        return GoToLocationResponse(success=False)

    def closest_location_to_position(
        self, req: ClosestLocationToPositionRequest
    ) -> ClosestLocationToPositionResponse:
        target_point = self.transform_to_tf(
            req.pose, LOCATIONS_FRAME, rospy.Time.now()
        ).pose.position

        closest_location = None
        closest_distance: float = float("inf")

        for location in self.locations_csv:
            location_point = Pose(
                position=Point(
                    x=float(location["x"]),
                    y=float(location["y"]),
                    z=0.0,
                ),
                orientation=Quaternion(
                    x=0.0,
                    y=0.0,
                    z=float(location["theta"]),
                    w=float(location["w"]),
                ),
            )

            location_vector = pose_to_point(location_point)
            point = SKOPoint([target_point.x, target_point.y, target_point.z])

            distance = location_vector.distance_point(point)
            if distance < closest_distance:
                closest_location = location_point
                closest_distance = float(distance)

        if closest_location is None:
            rospy.logerr("No closest location found")
            return ClosestLocationToPositionResponse(success=False)
        else:
            rospy.loginfo(
                f"Closest location is {closest_location} with distance {closest_distance}"
            )
            return ClosestLocationToPositionResponse(
                pose=PoseStamped(
                    header=rospy.Header(
                        stamp=rospy.Time.now(), frame_id=LOCATIONS_FRAME
                    ),
                    pose=closest_location,
                ),
                success=True,
            )


if __name__ == "__main__":
    trajectory_manager = TrajectoryManager()
    rospy.loginfo("Trajectory manager is running...")
    rospy.spin()
