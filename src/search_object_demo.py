#!/usr/bin/env python3

import yaml
from typing import List, Optional
import rospy
from enum import Enum
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker
from stretch_manipulation.srv import (
    GetCurrentLocation,
    GetCurrentLocationResponse,
    GetDetectedObjectNames,
    GetDetectedObjectNamesResponse,
    GoTo,
    GetRouteForObject,
    GetRouteForObjectResponse,
    GetObjectPosition,
    GetObjectPositionResponse,
    ClosestLocationToPosition,
    ClosestLocationToPositionResponse,
    LookAt,
    GraspPosition,
    GraspPositionResponse,
    SearchObject,
    SearchObjectRequest,
    SearchObjectResponse,
    SetSearchObjectState,
    SetSearchObjectStateRequest,
    SetSearchObjectStateResponse,
    Speak,
)

NODE_NAME = "search_object_demo"
SEARCH_SERVICE = f"/{NODE_NAME}/search"
GRASP_FAILURE_SERVICE = f"/{NODE_NAME}/grasp_failure"
PREAMBLE_SERVICE = f"/{NODE_NAME}/preamble"
TARGET_OBJECT_VISUALISATION_TOPIC = "/target_object_marker"

STOW_THE_ROBOT_SERVICE = "/stow_the_robot"

SPANISH_CONFIG = "config/speech_spanish.yaml"
CATALAN_CONFIG = "config/speech_catalan.yaml"
ENGLISH_CONFIG = "config/speech_english.yaml"

TRAJECTORY_MANAGER_NODE_NAME = "trajectory_manager"
GET_CURRENT_LOCATION_SERVICE = f"/{TRAJECTORY_MANAGER_NODE_NAME}/get_current_location"
GET_ROUTE_FOR_OBJECT_SERVICE = f"/{TRAJECTORY_MANAGER_NODE_NAME}/get_route_for_object"
CLOSEST_LOCATION_TO_POSITION_SERVICE = (
    f"/{TRAJECTORY_MANAGER_NODE_NAME}/closest_location_to_position"
)

OBJECT_MANAGER_NODE_NAME = "object_manager"
GET_OBJECT_POSITION_SERVICE = f"/{OBJECT_MANAGER_NODE_NAME}/get_object_position"
FORCE_OBJECT_POSITION_SERVICE = f"/{OBJECT_MANAGER_NODE_NAME}/force_object_position"
CLEAR_ALL_OBJECTS_SERVICE = f"/{OBJECT_MANAGER_NODE_NAME}/clear_all_objects"
GET_DETECTED_OBJECT_NAMES_SERVICE = f"/{OBJECT_MANAGER_NODE_NAME}/get_detected_object_names"

ROBOT_MANAGER_NODE_NAME = "robot_manager"
GO_TO_SERVICE = f"/{ROBOT_MANAGER_NODE_NAME}/go_to_service"
HEAD_SCAN_SERVICE = f"/{ROBOT_MANAGER_NODE_NAME}/head_scan_service"
LOOK_AT_SERVICE = f"/{ROBOT_MANAGER_NODE_NAME}/look_at_service"
MOVE_TO_GRASPING_POSITION_SERVICE = (
    f"/{ROBOT_MANAGER_NODE_NAME}/move_to_grasping_position_service"
)
GRASP_POSITION_SERVICE = f"/{ROBOT_MANAGER_NODE_NAME}/grasp_position_service"
RETRIEVE_THE_ROBOT_SERVICE = f"/{ROBOT_MANAGER_NODE_NAME}/retrieve_the_robot_service"
LOOK_AT_GRIPPER_SERVICE = f"/{ROBOT_MANAGER_NODE_NAME}/look_at_gripper_service"
HANDOVER_OBJECT_SERVICE = f"/{ROBOT_MANAGER_NODE_NAME}/handover_object_service"

SPEECH_MANAGER_NODE_NAME = "speech_manager"
SPEAK_SERVICE = f"/{SPEECH_MANAGER_NODE_NAME}/speak"
SPEECH_SENTENCES_FILE = SPANISH_CONFIG

CONFIRM_GRIP_SLEEP_TIME = 5
MAX_GRASP_ATTEMPTS = 2

class SearchObjectDemo:

    class SearchObjectState(Enum):
        INIT = 0
        ROUTING = 1
        SEARCHING = 2
        GRASPING = 3
        RETURN_TO_USER = 4

    state_names = {
        "INIT":SearchObjectState.INIT,
        "ROUTING":SearchObjectState.ROUTING,
        "SEARCHING":SearchObjectState.SEARCHING,
        "GRASPING":SearchObjectState.GRASPING,
        "RETURN_TO_USER":SearchObjectState.RETURN_TO_USER,
    }

    def __init__(self):
        # ROS
        rospy.init_node(NODE_NAME)

        # Variables
        self.state = self.SearchObjectState.INIT
        self.target_object = None
        self.user_location: Optional[PoseStamped] = None
        self.route: Optional[List[PoseStamped]] = None
        self.route_names = None
        self.object_position: Optional[PoseStamped] = None
        self.search_successes = []
        self.search_locations = []
        self.last_location = None
        self.grasp_successful = True
        self.num_grasp_attempts = 0
        with open(SPEECH_SENTENCES_FILE, "r", encoding="utf-8") as file:
            self.dialogue_phrases = yaml.safe_load(file)

        # Services: Driver
        self.stow_the_robot = rospy.ServiceProxy(STOW_THE_ROBOT_SERVICE, Trigger)

        # Services: Route Manager
        self.get_current_location = rospy.ServiceProxy(
            GET_CURRENT_LOCATION_SERVICE, GetCurrentLocation
        )
        self.closest_location_to_position = rospy.ServiceProxy(
            CLOSEST_LOCATION_TO_POSITION_SERVICE, ClosestLocationToPosition
        )
        self.get_route_for_object = rospy.ServiceProxy(
            GET_ROUTE_FOR_OBJECT_SERVICE, GetRouteForObject
        )

        # Services: Object Manager
        self.object_detection_service = rospy.ServiceProxy(
            GET_OBJECT_POSITION_SERVICE, service_class=GetObjectPosition
        )
        self.force_object_detection_service = rospy.ServiceProxy(
            FORCE_OBJECT_POSITION_SERVICE, service_class=GetObjectPosition
        )
        self.clear_all_objects = rospy.ServiceProxy(CLEAR_ALL_OBJECTS_SERVICE, Trigger)
        self.get_detected_object_names = rospy.ServiceProxy(GET_DETECTED_OBJECT_NAMES_SERVICE, GetDetectedObjectNames)

        # Services: Picker / Mover
        self.go_to_service = rospy.ServiceProxy(GO_TO_SERVICE, GoTo)
        self.head_scan_service = rospy.ServiceProxy(HEAD_SCAN_SERVICE, Trigger)
        self.look_at_service = rospy.ServiceProxy(LOOK_AT_SERVICE, LookAt)
        self.move_to_grasping_position_service = rospy.ServiceProxy(
            MOVE_TO_GRASPING_POSITION_SERVICE, GraspPosition
        )
        self.grasp_position_service = rospy.ServiceProxy(
            GRASP_POSITION_SERVICE, GraspPosition
        )
        self.look_at_gripper_service = rospy.ServiceProxy(
            LOOK_AT_GRIPPER_SERVICE, Trigger
        )
        self.retrieve_the_robot_service = rospy.ServiceProxy(
            RETRIEVE_THE_ROBOT_SERVICE, Trigger
        )
        self.handover_service = rospy.ServiceProxy(HANDOVER_OBJECT_SERVICE, Trigger)

        # Services: Speaker
        self.speak_service = rospy.ServiceProxy(SPEAK_SERVICE, Speak)

        # Services
        rospy.Service(SEARCH_SERVICE, SearchObject, self.search_callback)
        rospy.Service(GRASP_FAILURE_SERVICE, Trigger, self.grasp_failure_callback)
        rospy.Service(PREAMBLE_SERVICE,SearchObject,self.preamble_callback)
        
        # Publishers
        self.target_object_marker_pub = rospy.Publisher(TARGET_OBJECT_VISUALISATION_TOPIC, Marker, queue_size = 2)

    def say(self, sentence: str):
        try:
            self.speak_service(
                sentence=sentence, language=self.dialogue_phrases["language"]
            )
        except Exception as e:
            rospy.logerr(f"Error while speaking: {e}")

    def reset(self):
        self.state = self.SearchObjectState.INIT
        self.user_location = None
        self.route = None
        self.original_route = None
        self.route_names = None
        self.object_position = None
        self.target_object = None
        self.search_locations = []
        self.search_successes = []
        self.last_location = None
        self.num_grasp_attempts = 0
        self.grasp_failed = False
        self.clear_all_objects()

    def init(self):
        self.stow_the_robot()
        response: GetCurrentLocationResponse = self.get_current_location()
        self.user_location = PoseStamped(
            header=response.pose.header,
            pose=response.pose.pose.pose,
        )

    def publish_target_object_marker(self, target_object, pose: PoseStamped):
        marker = Marker()

        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = 0
        marker.text = target_object
        #marker.lifetime = rospy.Duration(5)

        # Set the scale of the marker
        scale = 0.1
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose = pose.pose
        
        # Publish
        self.target_object_marker_pub.publish(marker)

    def routing(self, object_class: str):
        response: GetRouteForObjectResponse = self.get_route_for_object(object_class)
        self.route = response.path.poses
        self.route_names = response.names
        self.original_route = self.route_names.copy()
        

    def searching(self, object_class: str):
        assert self.route  # Route should not be empty

        # Go to next location
        '''
        self.say(self.dialogue_phrases["next location"])
        '''
        

        next_location = self.route.pop(0)
        self.search_locations.append(self.route_names.pop(0))

        if self.trial == 2 or self.trial == 3:
            # Complete description of next action
            if self.search_locations[-1] == self.original_route[0]:
                # First location
                self.say(
                    self.dialogue_phrases["next location detailed"].replace(
                        "{location}", self.dialogue_phrases["location"][self.search_locations[-1]]
                    )
                )
            else:
                # Subsequent location
                self.say(
                    self.dialogue_phrases["next location detailed ongoing"].replace(
                        "{location}", self.dialogue_phrases["location"][self.search_locations[-1]]
                    ).replace(
                        "{object_class}", self.dialogue_phrases["object_class"][self.target_object]
                    )
                )
        
        go_to_response = self.go_to_service(next_location)
        
        self.last_location = next_location
        
        self.search_successes.append(go_to_response.success)
        if not go_to_response.success:
            self.object_position = None
            return

        # Search object
        self.head_scan_service()
        response: GetObjectPositionResponse = self.object_detection_service(
            object_class
        )

        # Set object
        if response.success:
            self.object_position = response.pose
        else:
            self.object_position = None

    def grasping(self, object_class: str):
        # Look at arm
        self.look_at_service(self.object_position)

        # Move to object
        closest_location: ClosestLocationToPositionResponse = (
            self.closest_location_to_position(self.object_position)
        )
        self.go_to_service(closest_location.pose)
        self.head_scan_service()
        response: GetObjectPositionResponse = self.object_detection_service(
            object_class
        )
        self.object_position = response.pose if response.success else None
        self.move_to_grasping_position_service(self.object_position)

        # Grasp object
        response2: GetObjectPositionResponse = self.force_object_detection_service(
            object_class
        )
        object_position = response2.pose if response2.success else None
        response3: GraspPositionResponse = self.grasp_position_service(object_position)
        return response3.success

    def return_to_user(self):
        if self.trial == 2 or self.trial == 3:
            self.say(self.dialogue_phrases["returning to user"])
        elif self.trial not in [1,2,3,4,5]:
            self.say(self.dialogue_phrases["returning to user"])
        self.go_to_service(self.user_location)
        
    def confirm_grasp_vision(self, object_class):
        self.look_at_gripper_service()
        rospy.loginfo("Looking at gripper")
        rospy.sleep(CONFIRM_GRIP_SLEEP_TIME)
        response: GetDetectedObjectNamesResponse = self.get_detected_object_names()
        rospy.loginfo(f"Objects: {response.names}")
        if object_class in response.names:
            return True
        return False
    
    def confirm_grasp_manual(self):
        self.grasp_successful = True
        self.look_at_gripper_service()
        rospy.sleep(CONFIRM_GRIP_SLEEP_TIME)
        return self.grasp_successful
        
    def grasp_failure_callback(self, *_) -> TriggerResponse:
        self.grasp_successful = False
        return TriggerResponse(success=True)
    
    def move_to_retry_grasping(self, target_object):
        if self.trial == 2 or self.trial == 3:
            self.say(
                self.dialogue_phrases["retry grasp"].replace(
                    "{object_class}",
                    self.dialogue_phrases["object_class"][target_object],
                )
            )
        elif self.trial not in [1,2,3,4,5]:
            self.say(
                self.dialogue_phrases["retry grasp"].replace(
                    "{object_class}",
                    self.dialogue_phrases["object_class"][target_object],
                )
            )
        self.stow_the_robot()
        self.go_to_service(self.last_location)
        
    def visualise(self,target_object):
        if self.object_position is not None:
            self.publish_target_object_marker(target_object,self.object_position)
            rospy.loginfo(f"Target object pose: {self.object_position}")

    def preamble_callback(self, request: SearchObjectRequest) -> SearchObjectResponse:
        self.reset()
        if request.trial == 1:
            # Trial 1, medium-length narration of the plan
            self.target_object = request.object_class
            self.routing(self.target_object)
            next_location = self.route_names[0]
            
            self.say(
                self.dialogue_phrases["initial_plan_medium"].replace(
                    "{object_class}", self.dialogue_phrases["object_class"][self.target_object],
                ).replace(
                    "{location}", self.dialogue_phrases["location"][next_location],
                )
            )
            return SearchObjectResponse(success=True)
        elif request.trial == 2:
            # Trial 2, listing possible failures
            self.target_object = request.object_class

            self.say(
                self.dialogue_phrases["initial_expectations_failure"].replace(
                    "{object_class}", self.dialogue_phrases["object_class"][self.target_object]
                )
            )
            return SearchObjectResponse(success=True)
        elif request.trial == 5:
            # Trial 5, medium-length narration of the plan
            self.target_object = request.object_class
            self.routing(self.target_object)
            next_location = self.route_names[0]

            self.say(
                self.dialogue_phrases["initial_plan_medium"].replace(
                    "{object_class}", self.dialogue_phrases["object_class"][self.target_object]
                ).replace(
                    "{location}", self.dialogue_phrases["location"][next_location],
                )
            )
            return SearchObjectResponse(success=True)
        else:
            # Other trials which only have confirmation
            self.target_object = request.object_class

            self.say(
                self.dialogue_phrases["task description"].replace(
                    "{object_class}", self.dialogue_phrases["object_class"][self.target_object]
                )
            )
            return SearchObjectResponse(success=True)
        
    def trajectory_summary(self):
        text = self.dialogue_phrases["trajectory summary start"]
        for location,success in zip(self.search_locations,self.search_successes):
            text += " " + self.dialogue_phrases["location"][location]

            if not success:
                text += " " + self.dialogue_phrases["location not accesible clause"]

            if location == self.search_locations[-1]:
                pass
            elif location == self.search_locations[-2]:
                text += ", " + self.dialogue_phrases["and"]
            else:
                text += ", "

        if self.object_position is not None:
            text += self.dialogue_phrases["location object found clause"].replace(
                    "{object_class}", self.dialogue_phrases["object_class"][self.target_object]
                )
            text += ". "
            
            if self.grasp_failed:
                text += self.dialogue_phrases["grasp failed summary"].replace(
                    "{object_class}", self.dialogue_phrases["object_class"][self.target_object]
                )
        else:
            text += ". "
            text += self.dialogue_phrases["object not found"].replace(
                    "{object_class}", self.dialogue_phrases["object_class"][self.target_object]
                )
        
        self.say(text)
            





    def search_callback(self, request: SearchObjectRequest) -> SearchObjectResponse:
        self.reset()
        self.target_object = request.object_class
        self.trial = request.trial
        success = self.search_object_loop(request.object_class,request.trial)
        return SearchObjectResponse(success=success)
    
    def search_object_loop(self, target_object: str, trial: int) -> bool:
        while not rospy.is_shutdown():
            self.visualise(target_object)
            
            # Initialize
            if self.state == self.SearchObjectState.INIT:
                if target_object not in self.dialogue_phrases["object_class"]:
                    self.say(
                        self.dialogue_phrases["object unknown"].replace(
                            "{object_class}", target_object
                        )
                    )
                    return False
                
                if trial not in [1,2,3,4,5]:
                    self.say(
                        self.dialogue_phrases["task description"].replace(
                            "{object_class}",
                            self.dialogue_phrases["object_class"][target_object],
                        )
                    )

                rospy.loginfo("State: INIT")
                self.init()
                self.state = self.SearchObjectState.ROUTING

            # Routing
            elif self.state == self.SearchObjectState.ROUTING:
                rospy.loginfo("State: ROUTING")
                self.routing(target_object)
                self.state = self.SearchObjectState.SEARCHING

            # Search object
            elif self.state == self.SearchObjectState.SEARCHING:
                rospy.loginfo("State: SEARCHING")
                self.searching(target_object)
                if self.object_position is not None:
                    self.state = self.SearchObjectState.GRASPING
                    if trial == 2:
                        self.say(
                            self.dialogue_phrases["found object grasp"].replace(
                                "{object_class}", self.dialogue_phrases["object_class"][target_object]
                            )
                        )
                    elif trial not in [1,2,3,4,5]:
                        self.say(
                            self.dialogue_phrases["found object"].replace(
                                "{object_class}", self.dialogue_phrases["object_class"][target_object]
                            )
                        )
                else:
                    # If there are more locations to search
                    if self.route:
                        self.state = self.SearchObjectState.SEARCHING
                    # If there are no more locations to search
                    else:
                        self.state = self.SearchObjectState.RETURN_TO_USER

            # Grasp object
            elif self.state == self.SearchObjectState.GRASPING:
                rospy.loginfo("State: GRASPING")
                grasping_executed = self.grasping(target_object)
                self.num_grasp_attempts += 1
                # Check if grasp was successfull
                grasping_successful = self.confirm_grasp_manual()
                if grasping_executed and grasping_successful:
                    if self.trial == 2 or self.trial == 3:
                        self.say(
                            self.dialogue_phrases["grasp successful"].replace(
                                "{object_class}", self.dialogue_phrases["object_class"][target_object]
                            )
                        )
                    self.retrieve_the_robot_service()
                    self.state = self.SearchObjectState.RETURN_TO_USER
                # Retry grasping
                elif self.num_grasp_attempts < MAX_GRASP_ATTEMPTS and self.trial != 3:
                    self.move_to_retry_grasping(target_object)
                    self.state = self.SearchObjectState.GRASPING
                else:
                    self.grasp_failed = True
                    self.retrieve_the_robot_service()
                    self.state = self.SearchObjectState.RETURN_TO_USER

            # Return to user
            elif self.state == self.SearchObjectState.RETURN_TO_USER:
                rospy.loginfo("State: RETURN_TO_USER")
                self.return_to_user()
                if self.trial == 4 or self.trial == 5:
                    self.trajectory_summary()
                if self.object_position is not None:
                    if grasping_successful:
                        self.say(
                            self.dialogue_phrases["deliver object"].replace(
                                "{object_class}",
                                self.dialogue_phrases["object_class"][target_object],
                            )
                        )
                        self.handover_service()
                    else:
                        # Grasp failed
                        if trial in [1,2,3]:
                            self.say(
                                self.dialogue_phrases["task outcome failure"].replace(
                                    "{object_class}",
                                    self.dialogue_phrases["object_class"][target_object],
                                )
                            )
                        elif trial not in [1,2,3,4,5]:
                            self.say(
                                self.dialogue_phrases["object cannot grasp"].replace(
                                    "{object_class}",
                                    self.dialogue_phrases["object_class"][target_object],
                                )
                            )
                else:
                    if False in self.search_successes:
                        # Failed to navigate to one or more locations
                        if trial == 1:
                            self.say(
                                self.dialogue_phrases["task outcome failure"].replace(
                                    "{object_class}",
                                    self.dialogue_phrases["object_class"][target_object],
                                )
                            )
                        elif trial not in [1,2,3,4,5]:
                            self.say(
                                self.dialogue_phrases["object not found unreachable locations"].replace(
                                    "{object_class}",
                                    self.dialogue_phrases["object_class"][target_object],
                                )
                            )
                    else:
                        # Searched all locations but object not found
                        if trial == 1:
                            self.say(
                                self.dialogue_phrases["task outcome failure"].replace(
                                    "{object_class}",
                                    self.dialogue_phrases["object_class"][target_object],
                                )
                            )
                        elif trial not in [1,2,3,4,5]:
                            self.say(
                                self.dialogue_phrases["object not found"].replace(
                                    "{object_class}",
                                    self.dialogue_phrases["object_class"][target_object],
                                )
                            )
                    return False
                self.stow_the_robot()
                return True
        return False


if __name__ == "__main__":
    search_object = SearchObjectDemo()
    rospy.loginfo("Search Object Demo is ready")
    rospy.spin()
