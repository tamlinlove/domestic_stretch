###########################
## Package Configuration ##
###########################

cmake_minimum_required(VERSION 3.0.2)
project(stretch_manipulation)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    message_generation
    actionlib_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################

add_service_files(
  FILES
  AddCurrentLocation.srv
  GetCurrentLocation.srv
  GetDetectedObjectNames.srv
  GetRouteForObject.srv
  ClosestLocationToPosition.srv
  GetObjectPosition.srv
  GoTo.srv
  GoToLocation.srv
  LookAt.srv
  GraspPosition.srv
  SearchObject.srv
  SetSearchObjectState.srv
  Speak.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
  actionlib_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS actionlib_msgs
)

###########
## Build ##
###########

include_directories(
    ${catkin_INCLUDE_DIRS}
)