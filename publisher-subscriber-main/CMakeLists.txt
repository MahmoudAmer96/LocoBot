cmake_minimum_required(VERSION 3.0.2)
project(beginner_tutorials)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES beginner_tutorials
CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/talker.py scripts/listener.py
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

