#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from math import atan2, sqrt
import matplotlib.pyplot as plt
import time

# Global variables for current position
x = 0.0
y = 0.0
theta = 0.0

# PID controller parameters for angle and distance
kp_angle = 0.3
ki_angle = 0.05
kd_angle = 0.0

kp_distance = 0.1
ki_distance = 0.01
kd_distance = 0.0

# Initialize PID controllers
pid_angle = PID(kp_angle, ki_angle, kd_angle)
pid_distance = PID(kp_distance, ki_distance, kd_distance)

# Callback function to update current position
def newOdom(msg):
    global x, y, theta

    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    rot_q = msg.pose.pose.orientation
    (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

# Initialize ROS node
rospy.init_node("pid_controller")

# Subscribe to odometry topic to get current position
sub = rospy.Subscriber("/odom", Odometry, newOdom)

# Publisher for velocity commands
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

# Initialize Twist message for velocity commands
speed = Twist()

# Rate for the loop
r = rospy.Rate(4)

# Initialize lists to store errors for plotting
x_error = []
y_error = []
time_points = []

# Goal position
goal_x = 3.0
goal_y = 3.0

# Time at start
start_time = time.time()

# Main loop
while not rospy.is_shutdown():
    # Calculate errors
    inc_x = goal_x - x
    inc_y = goal_y - y
    angle_to_goal = atan2(inc_y, inc_x)
    distance_to_goal = sqrt(inc_x * inc_x + inc_y * inc_y)

    # Calculate velocity commands using PID controllers
    speed.angular.z = pid_angle(angle_to_goal - theta)
    speed.linear.x = pid_distance(distance_to_goal)

    # Publish velocity commands
    pub.publish(speed)

    # Store errors and time for plotting
    x_error.append(inc_x)
    y_error.append(inc_y)
    time_points.append(time.time() - start_time)

    # Break if robot reaches the goal position
    if distance_to_goal < 0.1:
        break

    # Sleep to maintain the desired rate
    r.sleep()



