#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import atan2, sqrt

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, feedback):
        error = self.setpoint - feedback
        self.integral += error
        derivative = error - self.prev_error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output

class TurtleBotController:
    def __init__(self):
        rospy.init_node('turtlebot_controller')
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
        self.pid_controller = PIDController(Kp=1.0, Ki=0.0, Kd=0.0, setpoint=0.0)

    def odometry_callback(self, odom):
        # Extract the position from odometry message
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y

        # Calculate angle to the setpoint
        angle_to_setpoint = atan2(self.setpoint_y - y, self.setpoint_x - x)

        # Calculate linear velocity using PID controller
        linear_velocity = self.pid_controller.update(sqrt((x - self.setpoint_x)**2 + (y - self.setpoint_y)**2))

        # Create Twist message and publish it
        twist_msg = Twist()
        twist_msg.linear.x = linear_velocity
        twist_msg.angular.z = angle_to_setpoint
        self.velocity_publisher.publish(twist_msg)

    def set_setpoint(self, x, y):
        self.setpoint_x = x
        self.setpoint_y = y

if __name__ == '__main__':
    try:
        controller = TurtleBotController()
        # Set the setpoint
        controller.set_setpoint(3.0, 3.0)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
