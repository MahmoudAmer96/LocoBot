#!/usr/bin/env python

import math
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # to work with numerical data efficiently

def getGoals():
    fs = 10.  # sample rate
    f = 1.  # the frequency of the signal

    x = np.arange(fs + 1)  # the points on the x axis for plotting

    # compute the value (amplitude) of the sin wave at the for each sample
    y = 2. * np.sin(2 * np.pi * f * (x / fs))
    x = x / 2.

    return np.vstack((x, y)).T

class RobotControl():
    def __init__(self):
        rospy.init_node('robot_control_node', anonymous=True)
        self.vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odometryCb)
        self.cmd = Twist()
        self.ctrl_c = False
        self.rate = rospy.Rate(10)  # What is this?
        self.rate.sleep()
        self.distance_tolerance = 0.1
        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        # works better than the rospy.is_shutdown()
        self.ctrl_c = True

    def odometryCb(self, msg):
        # Do not change this. Read in quaternions and transform this to euclidean space.
        x = msg.pose.pose.position.x  # read positions
        y = msg.pose.pose.position.y
        q0 = msg.pose.pose.orientation.w
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        theta = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        self.pose = np.array([x, y, theta])

    def euclidean_distance(self):
        """
        Method to compute distance from current position to the goal
        @returns euclidean distance from current point to goal
        """
        current_pose = self.pose[:2]
        goal_pose = self.goal[:2]
        return np.linalg.norm(goal_pose - current_pose)

    def stop_robot(self):
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        self.publish_once_in_cmd_vel()

    def publish_once_in_cmd_vel(self):
        """
        This is because publishing in topics sometimes fails the first time you publish.
        In continuous publishing systems, there is no big deal but in systems that publish only
        once, it IS very important.
        """
        while not self.ctrl_c:
            connections = self.vel_publisher.get_num_connections()
            if connections > 0:
                self.vel_publisher.publish(self.cmd)
                # rospy.loginfo("Cmd Published")
                break
            else:
                self.rate.sleep()

    def moveGoal_pid(self):
        K_Pv = .3
        K_Pw = 1.0
        K_I = 0.01
        K_D = 0.1

        e_sum = 0
        e_prev = self.euclidean_distance()
        dt = 0.1  # time between updates

        while self.euclidean_distance() > self.distance_tolerance:
            current_pose = self.pose
            desired_pose = self.goal
            e = self.euclidean_distance()
            dedt = (e - e_prev) / dt
            e_sum += e * dt
            e_prev = e

            linear_velocity = K_Pv * e + K_I * e_sum + K_D * dedt

            theta_g = math.atan2(desired_pose[1] - current_pose[1], desired_pose[0] - current_pose[0])
            w = theta_g - current_pose[2]
            angular_velocity = K_Pw * w

            self.cmd.linear.x = linear_velocity
            self.cmd.angular.z = angular_velocity

            self.publish_once_in_cmd_vel()
            self.rate.sleep()

        self.stop_robot()

    def travel(self, goals):
        self.gather_traveled = []
        for goal in goals:
            self.start = (self.pose[0], self.pose[1])
            self.goal = goal
            self.moveGoal_pid()
            message = "Position " + str(self.goal) + " has been achieved."
            rospy.loginfo(message)

if __name__ == '__main__':
    robotcontrol_object = RobotControl()
    try:
        # get goals, move to them and plot the traj!
        goals = getGoals()
        robotcontrol_object.travel(goals)
        plt.stem(goals[:, 0], goals[:, 1], 'r', )
        plt.plot(goals[:, 0], goals[:, 1])

        plot_trajectory = np.array(robotcontrol_object.gather_traveled)
        plt.plot(plot_trajectory[:, 0], plot_trajectory[:, 1])
        plt.show()

    except rospy.ROSInterruptException:
        pass
