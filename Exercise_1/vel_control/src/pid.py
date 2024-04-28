#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt # For ploting
import numpy as np # to work with numerical data efficiently
import time

def getGoals():
    fs = 10.  # sample rate
    f = 1.  # the frequency of the signal

    x = np.arange(fs + 1)  # the points on the x axis for plotting

    # compute the value (amplitude) of the sin wave at the for each sample
    y = 2. * np.sin(2 * np.pi * f * (x / fs))
    x = x / 2.

    return np.vstack((x,y)).T

class RobotControl():

    def __init__(self):
        rospy.init_node('robot_control_node', anonymous = True)
        self.vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odometryCb)
        self.cmd = Twist()
        self.ctrl_c = False
        self.rate = rospy.Rate(10) # What is this?
        self.rate.sleep()
        self.LINEAR_VELOCITY = 0.5
        self.ANGULAR_VELOCITY = 0.5
        self.distance_tolerance = 0.5
        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        # works better than the rospy.is_shutdown()
        self.ctrl_c = True

    def odometryCb(self, msg):
        # Do not change this. Read in quaternions and tranfrom this to euclidean space.
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
        @returns 	euclidean distance from current point to goal
        """
        x = self.tmp[0] - self.goal[0]
        y = self.tmp[1] - self.goal[1]
        return math.sqrt((x ** 2) + (y ** 2))

    def stop_robot(self):
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        self.publish_once_in_cmd_vel()

    def publish_once_in_cmd_vel(self):
        """
        This is because publishing in topics sometimes fails the first time you publish.
        In continuos publishing systems there is no big deal but in systems that publish only
        once it IS very important.
        """
        while not self.ctrl_c:
            connections = self.vel_publisher.get_num_connections()
            if connections > 0:
                self.vel_publisher.publish(self.cmd)
                #rospy.loginfo("Cmd Published")
                break
            else:
                self.rate.sleep()

    def moveGoal_pid(self):
        self.tmp = self.pose - self.init_pose
        e_prev = self.euclidean_distance()
        w_prev = self.tmp[2]
        dt = 1 / self.rate # rate
        e_sum = 0.
        w_sum = 0.
        K_Pw = 1.
        K_Pv = 1
        K_I = 1.
        K_D = 1.
        goal_angle = math.atan2((self.goal[0] - self.init_pose[0]), (self.goal[0] - self.init_pose[0]))

        while (self.euclidean_distance() > self.distance_tolerance):
            self.tmp = self.pose - self.init_pose    # get current pose here. Do not forget to subtract by the origin to remove init. translations.
            e = self.euclidean_distance()   # get euclidean distance
            dedt = (e - e_prev) / dt
            e_sum = e_sum + e * dt
            e_prev = e

            w = min((2 * math.pi - (goal_angle - self.tmp[2])), (goal_angle - self.tmp[2]))   # choose the smaller angle
            dwdt = (w - w_prev) / dt
            w_sum = w_sum + w * dt
            w_prev = w

            # Set params of the cmd message
            self.cmd.linear.x = min((K_Pv * e + K_I * e_sum + K_D * dedt), self.LINEAR_VELOCITY)
            self.cmd.linear.y = 0.
            self.cmd.linear.z = 0.
            self.cmd.angular.x = 0.
            self.cmd.angular.y = 0.
            self.cmd.angular.z = min((K_Pw * w + K_I * w_sum + K_D * dwdt), self.ANGULAR_VELOCITY)

            self.vel_publisher.publish()
            self.gather_traveled.append([self.tmp[0], self.tmp[1]])
            self.rate.sleep()

        # set velocity to zero to stop the robot
        self.stop_robot()


    def travel(self, goals):
        self.init_pose = self.pose
        self.gather_traveled = []
        for goal in goals:
            self.start = (self.pose[0],  self.pose[1])
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
