#! /usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt # For ploting
import numpy as np # to work with numerical data efficiently

def getGoals(): # Use sine wave to simulate how the goal moves
    fs = 10.  # sample rate
    f = 1.  # the frequency of the signal

    x = np.arange(fs + 1)  # the points on the x axis for plotting

    # compute the value (amplitude) of the sin wave at the for each sample
    y = 2. * np.sin(2 * np.pi * f * (x / fs))
    x = x / 2.

    return np.vstack((x,y)).T

class RobotControl():

    def __init__(self):
        rospy.init_node('robot_control_node', anonymous=True)
        self.vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odometryCb)
        self.cmd = Twist()
        self.ctrl_c = False
        self.NODE_RATE = 10
        self.rate = rospy.Rate(self.NODE_RATE) # Loop frequency of the node (publishing and reading)
        self.rate.sleep()

        # TO-DO Modify the value of these 3 constants after testing on robot or reading from somewhere.. 
        self.LINEAR_VELOCITY = 0.3
        self.ANGULAR_VELOCITY = 0.5
        self.distance_tolerance =self.LINEAR_VELOCITY / 5
        rospy.on_shutdown(self.shutdownhook)
        
    def shutdownhook(self):
        # works better than the rospy.is_shutdown()
        self.ctrl_c = True
        self.stop_robot()

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
        current_position = self.pose[:2] - self.init_pose[:2]
        goal_position = self.goal[:2]
        return np.linalg.norm(current_position - goal_position)
    

    def goal_orientation(self):
        """
        Method to compute (smallest) turning angle from current orientation to facing the goal
        @returns    smallest angle from current orientation to the goal
        """
        theta_r = self.pose[2]
        current_position = self.pose[:2] - self.init_pose[:2]
        theta_g = math.atan2(self.goal[1] - current_position[1], self.goal[0] - current_position[0]) # (y_g -y_r, x_g - x_r)
        return  theta_g - theta_r


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
        e_prev = self.euclidean_distance()
        w_prev = self.goal_orientation() 
        dt = 1./self.NODE_RATE # TIme between 2 updates
        e_sum = 0.
        w_sum = 0.

        Kp_linear = 0.3
        Ki_linear = 0.1
        Kd_linear = 0.1

        Kp_angular = 0.8 #should be large enough, otherwise it turns too slow reletive to its movement, leading to more unnecessary bypass
        # Ki_angular = 0.1
        # Kd_angular = 0.2

        while (self.euclidean_distance() > self.distance_tolerance):
            e = self.euclidean_distance() # get euclidean distance = "error"
            dedt = (e - e_prev) / dt #rate that the error changes 
            e_sum += e * dt  #accumulated error
            e_prev = e
            
            # Set params of the cmd message
            # Linear velocity
            linear_velocity = Kp_linear * e + Ki_linear * e_sum + Kd_linear * dedt

            self.cmd.linear.x = min(linear_velocity, self.LINEAR_VELOCITY)  
            self.cmd.linear.y = 0
            self.cmd.linear.z = 0


            # Angular velocity
            self.cmd.angular.x = 0
            self.cmd.angular.y = 0

            w = self.goal_orientation() # error of angle (orientation)
            # dwdt = (w-w_prev)/dt
            # w_sum += w * dt
            # w_prev = w

            # angular_velocity = Kp_angular * w + Ki_angular * w_sum + Kd_angular * dwdt
            self.cmd.angular.z = Kp_angular * w

            self.vel_publisher.publish(self.cmd)
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
            self.tmp = self.pose - self.init_pose # get current pose relative to init_pose
            self.moveGoal_pid()
            message = "Position " + str(self.goal) + " has been achieved."
            rospy.loginfo(message)

if __name__ == '__main__':
    robotcontrol_object = RobotControl()
    try:
        # get goals, move to them and plot the traj!
        # goals: Positions of goal are different because of its movement
        goals = getGoals()
        robotcontrol_object.travel(goals)
        plt.stem(goals[:, 0], goals[:, 1], 'r', )
        plt.plot(goals[:, 0], goals[:, 1])

        plot_trajectory = np.array(robotcontrol_object.gather_traveled)
        plt.plot(plot_trajectory[:, 0], plot_trajectory[:, 1])
        plt.show()

    except rospy.ROSInterruptException:
        pass
