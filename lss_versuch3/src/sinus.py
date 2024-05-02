#!/usr/bin/python3
import math
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt # For ploting
import numpy as np # to work with numerical data efficiently

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
        rospy.init_node('robot_control_node', anonymous=True)
        self.vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odometryCb)

        self.cmd = Twist()
        self.ctrl_c = False
        self.rate = rospy.Rate(10)  # maintaining a particular rate for a loop
        self.rate.sleep()  # Q: why?
        self.LINEAR_VELOCITY = 0.4
        self.ANGULAR_VELOCITY = 0.8
        self.distance_tolerance = 0.1

        self.Kp = 0.4
        self.Kd = 0.02
        self.Ki = 0.02
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
        Method to compute distance from real position to the goal e(t) = g(t) − x(t)
        """
        return math.sqrt(math.pow(self.goal[1] - self.tmp[1], 2) + math.pow(self.goal[0] - self.tmp[0], 2))

    def theta_g(self):
        """
        Method to compute radians [-π, π] from real position to the goal math.atan2(y_g - y_r, x_g - x_r)
        """
        return math.atan2(self.goal[1] - self.tmp[1], self.goal[0] - self.tmp[0])
    

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
        w_prev = 0.
        e_sum = 0.
        while self.euclidean_distance() > self.distance_tolerance:
            self.tmp = self.pose - self.init_pose  # get current pose here. Do not forget to subtract by the origin to remove init. translations.
            dt = 0.1
            e = self.euclidean_distance()  # get euclidean distance
            dedt = (e - e_prev) / dt
            e_sum = e_sum + e * dt
            e_prev = e

            u = self.Kp * e + self.Ki * e_sum + self.Kd * dedt  # Kp*e + Kd*dedt + Ki*e_sum
            
            # Set params of the cmd message
            self.cmd.linear.x = min(self.LINEAR_VELOCITY, u)
            self.cmd.linear.y = 0.
            self.cmd.linear.z = 0.
            self.cmd.angular.x = 0.
            self.cmd.angular.y = 0.

            w = self.theta_g() - self.pose[2]
            #if w > 1 * np.pi:
            #    w = w - 2 * np.pi
            #elif w < -1 * np.pi:
            #    w = w + 2 * np.pi
            w = self.ANGULAR_VELOCITY * w
            dwdt = w / dt
            self.cmd.angular.z = w + self.Kd * dwdt

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
