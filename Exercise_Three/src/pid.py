#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math

#Mit hilfe von ChatGPT erstellt. Es wurde geprüft und es erfüllt seinen zweck
class PIDController:
    def __init__(self):
        rospy.init_node('pid_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10) #turtle1/cmd_vel ist die typische instanz die von turtlesim erstellt wird.
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.update_pose) #gleiche gild für turle1/pose

        self.pose = Pose()
        self.rate = rospy.Rate(10)  # 10hz rospy.Rate sind die Anfragen pro Sekunde (gemessen in Herz)

        # PID constants for linear velocity
        self.Kp_linear = 0.5 # Zuwachs von 0.5 Einheiten
        self.Ki_linear = 0.0
        self.Kd_linear = 0.0

        # PID constants for angular velocity
        self.Kp_angular = 1.0 #Zuwachs von 1.0 Einheiten vom "Winkel"
        self.Ki_angular = 0.0
        self.Kd_angular = 0.0

        # Previous error terms for linear and angular velocity
        self.prev_linear_error = 0.0
        self.prev_angular_error = 0.0

        # Sum of error terms for integral control
        self.sum_linear_error = 0.0
        self.sum_angular_error = 0.0

        # Distance tolerance
        self.distance_tolerance = 0.1 #Eigen erstellt

    def update_pose(self, data): #Selbsterklärend. Momentane position der Turtle wird aktualisiert
        self.pose = data

    def euclidean_distance(self, goal_pose): #Euclidean distanz wird berechnet. Hierbei handelt es sich um die differenz der aktuellen Position und des Ziels
        return math.sqrt((goal_pose.x - self.pose.x) ** 2 + (goal_pose.y - self.pose.y) ** 2)

    def linear_velocity_control(self, goal_pose):
        error = self.euclidean_distance(goal_pose)

        # Proportional term
        P = self.Kp_linear * error

        # Integral term
        self.sum_linear_error += error
        I = self.Ki_linear * self.sum_linear_error

        # Derivative term
        D = self.Kd_linear * (error - self.prev_linear_error)

        # PID control output
        linear_velocity = P + I + D

        # Save error for next iteration
        self.prev_linear_error = error

        return linear_velocity

    def angular_velocity_control(self, goal_pose):
        error = math.atan2(goal_pose.y - self.pose.y, goal_pose.x - self.pose.x) - self.pose.theta #Aus erhaltener PDF

        # Proportional term
        P = self.Kp_angular * error

        # Integral term
        self.sum_angular_error += error
        I = self.Ki_angular * self.sum_angular_error

        # Derivative term
        D = self.Kd_angular * (error - self.prev_angular_error)

        # PID control output
        angular_velocity = P + I + D

        # Save error for next iteration
        self.prev_angular_error = error

        return angular_velocity

    def move_towards_goal(self, goal_pose):
        while self.euclidean_distance(goal_pose) > self.distance_tolerance: #Solange ziel nicht erreicht
            # Calculate linear and angular velocities
            linear_velocity = self.linear_velocity_control(goal_pose)
            angular_velocity = self.angular_velocity_control(goal_pose)

            # Publish velocities
            twist_msg = Twist()
            twist_msg.linear.x = linear_velocity
            twist_msg.angular.z = angular_velocity
            self.velocity_publisher.publish(twist_msg)

            # Sleep to control loop rate
            self.rate.sleep()

        # Stop robot when goal is reached
        self.stop_robot()

    def stop_robot(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.velocity_publisher.publish(twist_msg)


if __name__ == '__main__':
    try:
        pid_controller = PIDController()

        # Set goal position - verbesserungwürdig aber soll für erste zwecke ausreichen!
        goal_pose = Pose()
        goal_pose.x = 8.0
        goal_pose.y = 9.0

        # Move towards goal
        pid_controller.move_towards_goal(goal_pose)

    except rospy.ROSInterruptException:
        pass
