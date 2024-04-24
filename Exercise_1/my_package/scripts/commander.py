#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def commander():
    pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size = 10)
    rospy.init_node('commander', anonymous = False)
    rate = rospy.Rate(1)    # 1 hz

    while not rospy.is_shutdown():
        msg = Twist()
        msg.linear.x = 2.0
        msg.angular.z = 1.8
        rospy.loginfo(msg)
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        commander()
    except rospy.ROSInterruptException:
        pass
