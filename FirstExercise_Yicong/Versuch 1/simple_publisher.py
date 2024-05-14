#! /usr/bin/env python3

# Import the Python library for ROS
import rospy 
# Import the String and Twist message from the packages of std_msgs and geometry_msgs
from std_msgs.msg import String  
from geometry_msgs.msg import Twist


def talker():
        pub_str = rospy.Publisher('chatter_str', String, queue_size=10)
        pub_twist = rospy.Publisher('chatter_twi', Twist, queue_size=10)

        rospy.init_node('talker', anonymous=True)

        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            #Publish String msg
           hello_str = "hello world %s" % rospy.get_time()
           rospy.loginfo(hello_str)
           pub_str.publish(hello_str)

            #Publish Twist msg
           twist_msg = Twist()
           twist_msg.linear.x = 0.2
           twist_msg.angular.z = 0.2
           pub_twist.publish(twist_msg)

           rate.sleep()

if __name__ == '__main__':
       try:
           talker()
       except rospy.ROSInterruptException:
           pass

