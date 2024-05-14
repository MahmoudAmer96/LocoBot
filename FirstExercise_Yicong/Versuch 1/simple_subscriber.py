#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        
def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("chatter_str", String, callback)
    rospy.Subscriber("chatter_twi", Twist, callback)
    rospy.spin()
   
    if __name__ == '__main__':
       listener()