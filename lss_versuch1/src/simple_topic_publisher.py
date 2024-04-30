#!/usr/bin/python3

import rospy
from std_msgs.msg import String

def publisher():
    rospy.init_node('simple_topic_publisher', anonymous=True)
    pub = rospy.Publisher('/simple_topic', String, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass