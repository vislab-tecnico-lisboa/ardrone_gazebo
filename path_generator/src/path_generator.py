#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('path_generator')
import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def talker():
    path_pub = rospy.Publisher('/ardrone/path_pub', Path, queue_size=10)
    pos_pub = rospy.Publisher('/ardrone/pos_pub', PoseStamped, queue_size=10)
    rospy.init_node('path_generator', anonymous=True)
    
    freq = 100;
    round_min = 0.1;
    
    path_msg = Path();
    path_msg.header.stamp = rospy.get_rostime();
    path_msg.poses = [];
    path_msg.header.frame_id = "nav";
    
    radius = 2.0;    
    rate = rospy.Rate(freq) # 100hz

    count = 0;
    while not rospy.is_shutdown():
        count += 1;
        pos_msg = PoseStamped();
        pos_msg.header.stamp = rospy.get_rostime();
        pos_msg.header.frame_id = "nav";
        
        pos_msg.pose.position.x = radius * np.cos((np.pi * 2) / freq * round_min * count);
        pos_msg.pose.position.y = radius * np.sin((np.pi * 2) / freq * round_min * count);
        pos_msg.pose.position.z = 1.5;       
        
        if count > (freq / round_min):
            path_msg.poses.pop(0)
        path_msg.poses.append(pos_msg);    
        
        path_pub.publish(path_msg)
        pos_pub.publish(pos_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
