#!/usr/bin/env python
from __future__ import print_function, division

import roslib
roslib.load_manifest('actor_critic_agent')
import sys
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ContactsState
from ardrone_autonomy.msg import Navdata
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
import time
import os

class actor_critic:

  def __init__(self):
      
    self.aruco_limit = 4.0
    self.bumper_msg = None
    self.navdata_msg = None
    self.aruco_msg = None    
    self.count = 0;
    self.bridge = CvBridge()   
    
    self.image_sub = rospy.Subscriber("/ardrone/front/image_raw",
                                      Image, self.callback_image)
    self.imu_sub = rospy.Subscriber("/ardrone/imu",
                                    Imu, self.callback_imu)
    self.bumper_sub = rospy.Subscriber("/ardrone/bumper",
                                       ContactsState, self.callback_bumper)
    self.navdata_sub = rospy.Subscriber("/ardrone/navdata",
                                       Navdata, self.callback_navdata)
    self.aruco_sub = rospy.Subscriber("/ardrone/aruco/pose",
                                       Twist, self.callback_aruco)
                                      
    rospy.loginfo("Subscribers initialized")
    
    self.model_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
    self.reset_pub = rospy.Publisher('/ardrone/reset', Empty, queue_size=10)
    self.takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=10)
    self.land_pub = rospy.Publisher('/ardrone/land', Empty, queue_size=10)
    
    rospy.loginfo("Publishers initialized")

  def callback_image(self,data):
    time_stamp = data.header.stamp.secs + \
                 data.header.stamp.nsecs / 1000000000
                 
    aruco_dist = 0.0
    
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      rospy.logerror(e)
      
    cv_image_resized = cv2.resize(cv_image, (160, 80))

    file_name = "image" + \
                str(self.count) + ".png"

    rospy.logdebug(file_name + " " + str(time_stamp) + "\n")
    
    if self.aruco_msg != None:
        aruco_dist = np.sqrt(self.aruco_msg.linear.x**2 + \
                             self.aruco_msg.linear.y**2 + \
                             self.aruco_msg.linear.z**2)
        if aruco_dist == 0.0 or aruco_dist > self.aruco_limit:
            aruco_dist = self.aruco_limit
        aruco_dist = 1.0 - (aruco_dist / self.aruco_limit)
        rospy.loginfo(str(aruco_dist))
    
    if (self.bumper_msg != None  and \
       self.navdata_msg != None and \
       self.bumper_msg.states != [] and \
       self.navdata_msg.state == 3) or \
       (aruco_dist > 0.9):
        model_state_msg = ModelState()
        empty_msg = Empty()
        
        model_state_msg.model_name = "quadrotor"
        model_state_msg.pose.position.x = 0.0
        model_state_msg.pose.position.y = 0.0
        model_state_msg.pose.position.z = 0.0
        model_state_msg.pose.orientation.x = 0.0
        model_state_msg.pose.orientation.y = 0.0
        model_state_msg.pose.orientation.z = 0.0
        model_state_msg.pose.orientation.w = 0.0
        model_state_msg.reference_frame = "world"

        self.model_state_pub.publish(model_state_msg)
        self.reset_pub.publish(empty_msg)        

  def callback_imu(self,data):
    time_stamp = data.header.stamp.secs + \
                 data.header.stamp.nsecs / 1000000000
                 
    info = str(data.orientation.x) + " "  + \
           str(data.orientation.y) + " "  + \
           str(data.orientation.z) + " "  + \
           str(data.orientation.w) + " "  + \
           str(data.angular_velocity.x) + " "  + \
           str(data.angular_velocity.y) + " "  + \
           str(data.angular_velocity.z) + " "  + \
           str(data.linear_acceleration.x) + " "  + \
           str(data.linear_acceleration.y) + " "  + \
           str(data.linear_acceleration.z) + " "  + \
           str(data.orientation_covariance[0]) + " "  + \
           str(data.orientation_covariance[1]) + " "  + \
           str(data.orientation_covariance[2]) + " "  + \
           str(data.orientation_covariance[3]) + " "  + \
           str(data.orientation_covariance[4]) + " "  + \
           str(data.orientation_covariance[5]) + " "  + \
           str(data.orientation_covariance[6]) + " "  + \
           str(data.orientation_covariance[7]) + " "  + \
           str(data.orientation_covariance[8]) + " "  + \
           str(data.angular_velocity_covariance[0]) + " "  + \
           str(data.angular_velocity_covariance[1]) + " "  + \
           str(data.angular_velocity_covariance[2]) + " "  + \
           str(data.angular_velocity_covariance[3]) + " "  + \
           str(data.angular_velocity_covariance[4]) + " "  + \
           str(data.angular_velocity_covariance[5]) + " "  + \
           str(data.angular_velocity_covariance[6]) + " "  + \
           str(data.angular_velocity_covariance[7]) + " "  + \
           str(data.angular_velocity_covariance[8]) + " "  + \
           str(data.linear_acceleration_covariance[0]) + " "  + \
           str(data.linear_acceleration_covariance[1]) + " "  + \
           str(data.linear_acceleration_covariance[2]) + " "  + \
           str(data.linear_acceleration_covariance[3]) + " "  + \
           str(data.linear_acceleration_covariance[4]) + " "  + \
           str(data.linear_acceleration_covariance[5]) + " "  + \
           str(data.linear_acceleration_covariance[6]) + " "  + \
           str(data.linear_acceleration_covariance[7]) + " "  + \
           str(data.linear_acceleration_covariance[8]) + " "  + \
           str(time_stamp) + "\n"
    
    rospy.logdebug(info)
    
  def callback_bumper(self,data):                 
    self.bumper_msg = data
    
  def callback_navdata(self,data): 
    # 0: Unknown, 1: Init, 2: Landed, 3: Flying, 4: Hovering, 5: Test
    # 6: Taking off, 7: Goto Fix Point, 8: Landing, 9: Looping
    # Note: 3,7 seems to discriminate type of flying (isFly = 3 | 7)                 
    self.navdata_msg = data
    
  def callback_aruco(self,data):                 
    self.aruco_msg = data

def main(args):
  #cv2.startWindowThread()
  #cv2.namedWindow("Image window")
  rospy.init_node('optical_flow', anonymous=True, log_level=rospy.INFO)
  
  ac = actor_critic()
  time.sleep(1)
  #of.taking_off()
  
  rospy.loginfo("<------Data recorder (Author: Nino Cauli)------>")
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")   
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
