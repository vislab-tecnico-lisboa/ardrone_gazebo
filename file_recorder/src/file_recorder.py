#!/usr/bin/env python
from __future__ import print_function, division

import roslib
roslib.load_manifest('file_recorder')
import sys
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import time
import os

class optical_flow:

  def __init__(self):
    self.count = 0;
    self.bridge = CvBridge()   
    
    index = 0
    self.file_path = "/media/nigno/Data/testDatasetDrone/Sequence" \
                + str(index) + "/"
    
    while os.path.isdir(self.file_path):
      index += 1
      self.file_path = "/media/nigno/Data/testDatasetDrone/Sequence" \
                + str(index) + "/"    

    os.mkdir(self.file_path)    
 
    with open(self.file_path + "image_info.txt", "a") as myfile:
      myfile.write("name timestamp\n")
      
    with open(self.file_path + "cmd_vel.txt", "a") as myfile:
      myfile.write("vel_x vel_y vel_z ang_vel timestamp\n")
      
    with open(self.file_path + "imu.txt", "a") as myfile:
      myfile.write("o_x o_y o_z o_w av_x av_y av_z la_x la_y la_z " + \
                   "oc0 oc1 oc2 oc3 oc4 oc5 oc6 oc7 oc8 " + \
                   "avc0 avc1 avc2 avc3 avc4 avc5 avc6 avc7 avc8 " + \
                   "lac0 lac1 lac2 lac3 lac4 lac5 lac6 lac7 lac8 " + \
                   "timestamp\n")
    
    self.image_sub = rospy.Subscriber("/ardrone/front/image_raw",
                                      Image, self.callback_image)
    self.imu_sub = rospy.Subscriber("/ardrone/imu",
                                      Imu, self.callback_imu)
    self.ctrl_sub = rospy.Subscriber('/cmd_vel', 
                                      Twist, self.callback_cmd_vel)
                                      
    rospy.loginfo("Subscribers initialized")

  def callback_image(self,data):   
    time_stamp = data.header.stamp.secs + \
                 data.header.stamp.nsecs / 1000000000
    
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      rospy.logerror(e)
      
    cv_image_risized = cv2.resize(cv_image, (160, 80))

    file_name = "image" + \
                str(self.count) + ".png"

    cv2.imwrite(self.file_path + file_name, cv_image_risized);
    
    with open(self.file_path + "image_info.txt", "a") as myfile:
      myfile.write(file_name + " " + str(time_stamp) + "\n")

    rospy.logdebug(file_name + " " + str(time_stamp) + "\n")
    self.count += 1

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
                 
    with open(self.file_path + "imu.txt", "a") as myfile:
      myfile.write(info)
    
    rospy.logdebug(info)
    
  def callback_cmd_vel(self,data):
    now = rospy.get_rostime()
    time_stamp = now.secs + \
                 now.nsecs / 1000000000       
    
    info = str(data.linear.x) + " "  + \
           str(data.linear.y) + " "  + \
           str(data.linear.z) + " "  + \
           str(data.angular.z) + " "  + \
           str(time_stamp) + "\n"    
    
    with open(self.file_path + "cmd_vel.txt", "a") as myfile:
      myfile.write(info)
                   
    rospy.logdebug(info)


def main(args):
  #cv2.startWindowThread()
  #cv2.namedWindow("Image window")
  rospy.init_node('optical_flow', anonymous=True, log_level=rospy.INFO)
  
  of = optical_flow()
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
