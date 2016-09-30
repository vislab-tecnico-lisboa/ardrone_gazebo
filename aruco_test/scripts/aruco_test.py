#!/usr/bin/env python
from __future__ import print_function, division

import roslib
roslib.load_manifest('aruco_test')
import sys
import rospy
import numpy as np
import cv2
import aruco
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from keypressed import getchar
import time

class ArucoTest:

  def __init__(self):
    self.first_image = True      
    self.marker_size = 0.15        
      
    # load board and camera parameters
    self.boardconfig = aruco.BoardConfiguration("../media/map.yml")   
    self.camparam = aruco.CameraParameters()
    self.camparam.readFromXMLFile("../media/front_camera.yml")   
    
    # create detector and set parameters
    self.detector = aruco.BoardDetector()
    self.detector.setParams(self.boardconfig, self.camparam)
    
    # set minimum marker size for detection
    self.markerdetector = self.detector.getMarkerDetector()
    self.markerdetector.setMinMaxSize(0.01)
    self.markerdetector.setThresholdParams(7, 7);
    self.markerdetector.setThresholdParamRange(2, 0);
    
    if self.boardconfig.isExpressedInPixels():
      print("pixels")
    else:
      print("meters")
    
    
    self.bridge = CvBridge()    
    self.image_sub = rospy.Subscriber("/ardrone/front/image_raw",
                                      Image, self.callback_image)
   
    print("Subscribers initialized")


  def callback_image(self,data):     
        
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
      
    #if self.camparam.isValid() and self.first_image:
      #self.camparam.resize(cv_image.size())
      #self.first_image = False

    frame_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    vis = cv_image.copy()
    
    
    likelihood = self.detector.detect_mat(vis)
        
    markers = self.markerdetector.detect(vis, self.camparam, self.marker_size)

    if likelihood > 0.1:
      # get board and draw it    
      board = self.detector.getDetectedBoard()
      board.draw(vis, np.array([255, 255, 255]), 2)

      for marker in board:
        print ("cornerpoints for marker %d:", marker.id)
        for i, point in enumerate(marker):
          print (i, point)

      print ("detected ids: ", ", ".join(str(m.id) for m in board))
    
    # show frame    
    cv2.imshow("Image window", vis)
    cv2.waitKey(1)


def main(args):
  #cv2.startWindowThread()
  #cv2.namedWindow("Image window")
  rospy.init_node('optical_flow', anonymous=True)

  at = ArucoTest()
  time.sleep(1)
  
  print("<------Aruco marker recognition (Author: Nino Cauli)------>")
  print("")
  print("Press the following keys in order to execute the correspondent commands:")
  print("q ---> quit")
  print("<----------------------------------------------------------------------------------->")
  
  
  key = ''  
  
  while not rospy.is_shutdown():
      key = getchar()
      if key == 'q':
          break
  
  print("Shutting down")      
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
