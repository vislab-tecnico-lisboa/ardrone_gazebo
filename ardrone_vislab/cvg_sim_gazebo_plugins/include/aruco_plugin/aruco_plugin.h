#ifndef ARUCO_PLUGIN_ARUCO_PLUGIN_H
#define ARUCO_PLUGIN_ARUCO_PLUGIN_H

#include "gazebo/common/Plugin.hh"
#include "gazebo/common/Time.hh"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Twist.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "aruco.h"

using namespace cv;
using namespace aruco;

namespace gazebo
{
   class ArucoPlugin : public ModelPlugin
   {
   private:
      // Pointer to the model
      physics::ModelPtr model;

      // Pointer to the update event connection
      event::ConnectionPtr updateConnection;
      
      /// \brief for setting ROS name space
      std::string robotNamespace;
      
      /// \brief topic name
      std::string topicName;
      
      std::string arucoMapFile, arucoFrontCameraFile;
      
      bool image_received;
      
      ros::NodeHandle nh_;
      image_transport::ImageTransport it_;
      image_transport::Subscriber image_sub_;
      ros::Publisher aruco_pose_pub_;
      
      bool first_image = true;
      
      int corner;
      int index;
      string TheMarkerMapConfigFile;
      bool The3DInfoAvailable = false;
      float TheMarkerSize = -1;
      VideoCapture TheVideoCapturer;
      Mat TheInputImage, TheInputImageCopy;
      CameraParameters TheCameraParameters;
      MarkerMap TheMarkerMapConfig;
      MarkerDetector TheMarkerDetector;
      MarkerMapPoseTracker TheMSPoseTracker;
      void cvTackBarEvents(int pos, void *);
      double ThresParam1, ThresParam2;
      int iThresParam1, iThresParam2;
      int waitTime = 10;
      std::map<int,cv::Mat> frame_pose_map;//set of poses and the frames they were detected
      
      cv_bridge::CvImagePtr cv_ptr;
      
   public:
      /// \brief Constructor
      ArucoPlugin();

      /// \brief Destructor
      virtual ~ArucoPlugin();

   protected:
      virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
      virtual void OnUpdate();
      
      void imageCb(const sensor_msgs::ImageConstPtr& msg);
   };
}

#endif // ARUCO_PLUGIN_ARUCO_PLUGIN_H
