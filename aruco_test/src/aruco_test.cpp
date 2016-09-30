#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "aruco.h"

using namespace cv;
using namespace aruco;

static const std::string OPENCV_WINDOW = "Image window";

class ArucoTest
{
private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  
  bool first_image = true;
  
  std::string aruco_map_file, aruco_front_camera_file;
  
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
  
public:
  ArucoTest()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/ardrone/image_raw", 1, 
      &ArucoTest::imageCb, this);
    image_pub_ = it_.advertise("/ardrone/aruco/image_raw", 1);

    cv::namedWindow(OPENCV_WINDOW);
    
    // aruco part
    ros::NodeHandle private_node_handle;
    private_node_handle.param<std::string>("", aruco_map_file, "aruco_map_file");
    private_node_handle.param<std::string>("", aruco_front_camera_file, "aruco_front_camera_file");
    
    corner = 0; //1 subpixel
    index = 0;
    
    cout << aruco_map_file;
    
    TheMarkerMapConfig.readFromFile("/home/nigno/Robot/catkinWS/src/ardrone_gazebo/aruco_test/media/map.yml");
    //TheMarkerMapConfig.readFromFile(aruco_map_file);
    
    TheMarkerMapConfigFile = "/home/nigno/Robot/catkinWS/src/ardrone_gazebo/aruco_test/media/map.yml";
    //TheMarkerMapConfigFile = aruco_map_file;
    TheMarkerSize = 0.15;
    
    // read first image to get the dimensions
    //TheInputImage = cv_ptr->image;
    
    // read camera parameters if passed
    //TheCameraParameters.readFromXMLFile(aruco_front_camera_file);
    TheCameraParameters.readFromXMLFile("/home/nigno/Robot/catkinWS/src/ardrone_gazebo/aruco_test/media/front_camera.yml");
    //TheCameraParameters.resize(TheInputImage.size());
      
    //prepare the detector
    string dict=TheMarkerMapConfig.getDictionary();//see if the dictrionary is already indicated in the configuration file. It should!
    if(dict.empty()) dict="ARUCO";
    TheMarkerDetector.setDictionary(dict);///DO NOT FORGET THAT!!! Otherwise, the ARUCO dictionary will be used by default!
    if (corner == 0)
      TheMarkerDetector.setCornerRefinementMethod(MarkerDetector::LINES);
    else{
      MarkerDetector::Params params=TheMarkerDetector.getParams();
      params._cornerMethod=MarkerDetector::SUBPIX;
      params._subpix_wsize= (15./2000.)*float(TheInputImage.cols) ;//search corner subpix in a 5x5 widow area
      TheMarkerDetector.setParams(params);
    }
    
    //prepare the pose tracker if possible
    //if the camera parameers are avaiable, and the markerset can be expressed in meters, then go
    
    if ( TheMarkerMapConfig.isExpressedInPixels() && TheMarkerSize>0)
      TheMarkerMapConfig=TheMarkerMapConfig.convertToMeters(TheMarkerSize);
    cout<<"TheCameraParameters.isValid()="<<TheCameraParameters.isValid()<<" "<<TheMarkerMapConfig.isExpressedInMeters()<<endl;
    if (TheCameraParameters.isValid() && TheMarkerMapConfig.isExpressedInMeters()  )
      TheMSPoseTracker.setParams(TheCameraParameters,TheMarkerMapConfig);
    
    
    // Create gui
    
    cv::namedWindow("thres", 1);
    cv::namedWindow("in", 1);
    
//     TheMarkerDetector.getThresholdParams(ThresParam1, ThresParam2);
//     iThresParam1 = ThresParam1;
//     iThresParam2 = ThresParam2;
//     cv::createTrackbar("ThresParam1", "in", &iThresParam1, 13, cvTackBarEvents);
//     cv::createTrackbar("ThresParam2", "in", &iThresParam2, 13, cvTackBarEvents);
  }

  ~ArucoTest()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    
    TheInputImage = cv_ptr->image;
    
    if (first_image)
    {
      TheCameraParameters.resize(TheInputImage.size());
      first_image = false;
    }
    
    TheInputImage.copyTo(TheInputImageCopy);
    index++; // number of images captured
    
    // Detection of the board
    vector<aruco::Marker> detected_markers=TheMarkerDetector.detect(TheInputImage);
    //print the markers detected that belongs to the markerset
    for(auto idx:TheMarkerMapConfig.getIndices(detected_markers))
      detected_markers[idx].draw(TheInputImageCopy, Scalar(0, 0, 255), 2);
    //detect 3d info if possible
    if (TheMSPoseTracker.isValid()){
      if ( TheMSPoseTracker.estimatePose(detected_markers)){
	aruco::CvDrawingUtils::draw3dAxis(TheInputImageCopy,  TheCameraParameters,TheMSPoseTracker.getRvec(),TheMSPoseTracker.getTvec(),TheMarkerMapConfig[0].getMarkerSize()*2);
	frame_pose_map.insert(make_pair(index,TheMSPoseTracker.getRTMatrix() ));
	cout<<"pose rt="<<TheMSPoseTracker.getRvec()<<" "<<TheMSPoseTracker.getTvec()<<endl;
      }
    }
    
    // show input with augmented information and  the thresholded image
    cv::imshow("in", TheInputImageCopy);
    cv::imshow("thres",TheMarkerDetector.getThresholdedImage());

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(1);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "aruco_test");
  ArucoTest at;
  ros::spin();
  return 0;
}