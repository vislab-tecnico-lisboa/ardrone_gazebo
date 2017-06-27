#include <aruco_plugin/aruco_plugin.h>
#include "gazebo/common/Events.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/physics/Model.hh"

namespace gazebo
{
  ////////////////////////////////////////////////////////////////////////////////
  // Constructor
  ArucoPlugin::ArucoPlugin()
  : it_(nh_)
  {
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Destructor
  ArucoPlugin::~ArucoPlugin()
  {
    //event::Events::DisconnectWorldUpdateStart(updateConnection);
    event::Events::DisconnectWorldUpdateBegin(updateConnection);

    nh_.shutdown();
    delete &nh_;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Load the controller
  void ArucoPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
  { 
      image_received = false;
    
      // Store the pointer to the model
      this->model = _parent;

      // load parameters
      if (!_sdf->HasElement("robotNamespace"))
	robotNamespace.clear();
      else
    //     robotNamespace = _sdf->GetElement("robotNamespace")->GetValueString() + "/";
	robotNamespace = _sdf->GetElement("robotNamespace")->Get<std::string>() + "/";

      if (!_sdf->HasElement("topicName"))
	topicName = "aruco/pose";
      else
    //     topicName = _sdf->GetElement("topicName")->GetValueString();
	topicName = _sdf->GetElement("topicName")->Get<std::string>();
	
      if (!_sdf->HasElement("arucoMapFile"))
	arucoMapFile = "/media/map.yml";
      else
    //     topicName = _sdf->GetElement("topicName")->GetValueString();
	arucoMapFile = _sdf->GetElement("arucoMapFile")->Get<std::string>();
	
      if (!_sdf->HasElement("arucoFrontCameraFile"))
	arucoFrontCameraFile = "/media/front_camera.yml";
      else
    //     topicName = _sdf->GetElement("topicName")->GetValueString();
	arucoFrontCameraFile = _sdf->GetElement("arucoFrontCameraFile")->Get<std::string>();
      
	// start ros node
      if (!ros::isInitialized())
      {
	int argc = 0;
	char** argv = NULL;
	ros::init(argc,argv,"gazebo",ros::init_options::NoSigintHandler|ros::init_options::AnonymousName);
      }
      
      //nh_ = new ros::NodeHandle(robotNamespace);
      
    // Subscrive to input video feed and publish output video feed
    
      string image_topic_name, pub_topic_name;
      image_topic_name.append("/");
      image_topic_name.append(this->model->GetName());
      image_topic_name.append("/ardrone/front/ardrone/front/image_raw");
      pub_topic_name.append("/");
      pub_topic_name.append(this->model->GetName());
      pub_topic_name.append("/");
      pub_topic_name.append(topicName);
      
      image_sub_ = it_.subscribe(image_topic_name, 1, &ArucoPlugin::imageCb, this);
      aruco_pose_pub_ = nh_.advertise<geometry_msgs::Pose>(pub_topic_name, 1);

      //cv::namedWindow(OPENCV_WINDOW);
      
      corner = 0; //1 subpixel
      index = 0;
      
      //TheMarkerMapConfig.readFromFile("/home/nigno/Robot/catkinWS/src/ardrone_gazebo/aruco_test/media/map.yml");
      TheMarkerMapConfig.readFromFile(arucoMapFile);
      
      //TheMarkerMapConfigFile = "/home/nigno/Robot/catkinWS/src/ardrone_gazebo/aruco_test/media/map.yml";
      TheMarkerMapConfigFile = arucoMapFile;
      TheMarkerSize = 0.156;
      
      // read first image to get the dimensions
      //TheInputImage = cv_ptr->image;
      
      // read camera parameters if passed
      TheCameraParameters.readFromXMLFile(arucoFrontCameraFile);
      //TheCameraParameters.readFromXMLFile("/home/nigno/Robot/catkinWS/src/ardrone_gazebo/aruco_test/media/front_camera.yml");
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
    
    // Listen to the update event. This event is broadcast every
    // simulation iteration.
    this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&ArucoPlugin::OnUpdate, this));
  }

  void ArucoPlugin::OnUpdate()
  {
    if (image_received)
    {
      geometry_msgs::Pose aruco_pose;
    
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
	  //cout << "pose rt = "<< TheMSPoseTracker.getRvec() << " " << TheMSPoseTracker.getTvec() << endl;
	  aruco_pose.position.x = TheMSPoseTracker.getTvec().at<float>(0);
	  aruco_pose.position.y = TheMSPoseTracker.getTvec().at<float>(1);
	  aruco_pose.position.z = TheMSPoseTracker.getTvec().at<float>(2);
	  aruco_pose.orientation.x = TheMSPoseTracker.getRvec().at<float>(0);
	  aruco_pose.orientation.y = TheMSPoseTracker.getRvec().at<float>(1);
	  aruco_pose.orientation.z = TheMSPoseTracker.getRvec().at<float>(2);
	  aruco_pose.orientation.w = TheMSPoseTracker.getRvec().at<float>(3);
	}
      }
      
      // Output aruco pose
      aruco_pose_pub_.publish(aruco_pose);
      image_received = false;
    }
  }
  
  void ArucoPlugin::imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    image_received = true;
  }
  
  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(ArucoPlugin)
} // namespace gazebo
