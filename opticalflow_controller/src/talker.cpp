#include <iostream>

#include <ros/ros.h>
#include <ardrone_autonomy/Navdata.h>

#define g 9.8

class Positions
{
	
private:

	ros::NodeHandle nh_;
	ros::Subscriber navdata_sub;

	float pos_x;
	float pos_y;
	float pos_z;

	float vel_x;
	float vel_y;
	float vel_z;

	float acc_x;
	float acc_y;
	float acc_z;

	double time;

public:

	Positions(ros::NodeHandle &nh)
	{
			nh_ = nh;
	}

	void callbackFunc(const ardrone_autonomy::Navdata robotInfo)
	{
		this->vel_x = robotInfo.vx;
		this->vel_y = robotInfo.vy;
		this->vel_z = robotInfo.vz;

		this->acc_x = robotInfo.ax;
		this->acc_y = robotInfo.ay;
		this->acc_z = robotInfo.az;

		this->time = robotInfo.header.stamp.sec + robotInfo.header.stamp.nsec*1E-09;
		//this->time = robotInfo.header.stamp.sec;

		ROS_INFO("Velocity_x = [%f] || Time = [%f]", this->vel_x, this->time);
	}

	float posCalc()
	{
		float t_stop;
		float t_go;

		float t = 0;
		float x0 = this->pos_x;
		float y0 = this->pos_y;
		float z0 = this->pos_z;

		float v0x = this->vel_x;
		float v0y = this->vel_y;
		float v0z = this->vel_y;
		
		float ax = this->acc_x;
		float ay = this->acc_y;
		float az = this->acc_z;



		
		float x = x0 + v0x*t + 0.5*t*t*ax;
		float y = y0 + v0y*t + 0.5*t*t*ay;
		float z = z0 + v0z*t + 0.5*t*t*(az-g);

		this->pos_x = x;
		this->pos_y = y;
		this->pos_z = z;

	}

	void listener()
	{
		navdata_sub = nh_.subscribe("/ardrone/navdata", 1, &Positions::callbackFunc, this);
	}
};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "Positions");
  	ros::NodeHandle nh;
	
	Positions teste(nh);
	teste.listener();
	teste.posCalc();
	ros::spin();
		
	return 0;
}