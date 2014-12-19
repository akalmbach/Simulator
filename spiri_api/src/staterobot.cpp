/*!
  \brief C++ API for Spiri
  \package spiri_api
  \author Rohan Bhargava and Arnold Kalmbach
  \version 1.1.3
  
  */

#include <spiri_api/staterobot.h>
#include <iostream>
#include <ctime>
#include <boost/timer.hpp>
#include <sstream>
/*!
  Constructor
  */
Staterobot::Staterobot()
{
    int dummy_argc=0;
    ros::init((int&)dummy_argc,NULL,"spiri_api");
    ros::NodeHandle n;
    success_plan=false;
    success_execution=false;


}

/*!
  Get the orientation in quaternion from IMU

  \return Orientation (x,y,z,w) of Spiri
  */

Staterobot::imu Staterobot::get_imu()
{

    imu_ptr=ros::topic::waitForMessage<sensor_msgs::Imu>("/raw_imu");


    imu imu_data;
    imu_data.x=imu_ptr->orientation.x;
    imu_data.y=imu_ptr->orientation.y;
    imu_data.z=imu_ptr->orientation.z;
    imu_data.w=imu_ptr->orientation.w;
    return imu_data;



}


/*!
  Get orientation from IMU

  @return The orientation in quaternion (x,y,z,w)

 */

boost::python::list Staterobot::get_imu_python()
{
    Staterobot::imu imu_data=this->get_imu();
    std::vector<double> v(4);
    v[0]=imu_data.x;
    v[1]=imu_data.y;
    v[2]=imu_data.z;
    v[3]=imu_data.w;

    return moveit::py_bindings_tools::listFromDouble(v);
}


/*!
  Get the state

  @return Position(x,y,z) and orientation (x,y,z,w) of Spiri
  */

Staterobot::state Staterobot::get_state()
{
    try
	{
	    ros::AsyncSpinner spinner(1);
	    spinner.start();
	    moveit::planning_interface::MoveGroup group("spiri");
	    std::vector<double> state_test(7);
	    state_test=group.getCurrentJointValues();

	    spinner.stop();
	    state state_data;
	    state_data.position.x=state_test[0];
	    state_data.position.y=state_test[1];
	    state_data.position.z=state_test[2];
	    state_data.orientation.x=state_test[3];
	    state_data.orientation.y=state_test[4];
	    state_data.orientation.z=state_test[5];
	    state_data.orientation.w=state_test[6];
	    return state_data;
	}
    catch(...)
	{
		state_ptr=ros::topic::waitForMessage<nav_msgs::Odometry>("/ground_truth/state");
		state state_data;
    		state_data.position.x=state_ptr->pose.pose.position.x;
	        state_data.position.y=state_ptr->pose.pose.position.y;
		state_data.position.z=state_ptr->pose.pose.position.z;
		state_data.orientation.x=state_ptr->pose.pose.orientation.x;
		state_data.orientation.y=state_ptr->pose.pose.orientation.y;
		state_data.orientation.z=state_ptr->pose.pose.orientation.z;
		state_data.orientation.w=state_ptr->pose.pose.orientation.w;
                return state_data;  			
		
	}
    

    


    ;

}


/*!
  Get the state of Spiri

  @return Position(x,y,z) and orientation(x,y,z,w)
 */

boost::python::list Staterobot::get_state_python()
{
    state pose_data=this->get_state();

    std::vector<double> v(7);
    v[0]=pose_data.position.x;
    v[1]=pose_data.position.y;
    v[2]=pose_data.position.z;
    v[3]=pose_data.orientation.x;
    v[4]=pose_data.orientation.y;
    v[5]=pose_data.orientation.z;
    v[6]=pose_data.orientation.w;

    return moveit::py_bindings_tools::listFromDouble(v);
}

/*!
  Get the height in metres from pressure sensor

  @return Altitude of Spiri
  */


float Staterobot::get_height_pressure()
{
    pressure=ros::topic::waitForMessage<geometry_msgs::PointStamped>("pressure_height");
    return pressure->point.z;
}


/*!
  Get the gps position

  @return GPS (latitude,longitude,altitude) of Spiri
  */
Staterobot::gps Staterobot::get_gps_data()
{
    gps_ptr=ros::topic::waitForMessage<sensor_msgs::NavSatFix>("/fix");
    gps gps_data;
    gps_data.latitude=gps_ptr->latitude;
    gps_data.longitude=gps_ptr->longitude;
    gps_data.altitude=gps_ptr->altitude;
    return gps_data;
}
/*!
    Get the gps data

    @return Latitude, longitude and altitude
 */
boost::python::list Staterobot::get_gps_data_python()
{
    gps gps_data=this->get_gps_data();

    std::vector<double> v(3);
    v[0]=gps_data.latitude;
    v[1]=gps_data.longitude;
    v[2]=gps_data.altitude;

    return moveit::py_bindings_tools::listFromDouble(v);
}

/*!
  Get the velocity reporetd by the GPS

  @return Velocity (x,y,z) of Spiri
  */
Staterobot::gps_vel Staterobot::get_gps_vel()
{
    gps_vel_ptr=ros::topic::waitForMessage<geometry_msgs::Vector3Stamped>("/fix_velocity");
    gps_vel gps_vel_data;
    gps_vel_data.x=gps_vel_ptr->vector.x;
    gps_vel_data.y=gps_vel_ptr->vector.y;
    gps_vel_data.z=gps_vel_ptr->vector.z;

    return gps_vel_data;
}
/*!
  Velocity reported by GPS

  @return Velocity in x,y and z direction
 */
boost::python::list Staterobot::get_gps_vel_python()
{
    gps_vel vel_data=this->get_gps_vel();

    std::vector<double> v(3);
    v[0]=vel_data.x;
    v[1]=vel_data.y;
    v[2]=vel_data.z;

    return moveit::py_bindings_tools::listFromDouble(v);

}


/*!
  Get the height in metres from altimeter

  @return Altitude of Spiri
  */
float Staterobot::get_height_altimeter()
{
    altimeter=ros::topic::waitForMessage<hector_uav_msgs::Altimeter>("/altimeter");
    return altimeter->altitude;
}


/*!
  Get the image from left front camera.

  @return Image (640X480)
  */

cv::Mat Staterobot::get_left_image()

{
    ros::NodeHandle n;
    n.setCallbackQueue(&image_queue);
    image_transport::ImageTransport it(n);
    sub=it.subscribe("/stereo/left/image_raw", 1, &Staterobot::image_left_callback,this);



    image_queue.callAvailable(ros::WallDuration(1.0));
    return this->left_image;

}

/*!
  Get image from the left camera

  @return Image which is converted by the python api into a numpy array
 */
std::string Staterobot::get_left_image_python()

{
    cv::Mat mat=this->get_left_image();
    cv::Size size = mat.size();
    int total = size.width * size.height * mat.channels();

    std::vector<uchar> data(mat.ptr(),mat.ptr()+total);
    std::string image(data.begin(),data.end());

    return image;

}



void Staterobot::image_left_callback(const sensor_msgs::ImageConstPtr & msg)
{


    left_image=cv_bridge::toCvShare(msg,"bgr8")->image;

}

/*!
  Get the image from right front camera

  @return Image (640X480)
  */
cv::Mat Staterobot::get_right_image()

{
    // @todo there is a better way to do this. Callback to a subscriber. No need to create a node

    // @todo the image callback needs to be called once or else it return an older image.
    ros::NodeHandle n;
    n.setCallbackQueue(&image_queue);
    image_transport::ImageTransport it(n);
    sub=it.subscribe("/stereo/right/image_raw", 1, &Staterobot::image_right_callback,this);



    image_queue.callAvailable(ros::WallDuration(1.0));
    return this->right_image;

}

/*!
  Get image from the right camera

  @return Image which is converted by the python api into a numpy array
 */
std::string Staterobot::get_right_image_python()

{
    cv::Mat mat=this->get_right_image();
    cv::Size size = mat.size();
    int total = size.width * size.height * mat.channels();

    std::vector<uchar> data(mat.ptr(),mat.ptr()+total);
    std::string image(data.begin(),data.end());

    return image;

}
void Staterobot::image_right_callback(const sensor_msgs::ImageConstPtr & msg)
{


    right_image=cv_bridge::toCvShare(msg,"bgr8")->image;

}
/*!
  Get the image from bottom camera

  @return Image (640X480)
  */
cv::Mat Staterobot::get_bottom_image()

{
    ros::NodeHandle n;
    n.setCallbackQueue(&image_queue);
    image_transport::ImageTransport it(n);
    sub=it.subscribe("/downward_cam/camera/image", 1, &Staterobot::image_bottom_callback,this);



    image_queue.callAvailable(ros::WallDuration(1.0));
    return this->bottom_image;

}


/*!
  Get image from the bottom camera

  @return Image which is converted by the python api into a numpy array
 */
std::string Staterobot::get_bottom_image_python()

{
    cv::Mat mat=this->get_bottom_image();
    cv::Size size = mat.size();
    int total = size.width * size.height * mat.channels();

    std::vector<uchar> data(mat.ptr(),mat.ptr()+total);
    std::string image(data.begin(),data.end());

    return image;

}
void Staterobot::image_bottom_callback(const sensor_msgs::ImageConstPtr & msg)
{


    bottom_image=cv_bridge::toCvShare(msg,"bgr8")->image;

}
/*!
  Save the image

  @param path location to save the files
  @param camera which camera to save the image from [left,right,bottom]

  @return Image (640X480)
  */
void Staterobot::save_image(const std::string path="",const std::string camera="")
{
    //std::cout<<path;
    cv::Mat image;
    if(camera=="left")
    {

            image=get_left_image();
    }
    else if(camera=="right")
    {
        image=get_right_image();
    }
    else if(camera=="bottom")
    {
        image=get_bottom_image();
    }
    else
    {
        std::cout<<"Please pass a valid camera type";

    }


    //cv::Mat image=get_left_image();

    cv::imwrite(path,image);
}
/*!
  If the goal has been reached or not


  @return True if goal has been reached otherwise False
  */
bool Staterobot::wait_goal()
{
    bool param;
    ros::param::get("execution",param);
    return param;


}

void Staterobot::callback_goal(const action_controller::MultiDofFollowJointTrajectoryActionResultPtr& msg)
{
    ROS_INFO("here");
    //std::cout<<"here";
    status=true;

}
/*!
  Send goal to Spiri

  @param x coordinate in x direction
  @param y coordinate in y direction
  @param z coordinate in z direction
  @param relative If set then goal is calculated with respect to the start position otherwise coordinates are with respect to the world

  @return Succesfully executed or not
  */
bool Staterobot::send_goal(float x,float y,float z, bool relative=false)
{
    ros::AsyncSpinner spinner(1);
    spinner.start();
    moveit::planning_interface::MoveGroup group("spiri");
    group.setPlannerId("RRTConnectkConfigDefault");
    group.setWorkspace(-5.0,-5.0,-5.0,100.0,100.0,100.0);



    group.setStartStateToCurrentState();
    // after setting the start state send the goal
    group.getCurrentState()->copyJointGroupPositions(group.getCurrentState()->getRobotModel()->getJointModelGroup(group.getName()), group_variable_values);
    if (relative==true)
    {
        state current_state=get_state();

        group_variable_values[0]=current_state.position.x+x;
        group_variable_values[1]=current_state.position.y+y;
        group_variable_values[2]=current_state.position.z+z;
    }
    else
    {
        group_variable_values[0]=x;
        group_variable_values[1]=y;
        group_variable_values[2]=z;
    }
    group.setJointValueTarget(group_variable_values);

    moveit::planning_interface::MoveGroup::Plan my_plan;
    // @todo Can combine plan and execute into one by using move.
    success_plan = group.plan(my_plan);
    if(success_plan==1)
    {
            ROS_INFO("Going to execute");
            success_execution=group.asyncExecute(my_plan);

    }
    else
    {
        ROS_INFO("Couldn't find a valid plan");
    }
    
    ROS_INFO("out of this loop");
    spinner.stop();
    return success_execution;

}

/*!
  Send goal to Spiri with respect to the world

  @param values Cooridinates in x,y,z

  @return Succesfull
 */
bool Staterobot::send_goal_python(boost::python::list &values)
{
    std::vector<double> v(3);
    v=moveit::py_bindings_tools::doubleFromList(values);
    bool flag=this->send_goal(v[0],v[1],v[2],false);
    return flag;
}

/*!
  Send goal to Spiri with respect to the current position

  @param values Cooridinates in x,y,z

  @return Succesfull
 */
bool Staterobot::send_goal_python_relative(boost::python::list &values)
{
    std::vector<double> v(3);
    v=moveit::py_bindings_tools::doubleFromList(values);
    bool flag=this->send_goal(v[0],v[1],v[2],true);
    return flag;
}
/*!
 Stop a exisiting trajectory being followed
 */

void Staterobot::stop_traj()
{
    ros::AsyncSpinner spinner(1);
    spinner.start();
    moveit::planning_interface::MoveGroup group("spiri");
    group.stop();
    spinner.stop();
  
}
/*!
  Send velocity to Spiri.

  @param x velocity in x direction
  @param y velocity in y direction
  @param z velocity in z direction


  */


void Staterobot::send_vel(float x,float y,float z)
{

    ROS_INFO("testing");
    ros::NodeHandle nh;
    bool latch=1;
    ros::Publisher vel_chatter = nh.advertise<geometry_msgs::Twist>("cmd_vel",1,latch);


    geometry_msgs::Twist vel;


    vel.linear.x=x;
    vel.linear.y=y;
    vel.linear.z=z;

    vel_chatter.publish(vel);
    // this is hack but it looks like it is not possible in ROS
    /*! @todo This is hack. If there exists a better way in ROS need to find that*/
    sleep(1.0);
    //ros::spinOnce();

}
/*!
  Send velocity commands to Spiri

  @param val Velocities in m/s in x,y,z direction


 */
void Staterobot::send_vel_python(boost::python::list &val)
{
    std::vector<double> v(3);
    v=moveit::py_bindings_tools::doubleFromList(val);
    this->send_vel(v[0],v[1],v[2]);
}

void Staterobot::land()
{
    actionlib::SimpleActionClient<spiri_motion_primitives::SpiriMoveToAction> ac("spiri_motion_primitives", true);

    ROS_INFO("Waiting for action server to start.");
    // wait for the action server to start
    ac.waitForServer(); //will wait for infinite time

    ROS_INFO("Action server started, sending goal.");
    // send a goal to the action
    spiri_motion_primitives::SpiriMoveToGoal goal;

    
    goal.pose.header.stamp = ros::Time::now();
    goal.pose.header.frame_id = "world";
    goal.pose.pose.position.x = 0.0;
    goal.pose.pose.position.y = 0.0;
    goal.pose.pose.position.z = 0.0;
    
    goal.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, 0.0);
    
    goal.speed = 1.0;
    goal.tolerance = 0.1;

    goal.use_distance_from_ground = true;

    ac.sendGoal(goal);

    //wait for the action to return
    bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

    if (finished_before_timeout)
    {
        actionlib::SimpleClientGoalState state = ac.getState();
        ROS_INFO("Action finished: %s",state.toString().c_str());
    }
    else
    {
        ac.cancelGoal();
        ROS_INFO("Action did not finish before the time out.");
    }
}

void Staterobot::takeoff()
{
    actionlib::SimpleActionClient<spiri_motion_primitives::SpiriMoveToAction> ac("spiri_motion_primitives", true);

    ROS_INFO("Waiting for action server to start.");
    // wait for the action server to start
    ac.waitForServer(); //will wait for infinite time

    ROS_INFO("Action server started, sending goal.");
    // send a goal to the action
    spiri_motion_primitives::SpiriMoveToGoal goal;

    
    goal.pose.header.stamp = ros::Time::now();
    goal.pose.header.frame_id = "world";
    goal.pose.pose.position.x = 0.0;
    goal.pose.pose.position.y = 0.0;
    goal.pose.pose.position.z = 1.0;
    
    goal.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, 0.0);
    
    goal.speed = 1.0;
    goal.tolerance = 0.1;

    goal.use_distance_from_ground = true;

    ac.sendGoal(goal);

    //wait for the action to return
    bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

    if (finished_before_timeout)
    {
        actionlib::SimpleClientGoalState state = ac.getState();
        ROS_INFO("Action finished: %s",state.toString().c_str());
    }
    else
    {
        ac.cancelGoal();
        ROS_INFO("Action did not finish before the time out.");
    }
}





