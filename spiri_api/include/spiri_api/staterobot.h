#ifndef STATEROBOT_H
#define STATEROBOT_H
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PointStamped.h"
#include "sensor_msgs/NavSatFix.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "hector_uav_msgs/Altimeter.h"
#include "ros/topic.h"
#include <ros/callback_queue.h>
#include <vector>
#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <opencv2/highgui/highgui.hpp>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <string.h>
#include <action_controller/MultiDofFollowJointTrajectoryActionResult.h>
#include <boost/shared_ptr.hpp>
#include <actionlib_msgs/GoalStatus.h>
#include <std_msgs/UInt32.h>
#include <boost/python.hpp>
#include <moveit/py_bindings_tools/py_conversions.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include <opencv2/core/core.hpp>

#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <tf/transform_datatypes.h>
#include <spiri_motion_primitives/SpiriMoveToAction.h>


class Staterobot
{
    std::vector<double> group_variable_values;
    moveit_msgs::RobotState start_state;
    geometry_msgs::Transform transform;
    std::vector<double> current_state;
    bool success_plan;
    bool success_execution;

    cv::Mat left_image;
    cv::Mat right_image;
    cv::Mat bottom_image;
    ros::CallbackQueue image_queue;




private:

public:
    Staterobot();
    struct imu{
        double x;
        double y;
        double z;
        double w;
    };
    struct position{
        double x;
        double y;
        double z;
    };
    struct orientation{
        double x;
        double y;
        double z;
        double w;
    };
    struct state{
        struct position position;
        struct orientation orientation;

    };
    struct gps
    {
        double latitude;
        double longitude;
        double altitude;
    };
    struct gps_vel{
        double x;
        double y;
        double z;
    };



    Staterobot::imu get_imu();
    void image_left_callback(const sensor_msgs::ImageConstPtr &);
    void image_right_callback(const sensor_msgs::ImageConstPtr &);
    void image_bottom_callback(const sensor_msgs::ImageConstPtr &);
    Staterobot::state get_state();
    float get_height_pressure();
    bool wait_goal();
    void callback_goal(const action_controller::MultiDofFollowJointTrajectoryActionResultPtr&);
    Staterobot::gps get_gps_data();
    Staterobot::gps_vel get_gps_vel();
    float get_height_altimeter();
    bool send_goal(float x,float y,float z,bool relative);
    cv::Mat get_left_image();
    cv::Mat get_right_image();
    cv::Mat get_bottom_image();
    void save_image(const std::string,const std::string);
    void send_vel(float x,float y,float z);
    sensor_msgs::ImuConstPtr imu_ptr;
    nav_msgs::OdometryConstPtr state_ptr;
    geometry_msgs::PointStampedConstPtr pressure;
    sensor_msgs::NavSatFixConstPtr gps_ptr;
    geometry_msgs::Vector3StampedConstPtr gps_vel_ptr;
    hector_uav_msgs::AltimeterConstPtr altimeter;
    sensor_msgs::ImageConstPtr img;
    cv_bridge::CvImagePtr cv_ptr;
    image_transport::Subscriber sub;
    action_controller::MultiDofFollowJointTrajectoryActionResult result;
    boost::shared_ptr<action_controller::MultiDofFollowJointTrajectoryActionResult const> errorPtr;
    bool status;
    boost::python::list get_imu_python();
    boost::python::list get_state_python();
    boost::python::list get_gps_data_python();
    boost::python::list get_gps_vel_python();
    std::string get_left_image_python();
    std::string get_right_image_python();
    std::string get_bottom_image_python();
    bool send_goal_python(boost::python::list &);
    bool send_goal_python_relative(boost::python::list &);
    void send_vel_python(boost::python::list &);

    void stop_traj();
    
    void land();
    void takeoff();





};

#endif // STATEROBOT_H
