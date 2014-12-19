/*!
  \package action_controller
  \author Rohan Bhargava
  \version 1.1.2
  */
#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include "trajectory_msgs/MultiDOFJointTrajectory.h"
#include "geometry_msgs/Pose.h"
#include <boost/bind.hpp>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <tf/transform_datatypes.h>
#include <spiri_motion_primitives/SpiriMoveToAction.h>
/*!
 * \brief Callback function for the command topic. It send goals to motion primitives server and waits for feedback.
 * \param msg Contains the trajectory published on the topic
 */

void chattercallback(const trajectory_msgs::MultiDOFJointTrajectoryConstPtr msg)
{
    ROS_INFO("inside the callback");
    ROS_INFO("Recieved the trajectory");

    actionlib::SimpleActionClient<spiri_motion_primitives::SpiriMoveToAction>ac("spiri_motion_primitives",true);
    ROS_INFO("Waiting for action server to start.");
    // wait for the action server to start
    ac.waitForServer(); //will wait for infinite time

    ROS_INFO("Action server started, sending goal.");
    // send a goal to the action
    spiri_motion_primitives::SpiriMoveToGoal goal;

    std::cout<<msg->points.size();
    for (int i=0;i<msg->points.size();i++)
    {
    goal.pose.header.stamp = ros::Time::now();
    goal.pose.header.frame_id = "world";
    goal.pose.pose.position.x = msg->points[i].transforms[0].translation.x;
    goal.pose.pose.position.y = msg->points[i].transforms[0].translation.y;
    goal.pose.pose.position.z = msg->points[i].transforms[0].translation.z;


    goal.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, 0.0);

    goal.speed = 1.0;
    goal.tolerance = 0.1;

    goal.use_distance_from_ground = false;

    ac.sendGoal(goal);
    bool has_active_goal;
    ros::param::get("/has_active_goal",has_active_goal);


    if(has_active_goal==0)
    {
        ac.cancelGoal();
        ROS_INFO("User cancelled the goal");
        break;

    }

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
    /*!
     * \param execution_completed Parameter to inform actioncontroller that the execution is comepleted
     * \param has_active_goal Parameter containing if a goal is being processed.
     */
    ros::param::set("/execution_completed",true);
    ros::param::set("/has_active_goal",false);

}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "spiri_command_listener");
  ROS_INFO("initated the node");
  //Spiri_ros_controller controller;
  ros::NodeHandle nh;
  ros::Subscriber sub=nh.subscribe("/command",1000,chattercallback);
  ROS_INFO("going to spin");
  ros::spin();

  return 0;
}
