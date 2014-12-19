/*!
  \package action_controller
  \author Rohan Bhargava
  \version 1.1.2
  */

#include <ros/ros.h>
#include <actionlib/server/action_server.h>
#include <pthread.h>
#include <action_controller/MultiDofFollowJointTrajectoryAction.h>
#include <geometry_msgs/Twist.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <nav_msgs/Odometry.h>
/*!
 * \brief This class listens for trajectories from moveit_simple_controller manager
 */
class Controller{
private:
    typedef actionlib::ActionServer<action_controller::MultiDofFollowJointTrajectoryAction> ActionServer;
    typedef ActionServer::GoalHandle GoalHandle;

public:
    /*!
     * \brief Constructor
     * \param n Handle for the node
     */
    Controller(ros::NodeHandle &n):
        node_(n),action_server_(node_,"multi_dof_joint_trajectory_action",boost::bind(&Controller::goalCB,this,_1),boost::bind(&Controller::cancelCB,this,_1),false),has_active_goal(false)
    {
      action_server_.start();
      ROS_INFO("Node ready");

    pub=node_.advertise<trajectory_msgs::MultiDOFJointTrajectory>("/command",1);
    sub=node_.subscribe<nav_msgs::Odometry>("/ground_truth/state",1000,boost::bind(&Controller::callback,this,_1));
    }
private:
    trajectory_msgs::MultiDOFJointTrajectory_<std::allocator<void> > traj;
    GoalHandle active_goal;
    ros::NodeHandle node_;
    ActionServer action_server_;
    ros::Publisher pub;
    ros::Subscriber sub;
    bool has_active_goal;
    bool execution_completed;
    /*!
     * \brief Functions is called whenever a new goal has been received
     * \param gh Goal handle
     */


    void goalCB(GoalHandle gh)
    {
        gh.setAccepted();
        traj=gh.getGoal()->trajectory;
        active_goal=gh;

        ros::param::set("/has_active_goal",true);
        pub.publish(traj);


    }
    /*!
     * \brief Functions is called whenever a goal has been cancelled
     * \param gh Goal handle
     */
    void cancelCB(GoalHandle gh)
    {
        if(active_goal==gh)
        {
            ros::param::set("/has_active_goal",false);
        }
    }
    /*!
     * \brief Callback functions for the state topic.
     * \todo This function is just used so that we can inform move it that executions is completed. Is it a hack?
     * \param msg Contains data published on the topic
     */
    void callback(const nav_msgs::OdometryConstPtr& msg)
    {
        ros::param::param<bool>("/has_active_goal",has_active_goal,1);
        if(has_active_goal==1)
        {
        return;
        }
        ros::param::param<bool>("/execution_completed",execution_completed,0);
        if( execution_completed==0)
        {
            return;
        }
        active_goal.setSucceeded();
        ros::param::set("/execution_completed",false);


    }
};
int main(int argc, char** argv)
{
    ros::init(argc, argv, "spiri_controller");
    ros::NodeHandle node;//("~");
    Controller control(node);

    ros::spin();

    return 0;
}


