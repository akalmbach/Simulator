#! /usr/bin/env python

## @package spiri_controller
# Listens to trajectory and generates the appropriate velocity to move the Quadcopter. 
# @todo This should also convert cmd_vel to pitch
# @author Rohan Bhargava
# @version 1.1.1
import roslib
roslib.load_manifest('spiri_motion_primitives')
import rospy
import trajectory_msgs.msg
import actionlib
from spiri_api import pid
from collections import deque
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry

class spiri_ros_controller(object):
  ## Constructor
  def __init__(self,name):
    self.state=Pose()
    rospy.Subscriber("/command",trajectory_msgs.msg.MultiDOFJointTrajectory,self.vel_callback)
    rospy.Subscriber("/ground_truth/state",Odometry,self.state_callback)
  ## Gets the waypoints with respect to the current position
  # @return A collection of waypoints
  def compute_goal(self,traj):
    
    current_state=self.state
    goal=deque()
    
    
    goal.append(current_state)

    for i in range(1,len(traj.points)):
	    temp=Pose()
	    temp.position.x=goal[i-1].position.x+(traj.points[i].transforms[0].translation.x-traj.points[i-1].transforms[0].translation.x)
	    temp.position.y=goal[i-1].position.y+(traj.points[i].transforms[0].translation.y-traj.points[i-1].transforms[0].translation.y)
	    temp.position.z=goal[i-1].position.z+(traj.points[i].transforms[0].translation.z-traj.points[i-1].transforms[0].translation.z)
	    goal.append(temp)
  
    return goal

  ## Publishes velocity to reach a waypoint
  # @param goal A collection of waypoints
  
  def pid_vel(self,goal):
    current_state=self.state
    rospy.loginfo("Publishing vel")
    kp=rospy.get_param('kp',2.0)
    ki=rospy.get_param('ki',0.0)
    kd=rospy.get_param('kd',2.0)

    pid_object=pid.PID([kp,kp,kp],[ki,ki,ki],[kd,kd,kd])
    pub=rospy.Publisher("cmd_vel",Twist,queue_size=10)
    vel=Twist()
    while len(goal)>0 and rospy.get_param('/has_active_goal')==True:
	    temp_goal=goal.popleft()
	    ac=actionlib.SimpleActionClien("spiri_motion_primitives",spiri_motion_primitives.msg._SpiriMoveToAction,True);
	    ac.wait_for_server()
	    pid_object.setPoint(temp_goal)
	    #current_state=spiri_obj.get_state()
	    while abs(current_state.position.x-temp_goal.position.x)>0.1 or abs(current_state.position.y-temp_goal.position.y)>0.1 or abs(current_state.position.z-temp_goal.position.z)>0.1:
		    
	      current_state=self.state
	      temp=pid_object.update(current_state)
	      vel.linear.x=min(1.0,temp[0])
	      vel.linear.y=min(1.0,temp[1])
	      vel.linear.z=min(1.0,temp[2])
	      
	      pub.publish(vel)
		    
    pub.publish(Twist())
    
   
    rospy.set_param('/has_active_goal',False)
    rospy.set_param('/execution_completed',True)

  ## Callback for trajectory messages
  def vel_callback(self,traj):
    rospy.loginfo("Recieved Trajectory")
    goal=self.compute_goal(traj)
    self.pid_vel(goal)
    rospy.loginfo("Completed trajectory")
    
  ## Callback for state estimate  
  def state_callback(self,data):
    self.state=data.pose.pose




if __name__ == '__main__':
  rospy.init_node('spiri_command_listener',anonymous=True)

  spiri_ros_controller(rospy.get_name())
  rospy.spin()
