#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from moveit_msgs.msg import PlanningSceneWorld
import time
from octomap_msgs.msg import Octomap
def talker(data):
    pub = rospy.Publisher('/planning_scene_world', PlanningSceneWorld, queue_size=10)
    #rospy.init_node('clear_ocotmap', anonymous=True)
    #r = rospy.Rate(10) # 10hz
    #while not rospy.is_shutdown():
    #for i in range(5):

    msg=PlanningSceneWorld()
    msg.octomap.header.stamp=rospy.Time.now()
    msg.octomap.header.frame_id='/nav'
    msg.octomap.octomap=data

    #rospy.loginfo(msg)
    pub.publish(msg)
    #time.sleep(1.0)
    #r.sleep()
def listener():
  rospy.init_node('octomap_moveit_server',anonymous=True)
  rospy.Subscriber('octomap_binary',Octomap,talker)
  rospy.spin()
if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException: pass
