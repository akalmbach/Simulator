#!/usr/bin/env python
import roslib; roslib.load_manifest('spiri_teleop')
import rospy
import time
from geometry_msgs.msg import Twist

import sys, select, termios, tty
from spiri_api import spiri_api_py
spiri=spiri_api_py.Staterobot()


msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
p   : move up in z direction
;   : move down in z directions
h   : Hover
n   : Land
anything else : stop

CTRL-C to quit
"""

moveBindings = {
		'i':(1,0,0),
		'o':(1,-1,0),
		'j':(0,1,0),
		'l':(0,-1,0),
		'u':(1,1,0),
		',':(-1,0,0),
		'.':(-1,1,0),
		'm':(-1,-1,0),
    		'p':(0,0,1),
                ';':(0,0,-1),
                'h':(0,0,0),
                'n':(0,0,0),
               }

speedBindings={
		'q':(1.1,1.1),
		'z':(.9,.9),
		'w':(1.1,1),
		'x':(.9,1),
		'e':(1,1.1),
		'c':(1,.9),
	      }

def getKey():
	tty.setraw(sys.stdin.fileno())
	select.select([sys.stdin], [], [], 0)
	key = sys.stdin.read(1)
	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key

speed = .5
turn = 1

def vels(speed,turn):
	return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    	settings = termios.tcgetattr(sys.stdin)

	pub = rospy.Publisher('cmd_vel', Twist)
	rospy.init_node('teleop_twist_keyboard')

	x = 0
	th = 0
	status = 0
	z=0
	
	try:
		print msg
		print vels(speed,turn)
		while(1):
			state=spiri.get_state()
			key = getKey()
			if key in moveBindings.keys():
				print key
				if key=='h':
				  
				  if state[2]<1.0:
				    twist=Twist()
				    twist.linear.z=1.0
				    pub.publish(twist)
				    time.sleep(1.0)
				    twist=Twist()
				    pub.publish(twist)
				    x=moveBindings[key][0]
				    th=moveBindings[key][1]
				    z=moveBindings[key][2]
				elif key=='n':
				  twist=Twist()
				  twist.linear.z=-state[2]
				  pub.publish(twist)
				  time.sleep(1.0)
				  twist=Twist()
				  pub.publish(twist)
				  x=moveBindings[key][0]
				  th=moveBindings[key][1]
				  z=moveBindings[key][2]
				      
				else:
				  x = moveBindings[key][0]
				  th = moveBindings[key][1]
				  #print moveBindings[key]
				  z=moveBindings[key][2]
			elif key in speedBindings.keys():
				speed = speed * speedBindings[key][0]
				turn = turn * speedBindings[key][1]

				print vels(speed,turn)
				if (status == 14):
					print msg
				status = (status + 1) % 15
			else:
				
				x = 0
				th = 0
   				z=0
				if (key == '\x03'):
					break

			twist = Twist()
			twist.linear.x = x*speed; twist.linear.y = 0; twist.linear.z = z
			twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th*turn
			pub.publish(twist)

	except:
		print 'error'

	finally:
		twist = Twist()
		twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
		twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
		pub.publish(twist)

    		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
