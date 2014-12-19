#! /usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, PoseStamped,Pose
class PID:
	"""
	Discrete PID control
	"""

	def __init__(self, P=[2.0,2.0,2.0], I=[0.0,0.0,0.0], D=[1.0,1.0,1.0], Derivator=[0.0,0.0,0.0], Integrator=0, Integrator_max=500, Integrator_min=-500):

		self.Kp=P

		self.Ki=I
		self.Kd=D
		self.Derivator=Derivator
		self.Integrator=Integrator
		self.Integrator_max=Integrator_max
		self.Integrator_min=Integrator_min

		#self.set_point=0.0
		self.set_point=Pose()
		#self.error=[]

	def update(self,current_value):
		"""
		Calculate PID output value for given reference input and feedback
		"""
		self.error=[]
		self.error.append(self.set_point.x - current_value.x)
		self.error.append(self.set_point.y - current_value.y)
		self.error.append(self.set_point.z - current_value.z)
		#print self.error
		PID=[]
		for i in range(3):
			#print type(self.Kp)
			#print type(self.error)

			self.P_value = self.Kp[i] * float(self.error[i])
			#print type(self.Derivator)
			#print 'error',self.Derivator[i]
			self.D_value = self.Kd[i] * (self.error[i] - self.Derivator[i])
			self.Derivator[i] = self.error[i]
			'''
			self.Integrator = self.Integrator + dt * self.error

			if self.Integrator > self.Integrator_max:
				self.Integrator = self.Integrator_max
			elif self.Integrator < self.Integrator_min:
				self.Integrator = self.Integrator_min

			self.I_value = self.Integrator * self.Ki[i]
			'''
			#PID = self.P_value + self.I_value + self.D_value
			PID.append(self.P_value + self.D_value)
		return PID
		'''
		self.error = self.set_point - current_value

		self.P_value = self.Kp * self.error
		self.D_value = self.Kd * ( self.error - self.Derivator)
		self.Derivator = self.error

		self.Integrator = self.Integrator + dt * self.error

		if self.Integrator > self.Integrator_max:
			self.Integrator = self.Integrator_max
		elif self.Integrator < self.Integrator_min:
			self.Integrator = self.Integrator_min

		self.I_value = self.Integrator * self.Ki

		#PID = self.P_value + self.I_value + self.D_value
		PID = self.P_value + self.D_value
		return PID
		'''
	def setPoint(self,set_point):
		"""
		Initilize the setpoint of PID
		"""
		self.set_point = set_point
		self.Integrator=0
		self.Derivator=[0.0,0.0,0.0]
	'''
	def setIntegrator(self, Integrator):
		self.Integrator = Integrator

	def setDerivator(self, Derivator):
		self.Derivator = Derivator

	def setKp(self,P):
		self.Kp=P

	def setKi(self,I):
		self.Ki=I

	def setKd(self,D):
		self.Kd=D

	def getPoint(self):
		return self.set_point

	def getError(self):
		return self.error

	def getIntegrator(self):
		return self.Integrator

	def getDerivator(self):
		return self.Derivator
	'''
