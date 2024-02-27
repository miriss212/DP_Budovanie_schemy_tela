import yarp
from iCubSim import iCubLimb

yarp.Network.init()

DEFAULT_ARM_POS = (0,80,0,50,0,0,0,59,20,20,20,10,10,10,10,10)
app = '/renewer'

left_arm = iCubLimb(app,'/icubSim/left_arm')
left_arm.set(DEFAULT_ARM_POS)

right_arm = iCubLimb(app,'/icubSim/right_arm')
right_arm.set(DEFAULT_ARM_POS)
