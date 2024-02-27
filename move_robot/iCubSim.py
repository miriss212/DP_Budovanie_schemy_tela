import yarp
import time
import math
import random

class iCubLimb:
    def __init__(self,app_name,port_name):
        # prepare a property object
        self.props = yarp.Property()
        self.props.put('device','remote_controlboard')
        self.props.put('local',app_name+port_name)
        self.props.put('remote',port_name)
        # create remote driver
        self.armDriver = yarp.PolyDriver(self.props)
        # query motor control interfaces
        self.iPos = self.armDriver.viewIPositionControl()
        #self.iVel = self.armDriver.viewIVelocityControl()
        self.iEnc = self.armDriver.viewIEncoders()
        # retrieve number of joints
        self.jnts = self.iPos.getAxes()
        print('Controlling', self.jnts, 'joints of', port_name)

        # ========== ATTRIBUTES RELEVANT TO BABBLING. REFACTOR THIS LATER =========

        # ----- JOINT INDICES INFO -----
        #   0: moves shoulder/arm front/backwards
        #   1: moves shoulder/arm toward/away from torso
        #   2: moves elbow/arm toward/away from torso
        #   3: moves elbow/arm front/back
        #   4: rotates arm from elbow down
        #   5: wrist movement up/down
        #   6: wrist movement sideways


        #these values are used during babbling
        self.joints_minIntervals = \
            {
                0: -50,          1: 0,
                2:20,            3: 70,
                4: 12,          5: -20,
                6: -19,
            }

        self.joints_maxIntervals = \
            {
                0: -30,         1: 50,
                2: 74,           3: 105,
                4: 12,         5: -4,
                6: 13,
            }

        # self.start position of arm for babbling
        self.start_command_arm = [0 for i in range(16)]
        self.start_command_arm[0] = -45.0; self.start_command_arm[1] = 0; self.start_command_arm[2] = 0.0;
        self.start_command_arm[3] = 50.0; self.start_command_arm[4] = 5.0; self.start_command_arm[5] = 0.0;
        self.start_command_arm[6] = 0.0; self.start_command_arm[7] = 0.0; self.start_command_arm[8] = 10.0;
        self.start_command_arm[9] = 0.0; self.start_command_arm[10] = 0.0; self.start_command_arm[11] = 0.0;
        self.start_command_arm[12] = 0.0; self.start_command_arm[13] = 0.0; self.start_command_arm[14] = 0.0;
        self.start_command_arm[15] = 0.0;

        for i in range(7):
            self.start_command_arm[i] = self.joints_minIntervals[i]

        self.amp = 4.0
        self.freq = 0.2

    def moveByAngle(self):
        #retrieve number of joints
        self.jnts=self.iPos.getAxes()
        print ('Controlling', self.jnts, 'joints')
        # read encoders
        encs=yarp.Vector(self.jnts)
        self.iEnc.getEncoders(encs.data())
        # store as home position
        home=yarp.Vector(self.jnts, encs.data())

        values = self.get()

        self.set(values, j0=values[0]+10, j1=values[1]+10)

        #and go back
        time.sleep(1)
        self.iPos.positionMove(home.data())

                
    def get(self):
        # read encoders
        encs = yarp.Vector(self.jnts)
        self.iEnc.getEncoders(encs.data())
        values = ()
        for i in range(self.jnts):
            values += (encs.get(i),)
        print(values)
        return values
        
    def set(self,values=(), \
        j0=None,j1=None,j2=None,j3=None,j4=None,j5=None,j6=None,j7=None, \
        j8=None,j9=None,j10=None,j11=None,j12=None,j13=None,j14=None,j15=None):
        # read encoders
        encs = yarp.Vector(self.jnts)
        self.iEnc.getEncoders(encs.data())
        # adjust joint positions
        for i in range(min(self.jnts,len(values))):
            if values[i] != None:
                encs.set(i,values[i])
        for i in range(16):
            value = eval('j'+str(i))
            if value != None:
                #print('j',i,'=',value)
                encs.set(i,value)
        # write to motors
        self.iPos.positionMove(encs.data())

    def doBabbling(self):
        
        #set the arm to start_pos_arm joints
        values = self.get()
        self.set(values, j0=self.start_command_arm[0], j1=self.start_command_arm[1], j2=self.start_command_arm[2],
                 j3=self.start_command_arm[3], j4=self.start_command_arm[4], j5=self.start_command_arm[5],
                 j6=self.start_command_arm[6], j7=self.start_command_arm[7], j8=self.start_command_arm[8],
                 j9=self.start_command_arm[9], j10=self.start_command_arm[10], j11=self.start_command_arm[11],
                 j12=self.start_command_arm[12], j13=self.start_command_arm[13], j14=self.start_command_arm[14],
                 j15=self.start_command_arm[15])

        durationSeconds = 20

        startTime = time.time()*1000.0

        print("============> babbling with COMMAND will START");
        print("AMP ", self.amp, "FREQ ", self.freq);

        impl = "new"

        # now loop and issue the actual babbling commands
        while time.time()*1000.0 < startTime + durationSeconds*1000:
            t = time.time()*1000.0 - startTime
            print("t = ", t, "/", durationSeconds)
            if impl == "old":
                self.babblingCommands(t)

            elif impl == "new":
                self.simplifiedBabbling(t)

        print("============> babbling with COMMAND is FINISHED");
        return True

    def simplifiedBabbling(self, t):

        moveBy = [5, 8, 10, 10, 12, 12, 15, 20, 30]
        ranges = \
        {
            0: [-35, -15], 1: [0, 60],
            2: [40, 50], 3:[85, 105],
            4: [-10, 30],
            5: [-30, 0],
            6: [-15, 15],
            7: [40, 57]
        }
        command = [ranges[i][0] for i in range(len(ranges))]  # command after correction
        # set the new joints
        values = self.get()
        self.set(values, j0=command[0], j1=command[1], j2=command[2],
                 j3=command[3], j4=command[4], j5=command[5], j6=command[6])

        #arm move: only move len(ranges) joints, in selected ranges, by a moveBy value
        for i in range(0, len(ranges)):
            # 50/50 chance to move either way, by either 10 or 20 degrees (except joint1 which moves by all movement always)

            commandMin = ranges[i][0]
            commandMax = ranges[i][1]

            direction = random.randrange(-1,2,2) #gives -1 or 1
            degrees = random.randint(0, len(moveBy)-1)
            howMuch = moveBy[degrees]
            if i == 1:
                howMuch = commandMax

            command[i] = command[i] + (howMuch * direction)

            if command[i] > commandMax: command[i] = commandMax
            if command[i] < commandMin: command[i] = commandMin

        # set the new joints
        values = self.get()
        self.set(values, j0=command[0], j1=command[1], j2=command[2],
                 j3=command[3], j4=command[4], j5=command[5], j6=command[6])

    def babblingCommands(self, t):

        MAX_COORD = 50
        MIN_COORD = -50

        ref_command = [0 for i in range(16)] #16 joints of the arm
        command = [0 for i in range(16)] #command after correction
        encoders_used = []

        #time-relevant joint change
        for i in range(16):
            ref_command[i] = self.start_command_arm[i] + self.amp * math.sin(self.freq * t * 2 * math.pi)


        print("Sin is " , math.sin(self.freq * t * 2 * math.pi))
        encoders_used = yarp.Vector(self.jnts)
        self.iEnc.getEncoders(encoders_used.data())

        move_hand = False
        if move_hand:
            for i in range(7, len(command)):
                command[i] = 10 * (ref_command[i] - encoders_used[i])
                if command[i] > MAX_COORD: command[i] = MAX_COORD
                if command[i] < MIN_COORD: command[i] = MIN_COORD

            #set the new joints
            values = self.get()
            self.set(values, j7=command[7], j8=command[8], j9=command[9],
                     j10=command[10], j11=command[11], j12=command[12],
                     j13=command[13], j14=command[14], j15=command[15])

        else: #move arm only
            for i in range(0, 7):
                command[i] = 10 * (ref_command[i] - encoders_used[i])
                if command[i] > 50: command[i] = 50
                if command[i] < -50: command[i] = -50

            # set the new joints
            values = self.get()
            self.set(values, j0=command[0], j1=command[1], j2=command[2],
                     j3=command[3], j4=command[4], j5=command[5],
                     j6=command[6])

        time.sleep(0.5)

    def size(self):
        # return number of joints
        return self.jnts
