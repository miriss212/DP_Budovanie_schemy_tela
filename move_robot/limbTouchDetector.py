import yarp

class LimbTouchDetector:
    def __init__(self):

        self.opened_ports = {}

        self.remote_touch_ports = \
            {
                'left-hand' : '/icubSim/skin/left_hand_comp',
                'right-hand' : '/icubSim/skin/right_hand_comp',
                'left-forearm' : '/icubSim/skin/left_forearm_comp',
                'right-forearm' : '/icubSim/skin/right_forearm_comp',

            }

        self.local_touch_ports = \
            {
                'left-hand': '/test/skin/touch/left_hand',
                'right-hand': '/test/skin/touch/right_hand',
                'left-forearm': '/test/skin/touch/left_forearm',
                'right-forearm': '/test/skin/touch/right_forearm'
            }

        self.init_touch_detection_ports()

    def init_touch_detection_ports(self):

        for key in self.local_touch_ports.keys():
            remote_port = self.remote_touch_ports[key]
            local_port = self.local_touch_ports[key]

            yarp_port = yarp.Port()
            yarp_port.open(local_port)
            yarp.Network.connect(remote_port, local_port)

            self.opened_ports[key] = yarp_port

    def close_touch_ports(self):
        for port in self.opened_ports.values():
            port.close()

    def read_touch_sensors(self, limb_part): #limbPart = string representing which limb part to read

        try:
            # remotePort = self.remoteTouchPorts[limbPart]
            # localPort = self.localTouchPorts[limbPart]
            #
            # port.open(localPort)
            #
            # yarp.Network.connect(remotePort, localPort)
            port = self.opened_ports[limb_part]
            value = self.get_bottle_data(port)
            return value

        except Exception:
            print("ERROR during limb yarp port read. Shouldn't have happened")

    def get_bottle_data(self, port):
        bottle = yarp.Bottle()
        #print("READING TO BOTTLE ")
        port.read(bottle)
        #print("DISCONNETIN")
        # yarp.Network.disconnect(remotePort, localPort)
        size = bottle.size()
        #print("Size of bottle: ", size)
        value = []
        for i in range(size):
            value.append(bottle.get(i).asDouble())

        print("Value: {0}".format(sum(value)))
        return value
