import sys
sys.path.insert(0, '/Users/luizeduardocartolano/Dropbox/DUDU/Unicamp/IC/MC906/workspace/robot-fuzzy/vrep-robot-master/src')
from robot import Robot

braitenbergL=[-0.2,-0.4,-0.6,-0.8,-1.0,-1.2,-1.4,-1.6]
braitenbergR=[-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2]

detect = [0,0,0,0,0,0,0,0]
noDetectionDist = 1.0
maxDetectionDist = 0.2

def braitenberg(dist, vel):
    """
        Control the robot movement by the distances read with the ultrassonic sensors. More info: https://en.wikipedia.org/wiki/Braitenberg_vehicle
        Args:
            dist: Ultrassonic distances list
            vel:  Max wheel velocities
    """
    vLeft = vRight = vel
    for i in range(len(dist)):
        if(dist[i] < noDetectionDist):
            detect[i] = 1 - ((dist[i]-maxDetectionDist)/(noDetectionDist-maxDetectionDist))
        else:
            detect[i]=0
        for i in range(8):
            vLeft = vLeft + braitenbergL[i]*detect[i]
            vRight = vRight+ braitenbergR[i]*detect[i]

    return [vLeft, vRight]

robot = Robot()
while(robot.get_connection_status() != -1):
    us_distances = robot.read_ultrassonic_sensors()
    vel = braitenberg(us_distances[:8], 3) #Using only the 8 frontal sensors
    robot.set_left_velocity(vel[0])
    robot.set_right_velocity(vel[1])
