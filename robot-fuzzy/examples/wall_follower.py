import sys
from pdb import set_trace
sys.path.insert(0, '/Users/luizeduardocartolano/Dropbox/DUDU/Unicamp/IC/MC906/workspace/robot-fuzzy/vrep-robot-master/src')
from robot import Robot


def find_wall(robot):
    """
        Funcao para encontrar uma parede a ser seguida
        :param: robot - objeto robo
    """
    us_distances = robot.read_ultrassonic_sensors()
    sonar_3 = us_distances[3]
    sonar_4 = us_distances[4]
    robot.set_left_velocity(6.0)
    robot.set_right_velocity(0.0)
    # robot.set_velocity(1.5,0.0)
    while sonar_3 >= 0.5 and sonar_4 >= 0.5:
        us_distances = robot.read_ultrassonic_sensors()
        sonar_3 = us_distances[3]
        sonar_4 = us_distances[4]

    robot.stop()

    return True


def align_wall(robot):
    us_distances = robot.read_ultrassonic_sensors()
    sonar_0 = us_distances[0]
    sonar_15 = us_distances[-1]
    robot.set_velocity(1.5, -1.5)

    while (sonar_0 > 0.45 and sonar_15 > 0.45) or (sonar_0 - sonar_15 >= 0.15) or (sonar_15 - sonar_0 >= 0.15):
        us_distances = robot.read_ultrassonic_sensors()
        sonar_0 = us_distances[0]
        sonar_15 = us_distances[-1]

    robot.stop()
    set_trace()

if __name__ == '__main__':

    robot = Robot()
    while(robot.get_connection_status() != -1):
        find_wall(robot)
        # set_trace()
        align_wall(robot)
        set_trace()
