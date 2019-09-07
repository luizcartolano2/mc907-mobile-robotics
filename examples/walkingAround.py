import sys, time
sys.path.insert(0, '../src')
from robot import Robot

robot = Robot()

for i in range(3):
    robot.set_left_velocity(2.0) #rad/s
    robot.set_right_velocity(2.0)
    time.sleep(10) #Go foward for 10 seconds!

    robot.stop()
    time.sleep(0.5) #Stop for half second

    robot.set_velocity(0, 1.5)
    time.sleep(1) #Turning left for 1 second

    robot.stop()
    time.sleep(0.5) #Stop for half second

    robot.set_velocity(0.1, -0.1)
    time.sleep(5) #Moving forward and to the right for 5 seconds

    robot.stop()
    time.sleep(0.5) #Stop for half second

    robot.set_velocity(-0.1, -0.1)
    time.sleep(5) #Moving backwards and to the right for 5 seconds

    robot.stop()
    time.sleep(0.5) #Stop for half second
