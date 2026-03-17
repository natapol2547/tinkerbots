from controller import Motor, Camera
import numpy as np

WHEEL_RADIUS = 0.123
LX = 0.2045  # lateral distance from robot's COM to wheel [m].
LY = 0.2225  # longitudinal distance from robot's COM to wheel [m].

def fps_to_samplingse_rate(fps):
    if fps <= 0:
        raise ValueError("FPS must be greater than 0")
    return int(1000 / fps)

def set_velocity(wheels, velocity=0.1):
    if len(wheels) != len(velocity):
        raise ValueError("Length of wheels and velocity must be the same")
    
    angular_velocity = [v / WHEEL_RADIUS for v in velocity]
    for wheel, vel in zip(wheels, angular_velocity):
        Motor.setVelocity(wheel, vel)

def set_position(wheels, position):
    if len(wheels) != len(position):
        raise ValueError("Length of wheels and position must be the same")

    for wheel, pos in zip(wheels, position):
        Motor.setPosition(wheel, pos)

def stop_robot(wheels):
    set_velocity(wheels, [0.0] * len(wheels))

def moveL(wheels, velocity):
    set_velocity(wheels, [velocity] * len(wheels))

def rotate(wheels, rps):
    gain_factor = 4
    angular_velocity = np.array([- rps * LX / WHEEL_RADIUS, rps * LX / WHEEL_RADIUS, - rps * LX / WHEEL_RADIUS, rps * LX / WHEEL_RADIUS]) * gain_factor
    for wheel, vel in zip(wheels, angular_velocity):
        Motor.setVelocity(wheel, vel)


