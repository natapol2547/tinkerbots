"""my_summit_controller controller."""
import numpy as np
from controller import Robot, Motor, Camera
from src.robot_func import *
from src.image import (
    gaussian_blur,
    read_image,
    write_image,
    resize_image,
    sobel_filter,
    anisotropic_gaussian_kernel,
    apply_equivalent_filter,
    rgb_to_gray
)
from src.optical_flow import optical_flow_vector, optical_flow_magnitude, plot_optical_flow, optical_flow_pyramid
import time

inf = float('inf')

robot = Robot()

#Wheels setup
fl_wheel = robot.getDevice("front_left_wheel_joint")
fr_wheel = robot.getDevice("front_right_wheel_joint")
bl_wheel = robot.getDevice("back_left_wheel_joint")
br_wheel = robot.getDevice("back_right_wheel_joint")
wheels = np.array([fl_wheel, fr_wheel, bl_wheel, br_wheel])
if None not in wheels:
    print("All wheels are found")
else:
    raise Exception("Not all wheels are found")

#Camera setup
rgb_camera = robot.getDevice("rgb_camera")
depth_camera = robot.getDevice("depth_camera")
if (rgb_camera is not None) and (depth_camera is not None):
    print("Cameras are found")
    fps = 30
    rgb_camera.enable(fps_to_samplingse_rate(fps))
    depth_camera.enable(fps_to_samplingse_rate(fps))
    print("Camera enabled with sampling rate: {} ms".format(fps_to_samplingse_rate(fps)), "|| FPS rate: {}".format(fps) )
    print("RGB Camera resolution: {}x{}".format(rgb_camera.getWidth(), rgb_camera.getHeight()))
    print("Depth Camera resolution: {}x{}".format(depth_camera.getWidth(), depth_camera.getHeight()))
else:    
    raise Exception("Camera is not found")



#for velocity control, set position to inf
set_position(wheels, [inf, inf, inf, inf])
set_velocity(wheels, [0.0, 0.0, 0.0, 0.0])

#initialize filter parameters
gaussian_wight_filter = anisotropic_gaussian_kernel((depth_camera.getHeight(), depth_camera.getWidth()))
write_image("anisotropic_gaussian_filter.jpg", gaussian_wight_filter / np.max(gaussian_wight_filter) * 255)


timestep = int(robot.getBasicTimeStep())
image_count = 0
simTime = 0

while robot.step(timestep) != -1:

    moveL(wheels, 0.5)
    #get camera images and save them
    rgb_camera.saveImage("RGB_camera_outputs/rgb_camera_image_{}.jpg".format(image_count), 100)
    depth_camera.saveImage("Depth_camera_outputs/depth_camera_image_{}.jpg".format(image_count), 100)

    rgb_image = read_image("RGB_camera_outputs/rgb_camera_image_{}.jpg".format(image_count))
    depth_image = read_image("Depth_camera_outputs/depth_camera_image_{}.jpg".format(image_count))
    image_count += 1

    depth_image = rgb_to_gray(depth_image)
    blurred_img = apply_equivalent_filter(depth_image, gaussian_wight_filter)
    write_image("Blurred_depth_images/blurred_depth_image_{}.jpg".format(image_count-1), blurred_img)

    simTime += timestep / 1000.0
    pass

# Enter here exit cleanup code.
