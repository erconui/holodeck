# This is for problems with clashing opencv versions from ROS installations
import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import cv2
import numpy as np
import math
import time

from uav_sim import UAVSim
from multi_image import MultiImage

urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'

edge_min = 150
edge_max = 200

def holodeck_sim():
    uav_sim = UAVSim(urban_world)
    uav_sim.init_teleop()
    uav_sim.init_plots(plotting_freq=5) # Commenting this line would disable plotting
    # uav_sim.command_velocity = True # This tells the teleop to command velocities rather than angles

    multi_img = MultiImage(2,2)
    # feature_params = dict( maxCorners = 100,
    #                    qualityLevel = 0.3,
    #                    minDistance = 7,
    #                    blockSize = 7 )

    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    uav_sim.step_sim()
    old_frame = uav_sim.get_camera()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGBA2GRAY)
    p0 = np.empty([64*10, 1, 2])
    for i in range(10):
        for j in range(64):
            p0[i*64+j][0] = [j*8,432+i*8]
    p0 = p0.astype(np.float32)
    # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    mask = np.zeros_like(old_frame)
    while True:
        # This is the main loop where the simulation is updated
        uav_sim.step_sim()
        cam = uav_sim.get_camera()

        frame_gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1#[st==1]
        good_old = p0#[st==1]

        bottom_sum = 0
        for i, (new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            bottom_sum += abs(d-b) + abs(c-a)
        print(bottom_sum)
            # mask = cv2.line(mask,(a,b),(c,d),(0,0,0),2)
            # frame = cv2.circle(cam, (a,b), 5, (0,0,0), -1)
        # img = cv2.add(frame, mask)
        # cv2.imshow("Optical Flow", img)

        # tmp = good_new.reshape(-1,1,2)
        # if len(tmp) == 0:
        #     p0 = tmp
        # print(p0)
        # I run my opencv stuff here
        # gray = cv2.cvtColor(cam, cv2.COLOR_RGBA2GRAY)
        # edge = cv2.Canny(cam, edge_min, edge_max)
        # bgr = cv2.cvtColor(cam, cv2.COLOR_RGBA2BGR)
        # hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # You can also give an external command and switch between automatic
        # commands given here and manual commands from the keyboard using the
        # key mapped to MANUAL_TOGGLE in uav_sim.py

        # In automatic mode, fly forward at 3m altitude at current heading
        yaw_c = uav_sim.yaw_c
        uav_sim.command_velocity(vx=2.0, vy=0.0, yaw=yaw_c, alt=3.0)

        # This is just a useful class for viewing multiple filters in one image
        # multi_img.add_image(cam, 0,0)
        # multi_img.add_image(gray, 0,1)
        # multi_img.add_image(edge, 1,0)
        # multi_img.add_image(hsv, 1,1)
        # display = multi_img.get_display()
        # cv2.imshow('Holodeck', display)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    holodeck_sim()
    print("Finished")
