#BM算法
import numpy as np
import cv2
import camera_configs
import time

t0=time.time()
frame1 = cv2.imread("./left02.jpg",0)   #黑白图像
frame2 = cv2.imread("./right02.jpg",0)

imgL = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
imgR = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

stereo = cv2.StereoBM_create(numDisparities=0, blockSize=13)
disparity = stereo.compute(imgL, imgR)

disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., camera_configs.Q)
#cv2.imwrite("BM_left.jpg", imgL)
#cv2.imwrite("BM_right.jpg",imgR)
cv2.imwrite("BM_depth.jpg",disp)
print('BMtime:',time.time()-t0)