#SGBM算法
import numpy as np
import cv2
import camera_configs
import time

t0=time.time()
frame1 = cv2.imread("./left02.jpg",0)   #黑白图像
frame2 = cv2.imread("./right02.jpg",0)
imgL = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
imgR = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

# disparity range tuning
window_size = 3
#min_disp = 0
#num_disp = 320 - min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=3,
    P1=8 * 1 * window_size ** 2,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 1 * window_size ** 2,
    disp12MaxDiff=1,
    preFilterCap=63,
    uniquenessRatio=15,
    speckleWindowSize=200,
    speckleRange=2,    
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
#disparity  = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
disparity_left = stereo.compute(imgL, imgR)
disparity_left = np.divide(disparity_left.astype(np.float32), 16.) 
disparity_left = cv2.normalize(disparity_left, disparity_left, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
points_3d = cv2.reprojectImageTo3D(disparity_left,camera_configs.Q)
cv2.imwrite('SGBM.png', disparity_left)
print('SGBMtime:',time.time()-t0)