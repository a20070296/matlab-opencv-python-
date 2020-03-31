#双目校准BM
#根据https://blog.csdn.net/Mike_slam/article/details/95375135
import cv2
import numpy as np
import glob
import time
#获取标定板角点的位置
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#board_size = [10,7]
w = 10
h = 7 
scale = 24.1
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:(w-1)*scale:complex(0,w),0:(h-1)*scale:complex(0,h)].T.reshape(-1, 2)
objpoints = [] # 存储3D点
objpoints1 = [] # 存储3D点
imgpoints = [] # 存储左侧相机2D点
imgpoints1 = [] # 存储右侧相机2D点

def disparity_SGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    param = {'minDisparity': 0,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 200,
             'speckleRange': 2,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }
 
    # 构建SGBM对象
    sgbm = cv2.StereoSGBM_create(**param)
 
    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = sgbm.compute(left_image, right_image)
        disparity_right = sgbm.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = size[0] / left_image_down.shape[1]
        disparity_left_half = sgbm.compute(left_image_down, right_image_down)
        disparity_right_half = sgbm.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA) 
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left *= factor 
        disparity_right *= factor
 
    return disparity_left, disparity_right


#左侧相机内参标定
images = glob.glob("./pic/Cam01/*.jpg")

for fname in images:
    img = cv2.imread(fname,0)
    gray = img
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #cv2.find4QuadCornerSubpix(gray,corners,(11,11))
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        #cv2.drawChessboardCorners(img, (w,h), corners, ret)
        #cv2.imshow('findCorners',img)
        #cv2.waitKey(1)
      
cv2.destroyAllWindows()
size = gray.shape[::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#print("ret:", ret)
#print("mtx:\n", mtx) # 内参数矩阵
#print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

#右侧相机内参标定
images1 = glob.glob("./pic/Cam11/*.jpg")

for fname in images1:
    img = cv2.imread(fname,0)
    gray = img
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #cv2.find4QuadCornerSubpix(gray,corners,(11,11))
        objpoints1.append(objp)
        imgpoints1.append(corners)
        # 将角点在图像上显示
#        cv2.drawChessboardCorners(img, (w,h), corners, ret)
#        cv2.imshow('findCorners',img)
#        cv2.waitKey(1)
#cv2.destroyAllWindows()

ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1, imgpoints1, gray.shape[::-1], None, None)
#print("ret1:", ret1)
#print("mtx1:\n", mtx1) # 内参数矩阵
#print("dist1:\n", dist1)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
#print("-------------------计算反向投影误差-----------------------")

#双目立体矫正及左右相机内参进一步修正
rms, C1, dist1, C2, dist2, R, T, E,F = cv2.stereoCalibrate(objpoints, imgpoints, imgpoints1, mtx, dist,mtx1, dist1,gray.shape[::-1],flags=cv2.CALIB_USE_INTRINSIC_GUESS )
#tx为左右相机距离，本例中为62mm
print('-------------------------')
print(R)
print(T)
print('-------------------------')
R1,R2,P1,P2,Q,validPixROI1,validPixROI2 = cv2.stereoRectify(C1,dist1,C2,dist2,size,R,T,alpha = -1)
left_map1, left_map2 = cv2.initUndistortRectifyMap(C1, dist1, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(C2, dist2, R2, P2, size, cv2.CV_16SC2)

frame1 = cv2.imread("./left02.jpg",0)
frame2 = cv2.imread("./right02.jpg",0)
#cv2.waitKey(10)
img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)
cv2.imshow('left', img1_rectified)
cv2.imshow('right',img2_rectified)
cv2.waitKey(10)
cv2.destroyAllWindows()
imgL = img1_rectified
imgR = img2_rectified
#num = cv2.getTrackbarPos('num', 'depth')
#blockSize = cv2.getTrackbarPos('blockSize', 'depth')
#if blockSize % 2 == 0:
#    blockSize += 1
#if blockSize < 5:
#    blockSize = 5
t1= time.time()
stereo = cv2.StereoBM_create(numDisparities=0, blockSize=11)
disparity = stereo.compute(imgL, imgR)
print('BM-time',time.time()-t1)
disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) #归一化
threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q) #此三维坐标点的基准坐标系为左侧相机坐标系
cv2.imshow('depth', disp)
cv2.waitKey(10)
cv2.destroyAllWindows()
cv2.imwrite('BM.png', disp)
#print(threeD[150:160,280:300,2]) 
print("-------------------SGBM-----------------------")
#t2= time.time()
disp1, _ = disparity_SGBM(img1_rectified, img2_rectified)
disp1 = np.divide(disp1.astype(np.float32), 16.) 
#print('SGBM-time',time.time()-t2)
disp1 = cv2.normalize(disp1, disp1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('SGBM',disp1)
cv2.waitKey(10)
cv2.destroyAllWindows()
points_3d = cv2.reprojectImageTo3D(disp1, Q)
#print(points_3d[150:160,280:300,2]) 
cv2.imwrite('SGBM.png', disp1)