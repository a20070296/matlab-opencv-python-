import cv2
import numpy as np
 
#左摄像头参数
left_camera_matrix = np.array([[463.43226044 ,  0 ,   322.24293063],
                               [0, 463.11529752, 204.67667727],
                               [0, 0, 1]])
left_distortion = np.array([[ 6.30060753e-02 , 3.82822216e-01, 1.01221526e-03 , 1.20977256e-03 , -1.03750891]])
 
#右摄像头参数
right_camera_matrix = np.array([[461.88057269, 0, 306.39557947],
                                [0, 461.72807746, 197.87660275],
                                [0, 0, 1]])
right_distortion = np.array([[0.08709285 , 0.17217698  ,0.00497989 ,-0.00377567, -0.65259759]])
 
#om = np.array([0.00456, 0.01463, 0.00042])        # 旋转关系向量
#R = cv2.Rodrigues(om)[0]                           # 使用Rodrigues变换将om变换为R
R = np.array([[9.99450171e-01, -1.70623454e-04 , 3.31560917e-02],
              [-5.47867342e-05 , 9.99976896e-01 , 6.79742729e-03],
              [-3.31564855e-02 ,-6.79550639e-03 , 9.99427070e-01]])
T = np.array([-61.64677512,0.17481027,1.48614977])      # 平移关系向量
 
size = (640, 400) # 图像尺寸
 
# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
#print('roi1',P2)