
#导入模块
import cv2
import numpy

#读取图像
img = cv2.imread(r'E:\Nibiru_OpenCV\POpenCV\1.jpg')
#创建窗口并显示图像
cv2.namedWindow('Image')
cv2.imshow('Image',img)
cv2.waitKey(0) # 按任意键关闭窗口，cv2.waitKey(1000) 延时一秒关闭窗口
#释放窗口
cv2.destroyAllWindows()
cv2.imwrite('2.jpg',img)

print ("total error: ", total_error/len(objpoints))
fid=open('1.txt','w')
fid.write('example03')
fid.close()
