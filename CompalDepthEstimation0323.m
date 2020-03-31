
%Load stereo parameters.
load('Compal2Camera.mat');
%Read in the stereo pair of images.
I01 = imread('left-02.jpg');
I02 = imread('right-02.jpg');
%Undistort the images.
%I1 = undistortImage(I01,stereoParams.CameraParameters1);
%I2 = undistortImage(I02,stereoParams.CameraParameters2);

[J1, J2] = rectifyStereoImages(I01,I02,stereoParams,'OutputView','valid');

disparityMap = disparity(J1, J2,'BlockSize',11);%
figure 
imshow(disparityMap,[0,64],'InitialMagnification',50);

xyzPoints = reconstructScene(disparityMap,stereoParams);

Z = xyzPoints(:,:,3);
mask = repmat(Z ~= 0 ,[1,1,1]); %显示400mm-600mm范围内图像
J11=J1;
J11(~mask) = 0;
figure;
imshow(J11,'InitialMagnification',50);

centroids = [296 186];
centroidsIdx = sub2ind(size(disparityMap), centroids(:, 2), centroids(:, 1));
dists = Z(centroidsIdx);
position = [centroids 16];%设置圆位置。前两个值表示中心位于（x，y），第三个值表示半径。
label = sprintf('%0.0f mm', dists);      %设置标签显示数据。
RGB = insertObjectAnnotation(J11,'circle',position,label,'Color',{'cyan'});      %插入的批注。
imshow(RGB)
