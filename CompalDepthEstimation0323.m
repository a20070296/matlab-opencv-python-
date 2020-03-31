
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
mask = repmat(Z ~= 0 ,[1,1,1]); %��ʾ400mm-600mm��Χ��ͼ��
J11=J1;
J11(~mask) = 0;
figure;
imshow(J11,'InitialMagnification',50);

centroids = [296 186];
centroidsIdx = sub2ind(size(disparityMap), centroids(:, 2), centroids(:, 1));
dists = Z(centroidsIdx);
position = [centroids 16];%����Բλ�á�ǰ����ֵ��ʾ����λ�ڣ�x��y����������ֵ��ʾ�뾶��
label = sprintf('%0.0f mm', dists);      %���ñ�ǩ��ʾ���ݡ�
RGB = insertObjectAnnotation(J11,'circle',position,label,'Color',{'cyan'});      %�������ע��
imshow(RGB)
