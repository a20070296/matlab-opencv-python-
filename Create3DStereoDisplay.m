% Create 3-D Stereo Display

%Load parameters for a calibrated stereo pair of cameras.
load('webcamsSceneReconstruction.mat')
%Load a stereo pair of images.
I1 = imread('sceneReconstructionLeft.jpg');
I2 = imread('sceneReconstructionRight.jpg');%Rectify the stereo images.
[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);
%Create the anaglyph.
A = stereoAnaglyph(J1, J2);
%Display the anaglyph. Use red-blue stereo glasses to see the stereo effect.
figure; imshow(A);
