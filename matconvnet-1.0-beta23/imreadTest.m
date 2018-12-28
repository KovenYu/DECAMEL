clear
clc
rand('seed', 1);
f = rand(3,3,3,4, 'single');
img = imread('mini.jpg');
% imshow(img)
% figure;
% path = cellstr('/home/hongxing/matconvnet-1.0-beta23/mini.jpg');
% img1 = vl_imreadjpeg(path, 'Resize', [400, 400], 'CropAnisotropy', [1,1], ...
%     'CropSize', [0.5, 0.5], 'CropLocation', 'random');
% imshow(uint8(img1{1}))

img = single(img);
y = vl_nnconv(img, f, [], 'stride', 1, 'pad', 1);
y2 = vl_nnconvt(y, f, [], 'upsample', 1, 'crop', 1);