close all;
%% 打开文件
path = 'data/SAR_Airplane_Recognition_trainData/trainData/';
img_path = [path,'Images/','47.tif'];
img = imread(img_path);
%% 时域处理
x = double(img)/255;
x_light = x+0.5; 
x_med = medfilt2(x,[5,5]);
x_hist = histeq(x);
xx = (x - mean(x,'all'))/std(x(:));
figure(1)
subplot(2,2,1)
imshow(img)
subplot(2,2,2)
imshow(x_light)
subplot(2,2,3)
imshow(xx)
subplot(2,2,4)
imshow(x_hist)
figure(2)
histogram(double(img));
%% 观察频域
x_fft = fft2(x);
x_fft_abs = fftshift(abs(x_fft));
x_fft_angle = angle(x_fft);

figure(3)
subplot(1,2,1)
imshow(0.01*x_fft_abs)
subplot(1,2,2)
imshow((x_fft_angle+pi)/(2*pi))

%% 频域处理





