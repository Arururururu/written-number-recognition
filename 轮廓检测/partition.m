function [] = partition()
input = imread('input.png');
figure;imshow(input);
bw = rgb2gray(input);
%边缘/轮廓检测
t = edge(bw,'sobel');
figure;imshow(t);

%填充轮廓
im2=imfill(t,'holes');
%figure;imshow(im2);

%腐蚀处理
B=strel('disk',1);
im1=imerode(im2,B);
figure;imshow(im1);

%膨胀处理
B=strel('disk',6);
im1=imdilate(im1,B);
%figure;imshow(im1);

%保存每个线条的所在矩形区域位置
x=zeros(1,length(B));
y=zeros(1,length(B));
width=zeros(1,length(B));
height=zeros(1,length(B));
index = 1;

%轮廓检测
B = bwboundaries(im1);
figure;imshow(input); hold on;

%描绘每个线条所在的矩形区域
for k=1:length(B)
   boundary = B{k};
   %显示每个线条的轮廓，需要把上面的imshow(input)改为imshow(im1)
   %plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2);
   x(1,index)=min(boundary(:,2));
   y(1,index) = min(boundary(:,1));
   width(1,index)=max(boundary(:,2))-min(boundary(:,2));
   height(1,index)=max(boundary(:,1))-min(boundary(:,1));
   index = index+1;
   rectangle('Position',[x(1,k),y(1,k),width(1,k),height(1,k)],'edgecolor','r');
end
hold off

%根据矩形区域位置裁剪出每个线条
for k=1:length(B)
   g = imcrop(input, [x(1,k),y(1,k),width(1,k),height(1,k)]);
   figure;imshow(g);
end

end

