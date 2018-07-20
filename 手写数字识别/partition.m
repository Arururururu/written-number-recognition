function [] = partition()
input = imread('out.png');
%figure;imshow(input);
bw = rgb2gray(input);
%边缘/轮廓检测
%t = edge(bw);
t = im2bw(bw,0.68);
%figure;imshow(t);

%膨胀处理
B=strel('disk',1);
im2 = imclose(1-t,B);
%figure;imshow(im2);


%保存每个线条的所在矩形区域位置
x=zeros(1,length(B));
y=zeros(1,length(B));
width=zeros(1,length(B));
height=zeros(1,length(B));
index = 1;

%轮廓检测
B = bwboundaries(im2,'noholes');
figure;imshow(input); hold on;

%描绘每个线条所在的矩形区域
for k=1:length(B)
   boundary = B{k};
   %显示每个线条的轮廓，需要把上面的imshow(input)改为imshow(im2)
   plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2);
   x(1,index)=min(boundary(:,2));
   y(1,index) = min(boundary(:,1));
   width(1,index)=max(boundary(:,2))-min(boundary(:,2));
   height(1,index)=max(boundary(:,1))-min(boundary(:,1));
   
   if width(1,index)*height(1,index)>100
        rectangle('Position',[x(1,index),y(1,index),width(1,index),height(1,index)],'edgecolor','r');
        index = index+1;
   end
end
hold off

%根据矩形区域位置裁剪出每个线条
for k=1:index-1
   g = imcrop(input, [x(1,k),y(1,k),width(1,k),height(1,k)]);
   %figure;imshow(g);
   c = strcat('Picture\\',num2str(k),'.png');
   g = im2bw(255-g,0.39);
   %figure;imshow(g);
   g = 1-process(g);
   %figure;imshow(g);
   imwrite(g,c);
end

end
%预处理图像，将每个数字的图像设为28*28，为了让图像比较中心对齐并保持原长宽比
%       先将图像补为正方形再缩小为22*22，然后对图像四周补0至28*28
%       由于图像较细，先对线条膨胀并闭操作，且mnist数据集中的数字比较歪
%       先将图像倾斜一定角度预测效果更好
function [output]= process(input)
    [height,width] = size(input);
    a = max(height, width);
    temp = zeros(a,a);
    for i=1:height
        for j=1:width
            temp(i,int32(j+a/2-width/2)) = input(i,j);
        end
    end
    B=strel('disk',1);
    %output=imclose(output,B);
    temp=imdilate(temp,B);
    %figure,imshow(output);
    temp=imclose(temp,B);
    %figure;imshow(temp);
    temp = imresize(temp,[22 22]);
    output = zeros(28,28);
    output(4:25,4:25) = temp(:,:);
    output = imrotate(output,-10,'bicubic','crop');
end


