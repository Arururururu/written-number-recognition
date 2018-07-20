function [] = Rectification()
%转为灰度图
input_img = imread('input.jpg');
%figure;imshow(input_img);
gray = rgb2gray(input_img);
%边缘提取
bw = edge(gray,'sobel');
%figure;imshow(bw);
%霍夫变换
[H,Theta,Rho] = hough(bw);
%霍夫变换峰值，四条直线四个峰值点
P  = houghpeaks(H,4,'threshold',ceil(0.2*max(H(:))));
%根据霍夫变换和峰值点寻找和连接线段
lines = houghlines(bw,Theta,Rho,P,'FillGap',250,'MinLength',10);
figure, imshow(input_img),hold on
%记录四个峰值点在原图上的坐标表示
points = zeros(4,2);
index = 1;
for k = 1:length(lines)
    %每条直线的起始点和终止点
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','r');
    if index < 5
        points(index,1) = xy(1,1);
        points(index,2) = xy(1,2);
        index = index+1;
    end
    if index < 5 
        points(index,1) = xy(2,1);
        points(index,2) = xy(2,2);
        index = index+1; 
    end
end

%display(points);
hold off
target = [1 1;1 720;540 1;540 720];
%根据原图的四个顶点找出透视变换的矩阵
TForm = cp2tform(points,target,'projective');
%根据转换矩阵转换原图
[bw_trans, xdata, ydata] = imtransform(bw, TForm,'FillValues', 0);
[cb_trans, xdata1, ydata1] = imtransform(input_img, TForm,'FillValues', 0);
%霍夫变换
[H,Theta,Rho] = hough(bw_trans);
P  = houghpeaks(H,4,'threshold',ceil(0.2*max(H(:))));
%根据霍夫变换和峰值点寻找和连接线段
lines = houghlines(bw_trans,Theta,Rho,P,'FillGap',80,'MinLength',1);

%记录四个峰值点在原图上的坐标表示
points1 = zeros(4,2);
index = 1;
%在原图上标出四条直线
for k = 1:length(lines)
    %每条直线的起始点和终止点
    xy = [lines(k).point1; lines(k).point2];
    %plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','r');
    lines(k).point1
    lines(k).point2
    if index < 5
        points1(index,1) = xy(1,1);
        points1(index,2) = xy(1,2);
        index = index+1;
    end
    if index < 5 
        points1(index,1) = xy(2,1);
        points1(index,2) = xy(2,2);
        index = index+1; 
    end
end
%display(points1);

%根据四个顶点剪切图像
g = imcrop(cb_trans, [points1(1,1)+2,points1(1,2)+3,points1(3,1)-points1(1,1)-3,points1(2,2)-points1(1,2)-30]);
figure;imshow(g);
imwrite(g,'out.png');

end

