function [cb_trans] = Rectification( input_img )
%转为灰度图
gray = rgb2gray(input_img);
%边缘提取
bw = edge(gray,'sobel');
figure;imshow(bw);
%霍夫变换
[H,Theta,Rho] = hough(bw);
%输出霍夫变换结果
figure;imshow(H,[],'XData',Theta,'YData',Rho,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
%霍夫变换峰值，四条直线四个峰值点
P  = houghpeaks(H,4,'threshold',ceil(0.2*max(H(:))));
x = Theta(P(:,2));
y = Rho(P(:,1));
%标出四个霍夫变换峰值点
plot(x,y,'*','color','r');
%根据霍夫变换和峰值点寻找和连接线段
lines = houghlines(bw,Theta,Rho,P,'FillGap',100,'MinLength',80);
figure, imshow(input_img), hold on
%记录四个峰值点在原图上的坐标表示
points = zeros(4,2);
index = 1;
%在原图上标出四条直线
for k = 1:length(lines)
    %每条直线的起始点和终止点
    xy = [lines(k).point1; lines(k).point2];

    if index < 5
        points(index,1) = xy(1,1);
        points(index,2) = xy(1,2);
        index = index+1;
    end
    %在原图标出四个顶点
    %plot(xy(1,1),xy(1,2),'x','Color','r');
    if index < 5 
        points(index,1) = xy(2,1);
        points(index,2) = xy(2,2);
        index = index+1; 
    end
    %在原图标出每条线段
    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','r');
end
display(points);
target = [1 1;1 960;540 1;540 960];
%根据原图的四个顶点找出透视变换的矩阵
TForm = cp2tform(points,target,'projective');
%根据转换矩阵转换原图
[cb_trans xdata ydata] = imtransform(input_img, TForm,'FillValues', 255);
%输出透视变换后的图片
cb_trans=imresize(cb_trans,0.5);
gray = rgb2gray(cb_trans);
%边缘提取
bw = edge(gray,'sobel');
%霍夫变换
[H,Theta,Rho] = hough(bw);
P  = houghpeaks(H,4,'threshold',ceil(0.2*max(H(:))));
%根据霍夫变换和峰值点寻找和连接线段
lines = houghlines(bw,Theta,Rho,P,'FillGap',20,'MinLength',50);
figure, imshow(cb_trans), hold on
%记录四个峰值点在原图上的坐标表示
points1 = zeros(4,2);
index = 1;
%在原图上标出四条直线
for k = 1:length(lines)
    %每条直线的起始点和终止点
    xy = [lines(k).point1; lines(k).point2];

    if index<2
        points1(index,1) = xy(1,1);
        points1(index,2) = xy(1,2);
        index = index+1;
        points1(index,1) = xy(2,1);
        points1(index,2) = xy(2,2);
        index = index+1;
    end
    if k==4
        points1(index,1) = xy(2,1);
        points1(index,2) = xy(2,2);
        index = index+1;
    end
end
points1(4,1)=points1(3,1);
points1(4,2)=points1(1,2);
plot(points1(1,1),points1(1,2),'x','Color','r');
plot(points1(2,1),points1(2,2),'x','Color','r');
plot(points1(3,1),points1(3,2),'x','Color','r');
plot(points1(4,1),points1(4,2),'x','Color','r');

hold off
%根据四个顶点剪切图像
g = imcrop(cb_trans, [points1(1,1)+1,points1(1,2)+3,points1(3,1)-points1(1,1)-4,points1(3,2)-points1(1,2)-8]);
g=imresize(g,1.5);
imwrite(g,'out.png');
figure;imshow(g);
end

