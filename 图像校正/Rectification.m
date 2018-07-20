function [cb_trans] = Rectification( input_img )
%תΪ�Ҷ�ͼ
gray = rgb2gray(input_img);
%��Ե��ȡ
bw = edge(gray,'sobel');
figure;imshow(bw);
%����任
[H,Theta,Rho] = hough(bw);
%�������任���
figure;imshow(H,[],'XData',Theta,'YData',Rho,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
%����任��ֵ������ֱ���ĸ���ֵ��
P  = houghpeaks(H,4,'threshold',ceil(0.2*max(H(:))));
x = Theta(P(:,2));
y = Rho(P(:,1));
%����ĸ�����任��ֵ��
plot(x,y,'*','color','r');
%���ݻ���任�ͷ�ֵ��Ѱ�Һ������߶�
lines = houghlines(bw,Theta,Rho,P,'FillGap',100,'MinLength',80);
figure, imshow(input_img), hold on
%��¼�ĸ���ֵ����ԭͼ�ϵ������ʾ
points = zeros(4,2);
index = 1;
%��ԭͼ�ϱ������ֱ��
for k = 1:length(lines)
    %ÿ��ֱ�ߵ���ʼ�����ֹ��
    xy = [lines(k).point1; lines(k).point2];

    if index < 5
        points(index,1) = xy(1,1);
        points(index,2) = xy(1,2);
        index = index+1;
    end
    %��ԭͼ����ĸ�����
    %plot(xy(1,1),xy(1,2),'x','Color','r');
    if index < 5 
        points(index,1) = xy(2,1);
        points(index,2) = xy(2,2);
        index = index+1; 
    end
    %��ԭͼ���ÿ���߶�
    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','r');
end
display(points);
target = [1 1;1 960;540 1;540 960];
%����ԭͼ���ĸ������ҳ�͸�ӱ任�ľ���
TForm = cp2tform(points,target,'projective');
%����ת������ת��ԭͼ
[cb_trans xdata ydata] = imtransform(input_img, TForm,'FillValues', 255);
%���͸�ӱ任���ͼƬ
cb_trans=imresize(cb_trans,0.5);
gray = rgb2gray(cb_trans);
%��Ե��ȡ
bw = edge(gray,'sobel');
%����任
[H,Theta,Rho] = hough(bw);
P  = houghpeaks(H,4,'threshold',ceil(0.2*max(H(:))));
%���ݻ���任�ͷ�ֵ��Ѱ�Һ������߶�
lines = houghlines(bw,Theta,Rho,P,'FillGap',20,'MinLength',50);
figure, imshow(cb_trans), hold on
%��¼�ĸ���ֵ����ԭͼ�ϵ������ʾ
points1 = zeros(4,2);
index = 1;
%��ԭͼ�ϱ������ֱ��
for k = 1:length(lines)
    %ÿ��ֱ�ߵ���ʼ�����ֹ��
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
%�����ĸ��������ͼ��
g = imcrop(cb_trans, [points1(1,1)+1,points1(1,2)+3,points1(3,1)-points1(1,1)-4,points1(3,2)-points1(1,2)-8]);
g=imresize(g,1.5);
imwrite(g,'out.png');
figure;imshow(g);
end

