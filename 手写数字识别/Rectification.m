function [] = Rectification()
%תΪ�Ҷ�ͼ
input_img = imread('input.jpg');
%figure;imshow(input_img);
gray = rgb2gray(input_img);
%��Ե��ȡ
bw = edge(gray,'sobel');
%figure;imshow(bw);
%����任
[H,Theta,Rho] = hough(bw);
%����任��ֵ������ֱ���ĸ���ֵ��
P  = houghpeaks(H,4,'threshold',ceil(0.2*max(H(:))));
%���ݻ���任�ͷ�ֵ��Ѱ�Һ������߶�
lines = houghlines(bw,Theta,Rho,P,'FillGap',250,'MinLength',10);
figure, imshow(input_img),hold on
%��¼�ĸ���ֵ����ԭͼ�ϵ������ʾ
points = zeros(4,2);
index = 1;
for k = 1:length(lines)
    %ÿ��ֱ�ߵ���ʼ�����ֹ��
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
%����ԭͼ���ĸ������ҳ�͸�ӱ任�ľ���
TForm = cp2tform(points,target,'projective');
%����ת������ת��ԭͼ
[bw_trans, xdata, ydata] = imtransform(bw, TForm,'FillValues', 0);
[cb_trans, xdata1, ydata1] = imtransform(input_img, TForm,'FillValues', 0);
%����任
[H,Theta,Rho] = hough(bw_trans);
P  = houghpeaks(H,4,'threshold',ceil(0.2*max(H(:))));
%���ݻ���任�ͷ�ֵ��Ѱ�Һ������߶�
lines = houghlines(bw_trans,Theta,Rho,P,'FillGap',80,'MinLength',1);

%��¼�ĸ���ֵ����ԭͼ�ϵ������ʾ
points1 = zeros(4,2);
index = 1;
%��ԭͼ�ϱ������ֱ��
for k = 1:length(lines)
    %ÿ��ֱ�ߵ���ʼ�����ֹ��
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

%�����ĸ��������ͼ��
g = imcrop(cb_trans, [points1(1,1)+2,points1(1,2)+3,points1(3,1)-points1(1,1)-3,points1(2,2)-points1(1,2)-30]);
figure;imshow(g);
imwrite(g,'out.png');

end

