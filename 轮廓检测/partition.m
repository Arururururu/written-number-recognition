function [] = partition()
input = imread('input.png');
figure;imshow(input);
bw = rgb2gray(input);
%��Ե/�������
t = edge(bw,'sobel');
figure;imshow(t);

%�������
im2=imfill(t,'holes');
%figure;imshow(im2);

%��ʴ����
B=strel('disk',1);
im1=imerode(im2,B);
figure;imshow(im1);

%���ʹ���
B=strel('disk',6);
im1=imdilate(im1,B);
%figure;imshow(im1);

%����ÿ�����������ھ�������λ��
x=zeros(1,length(B));
y=zeros(1,length(B));
width=zeros(1,length(B));
height=zeros(1,length(B));
index = 1;

%�������
B = bwboundaries(im1);
figure;imshow(input); hold on;

%���ÿ���������ڵľ�������
for k=1:length(B)
   boundary = B{k};
   %��ʾÿ����������������Ҫ�������imshow(input)��Ϊimshow(im1)
   %plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 2);
   x(1,index)=min(boundary(:,2));
   y(1,index) = min(boundary(:,1));
   width(1,index)=max(boundary(:,2))-min(boundary(:,2));
   height(1,index)=max(boundary(:,1))-min(boundary(:,1));
   index = index+1;
   rectangle('Position',[x(1,k),y(1,k),width(1,k),height(1,k)],'edgecolor','r');
end
hold off

%���ݾ�������λ�òü���ÿ������
for k=1:length(B)
   g = imcrop(input, [x(1,k),y(1,k),width(1,k),height(1,k)]);
   figure;imshow(g);
end

end

