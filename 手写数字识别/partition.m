function [] = partition()
input = imread('out.png');
%figure;imshow(input);
bw = rgb2gray(input);
%��Ե/�������
%t = edge(bw);
t = im2bw(bw,0.68);
%figure;imshow(t);

%���ʹ���
B=strel('disk',1);
im2 = imclose(1-t,B);
%figure;imshow(im2);


%����ÿ�����������ھ�������λ��
x=zeros(1,length(B));
y=zeros(1,length(B));
width=zeros(1,length(B));
height=zeros(1,length(B));
index = 1;

%�������
B = bwboundaries(im2,'noholes');
figure;imshow(input); hold on;

%���ÿ���������ڵľ�������
for k=1:length(B)
   boundary = B{k};
   %��ʾÿ����������������Ҫ�������imshow(input)��Ϊimshow(im2)
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

%���ݾ�������λ�òü���ÿ������
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
%Ԥ����ͼ�񣬽�ÿ�����ֵ�ͼ����Ϊ28*28��Ϊ����ͼ��Ƚ����Ķ��벢����ԭ�����
%       �Ƚ�ͼ��Ϊ����������СΪ22*22��Ȼ���ͼ�����ܲ�0��28*28
%       ����ͼ���ϸ���ȶ��������Ͳ��ղ�������mnist���ݼ��е����ֱȽ���
%       �Ƚ�ͼ����бһ���Ƕ�Ԥ��Ч������
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


