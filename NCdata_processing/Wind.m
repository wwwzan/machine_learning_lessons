clc;
clear all;
%% ��ȡ����
path1='wind_hour_2020082312.nc'; 
v10=double(ncread(path1,'V10'));  %��ı������
u10=double(ncread(path1,'U10'));  %��Ķ������
nlon=double(ncread(path1,'LONGITUDE1_151')); %��ȡ���ȱ���
nlat=double(ncread(path1,'LATITUDE1_151')); %��ȡγ�ȱ���

%% ���ӻ�
[mlat,mlon]=meshgrid(nlat,nlon);
time=169; %չʾ֮��169��Сʱ��Ԥ������
v_time = v10(:,:,time);
u_time = u10(:,:,time);
P = sqrt(v_time.^2 + u_time.^2);
%ͶӰ���� 'Miller'��'hammer-aitoff','Equidistant Cylindrical'
m_proj('Equidistant Cylindrical','lon',[105 135],'lat',[10 40]);

%��ͼ���ֱ��ʽϵͣ�
m_pcolor(mlon,mlat,P);      
%m_contourf(mlon,mlat,P,'linestyle','none'); %��ֵ�� ---�����ڲ�ֵ���Ч��

colormap(flipud(m_colmap('jet')));%������ɫlegend
hold on;
shading flat;
m_coast('line','Color', [.0 .0 .0]);% ֻ������½������line�������ߵ���ɫ
%m_coast('patch',[.6 1 .6]);%m_coast('color','k');
m_quiver(mlon,mlat,u_time,v_time,'color','k');   % ��ӷ糡ʸ���ķ����ͷ
% ������ʽ����
m_grid('linestyle','none','box','fancy','tickdir','out','LineWidth',0.5);
title({'ʵ������糡ʾ��ͼ',''},'fontsize',14,'fontweight','bold');
hh=colorbar('h');
set(hh,'tickdir','out');
xlabel(hh,'Wind');
hold off;

%ncdisp(path1);%��ʾnc�ļ������нṹ���Ա����˽����������

%ncdisp(path1,'V10');%��ʾָ�����������ݣ�ע��һ��Ҫ�Ǳ���variables�ſ���
%ncdisp(path1,'/','min');%����ʾ�ṹ�Լ�����
%ncdisp(path1,'/','full');%ȫ����ʾ���нṹ�Ͷ�����Ϣ

