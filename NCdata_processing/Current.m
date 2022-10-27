clc;
clear all;
%% ��ȡ����
path1='current_hour_20200824.nc'; 
v10=double(ncread(path1,'v'));  %�����������ٵı�������
u10=double(ncread(path1,'u'));  %�����������ٵĶ�������
mlon=double(ncread(path1,'lon')); %�������ľ���
mlat=double(ncread(path1,'lat')); %��������γ��

%% ���ӻ�
time=24; %չʾ֮��24��Сʱ��Ԥ������
v_time = v10(:,:,time);
u_time = u10(:,:,time);
P = sqrt(v_time.^2 + u_time.^2);
%ͶӰ���� 'Miller'��'hammer-aitoff','Equidistant Cylindrical'
m_proj('Equidistant Cylindrical','lon',[100 130],'lat',[15 30]);

%��ͼ���ֱ��ʽϵͣ�
m_pcolor(mlon,mlat,P);      
%m_contourf(mlon,mlat,P,'linestyle','none'); %��ֵ�� ---�����ڲ�ֵ���Ч��


colormap(flipud(m_colmap('water')));%������ɫlegend
hold on;
shading flat;
m_coast('line','Color', [.0 .0 .0]);% ֻ������½������line�������ߵ���ɫ
%m_coast('patch',[.6 1 .6]);%m_coast('color','k');
m_quiver(mlon,mlat,u_time,v_time,'color','k');   % ��ӷ糡ʸ���ķ����ͷ
% ������ʽ����
m_grid('linestyle','none','box','fancy','tickdir','out','LineWidth',0.5);
title({'ʵ����������ʾ��ͼ',''},'fontsize',14,'fontweight','bold');
hh=colorbar('h');
set(hh,'tickdir','out');
xlabel(hh,'Current');
hold off;

%ncdisp(path1);%��ʾnc�ļ������нṹ���Ա����˽����������

%ncdisp(path1,'V10');%��ʾָ�����������ݣ�ע��һ��Ҫ�Ǳ���variables�ſ���
%ncdisp(path1,'/','min');%����ʾ�ṹ�Լ�����
%ncdisp(path1,'/','full');%ȫ����ʾ���нṹ�Ͷ�����Ϣ

