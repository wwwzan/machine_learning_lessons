clc;
clear all;
%% 读取数据
path1='wind_hour_2020082312.nc'; 
v10=double(ncread(path1,'V10'));  %风的北向分量
u10=double(ncread(path1,'U10'));  %风的东向分量
nlon=double(ncread(path1,'LONGITUDE1_151')); %读取经度变量
nlat=double(ncread(path1,'LATITUDE1_151')); %读取纬度变量

%% 可视化
[mlat,mlon]=meshgrid(nlat,nlon);
time=169; %展示之后169个小时的预报数据
v_time = v10(:,:,time);
u_time = u10(:,:,time);
P = sqrt(v_time.^2 + u_time.^2);
%投影类型 'Miller'，'hammer-aitoff','Equidistant Cylindrical'
m_proj('Equidistant Cylindrical','lon',[105 135],'lat',[10 40]);

%画图（分辨率较低）
m_pcolor(mlon,mlat,P);      
%m_contourf(mlon,mlat,P,'linestyle','none'); %等值线 ---类似于插值后的效果

colormap(flipud(m_colmap('jet')));%设置颜色legend
hold on;
shading flat;
m_coast('line','Color', [.0 .0 .0]);% 只保留大陆轮廓线line；轮廓线的颜色
%m_coast('patch',[.6 1 .6]);%m_coast('color','k');
m_quiver(mlon,mlat,u_time,v_time,'color','k');   % 添加风场矢量的方向箭头
% 格网格式设置
m_grid('linestyle','none','box','fancy','tickdir','out','LineWidth',0.5);
title({'实验区域风场示意图',''},'fontsize',14,'fontweight','bold');
hh=colorbar('h');
set(hh,'tickdir','out');
xlabel(hh,'Wind');
hold off;

%ncdisp(path1);%显示nc文件的所有结构，以便大概了解里面的内容

%ncdisp(path1,'V10');%显示指定变量的内容，注意一定要是变量variables才可以
%ncdisp(path1,'/','min');%简单显示结构以及定义
%ncdisp(path1,'/','full');%全部显示所有结构和定义信息

