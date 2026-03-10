clear
% fs=2^6;    %采样频率
% dt=1/fs;    %时间精度
% timestart=-8;
% timeend=8;
% t=(0:(timeend-timestart)/dt-1)*dt+timestart;
% L=length(t);
% z=4*sin(2*pi*linspace(6,12,L).*t);
x = -10:0.01:10;
y = normpdf(x,0,1);
y1 = normpdf(x,0.1,1);
y3 = normpdf(x,1,1);
figure(1)
plot(x,y)
figure(2)
c = cwt(y);
cwt(y)
figure(3)
plot(x,y1)
figure(4)
c1 = cwt(y1);
cangle = angle(c);
c1angle = angle(c1);
delta = (cangle - c1angle)*10;
delta = Unwrap_TIE_DCT_Iter(delta);
mesh(delta)
c2 = c1 .* exp(delta*1i);
figure(5)
y2 = icwt(c2);
plot(x,y2)
figure(6)
plot(x,y3)
figure(7)
c3 = cwt(y3);
c3angle = angle(c3);
c1angle = Unwrap_TIE_DCT_Iter(c1angle);
c3angle = Unwrap_TIE_DCT_Iter(c3angle);
delta13 = (c1angle - c3angle);
figure(8)
dwt(y,db4)
d = dwt(y,db4);
d1 = dwt(y1,db4);
d2 = dwt(y2,db4);
dangle = angle(d);
d1angle = angle(d1);
d2angle = angle(d2);
deltad1 = Unwrap_TIE_DCT_Iter(dangle - d1angle);
deltad2 = Unwrap_TIE_DCT_Iter(dangle - d2angle);
figure(9)
mesh(deltad2)
figure(10)
mesh(deltad1)