function [x2 y2] = corVal(x1, y1, ds, theta_radian)

d2x = ds*cos(theta_radian);
x2 = x1 + d2x;
d2y = ds*sin(theta_radian);
y2 = y1 + d2y;