function out=rmean(y,horz)

NN=horz;
m1=[];
N=size(y,1);
for i=1:NN:N-NN
temp=y(i:i+NN-1,:);
    m1=[m1;mean(temp)];
    
end
out=m1