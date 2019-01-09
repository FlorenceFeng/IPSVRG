%% <<-- CODE ARCHIVE -->>
%% Project name: Inexact Preconditioning on Prox-SVRG and Katyusha 
%% Coded by:     Fei Feng 
%% Last update:  01/07/2019
%% Content:      build data for PCA
%% Details:      min_x (1/2n) (x'Ax + b'x) +\lambda \|x\|_1

function data = buildPCA
n = 100;
p = 10;
DSTYLE = 1;
DELTA = 0.5;
K = 10;
if(p/2 ~= 0)
    fprintf('the dimension of PCA should be an even number.')
    return;
end
data.A_group = normc(rand(p, n));
data.A = zeros(p,p);
for i = 1:n
    data.A = data.A + data.A_group(:,i)*data.A_group(:,i)'/(norm(data.A_group(:,i))^2);
end
data.A = data.A/n;
data.b = rand(p, 1);
if(DSTYLE == 1)
    data.D_group = DELTA * ones(p, n);
    for i = 1:n
        y = randsample(p, p/2);
        data.D_group(y,i) = -DELTA;
    end
else
    data.D_group = K/(n-1) * ones(p,n);
    y = randi(n,p,1);
    for i = 1:p
        data.D_group(i,y(i))= -K;
    end
end
end