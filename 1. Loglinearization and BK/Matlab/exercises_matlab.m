%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Macroeconomics II - Practical Session 1
% Author: Matheus Franciscão
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


beta = 0.99;
sigma = 1;
kappa = 1.2;
phi = 1.5;
rho = 0.05;
rho_m = 0.9;

g0 = [1 1/sigma 0 0; 0 beta 0 0; 0 -phi 1 -1; 0 0 0 1];

g1 = [1 0 1/sigma 0; -kappa 1 0 0; 0 0 0 0; 0 0 0 rho_m];

Psi = [0;0;0;1];

Pi = [-1 -1/sigma; -beta 0; 0 0; 0 0];

Const = [-rho/sigma; 0; rho; 0];

[G1,C,impact,fmat,fwt,ywt,gev,eu,loose] = gensys(g0,g1,Const,Psi,Pi,1);

disp("eu");
disp(eu);

if eu==[1 1]
    T=30;
    
    series = zeros(T+1,4);
    series(1, :) = impact;
    for t=1:T
        series(t+1, :) = G1*series(t,:).';
    end
    
    subplot(321)
    plot(0:T, series(:,1),'LineWidth',2)
    title('y_t')
    
    subplot(322)
    plot(0:T, series(:,2),'LineWidth',2)
    title('pi_t')
    
    subplot(323)
    plot(0:T, series(:,3),'LineWidth',2)
    title('r_t')
    
    subplot(324)
    plot(0:T, series(:,4),'LineWidth',2)
    title('u_t')
end
