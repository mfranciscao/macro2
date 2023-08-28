/****************************************************************************
            Macroeconomics II - Practical Session 1
                 Author: Matheus Franciscão
*****************************************************************************/

/*****************************
Variables
******************************/

var y_t pi_t r_t u_t;

varexo epsilon_m;

parameters beta sigma kappa phi rho rho_m sigma_m;


/*****************************
Parameters
******************************/

beta = 0.99;
sigma = 5;
kappa = 1.2;
phi = 1.5;
rho = 0.05;
rho_m = 0.5;
sigma_m = 0.025;


/************************************************
Loglinear Equilibrium
*************************************************/

model (linear);

y_t = EXPECTATION(0)(y_t(+1)) - (1/sigma)*(r_t - EXPECTATION(0)(pi_t(+1)) - rho);
pi_t = beta*EXPECTATION(0)(pi_t(+1)) + kappa*y_t;
r_t = rho + phi*pi_t + u_t;
u_t = rho_m*u_t(-1) + sigma_m*epsilon_m;

end;





/*****************************
Shocks
******************************/

shocks;    
    var epsilon_m;
    stderr 1;
end;


/*****************************
IRFS
******************************/
steady;
stoch_simul(irf=30);
