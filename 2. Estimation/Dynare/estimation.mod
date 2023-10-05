/**************************************************************************************************
                                        Macroeconomics II
                                            Monitoria 2
                                     Bayesian DSGE Estimation
**************************************************************************************************/

/**************************************************************************************************
                                   Definitions and Calibration
**************************************************************************************************/

// Endogenous Variables:
var x pi R g u v;

// Exogenous Variables:
varexo e_g e_u e_v;

// Parameters:
parameters sigma delta alpha omega rho1 rho2 rho3 sigma1 sigma2 sigma3 beta;

// Initial Guess:

beta = 0.99;
sigma = 0.5;
delta = 1.1;
alpha = 0.5;
omega = 1.1;
rho1 = 0.5;
sigma1 = 0.3;
rho2 = 0.5;
sigma2 = 0.3;
rho3 = 0.5;
sigma3 = 0.3;

model (linear);

x = EXPECTATION(0)(x(+1)) - 1/sigma*(R - EXPECTATION(0)(pi(+1))) + g;

pi = beta*EXPECTATION(0)(pi(+1)) + ((1-omega)*(1-beta*omega)/(alpha*omega))*x + u;

R = delta*pi + v;

g = rho1*g(-1) + sigma1*e_g;

u = rho2*u(-1) + sigma2*e_u;

v = rho3*v(-1) + sigma3*e_v;

end;


/**************************************************************************************************
                                            Shocks
**************************************************************************************************/

shocks;    
    var e_g;
    stderr 1;

    var e_u;
    stderr 1;
    
    var e_v;
    stderr 1;
end;


/**************************************************************************************************
                                           Estimation
**************************************************************************************************/

estimated_params; 

sigma, GAMMA_PDF, 1, 0.8, 0.1;
delta, GAMMA_PDF, 1.5, 1;
alpha, GAMMA_PDF, 3, 1;
omega, GAMMA_PDF, 1.5, 1;
rho1, BETA_PDF, 0.5, 0.2, 0.01, 1;
rho2,BETA_PDF, 0.5, 0.2, 0.01, 1;
rho3,BETA_PDF, 0.5, 0.2, 0.01, 1;
sigma1, INV_GAMMA_PDF, 1, 0.5;
sigma2, INV_GAMMA_PDF, 1, 0.5;
sigma3, INV_GAMMA_PDF, 1, 0.5;

end;


varobs x pi R;

estimation(

	optim=('Algorithm','active-set'),
    datafile=dados,xls_sheet=Sheet1,xls_range=A1:C201,

	mode_compute = 1,

    first_obs = 1,

   	presample = 4,

	lik_init = 2, 
	prefilter = 0,

	mh_replic = 35000,

	mh_nblocks = 3,

	mh_jscale = 0.350,

	mh_drop = 0.5, 

    bayesian_irf, 

    irf = 5

); 





