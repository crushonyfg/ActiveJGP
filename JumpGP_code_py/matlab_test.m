// test covNoise
loghyper = [1.07160318, 0.47091217, -1.3225513, -1.12983861];

x = [1.20663103, 2.06896326, 0.81121173;
    -0.91507893, 1.82804932, -0.57245283;
    -0.15962585, -0.71187307, -0.21697617;
    0.24503187, 0.26066701, -1.0035294;
    -1.48781972, -1.05787692, -0.17660721];

z = [0.10125506, 0.48125236, 1.08439458;
    2.24963887, -1.26946239, -1.38533352;
    0.85889631, -1.45310838, 0.25771964;
    -1.7467173, 0.30137992, 0.87534816];

xt = x0;
addpath('./cov');
addpath('./lik');

cv  = {'covSum', {'covSEard', 'covNoise'}};

[mu_t, sig2_t, model, h] = JumpGP_LD(x, y, xt, 'CEM', 1);

logtheta = minimize(logtheta0, 'loglikelihood', -100, cv, x, y);
logtheta
loglike = loglikelihood(logtheta, cv, x, y)

d = 5;N = 100;Nt=5;Nc=20;S=5;outputfile='./res';
d = 2;
N = d * 20;
S = 5;
Nc = d * 100;
Nt = d * 40;

xt = x0;
[mu_t, sig2_t, model, h] = JumpGP_LD(x, y, xt, 'CEM', 0);
[bias2, var, bias, parts] = calculate_bias_and_variance(model, xt)