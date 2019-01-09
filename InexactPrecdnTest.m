%% <<-- ARCHIVE -->>
%% Project name: Inexact Preconditioning on Prox-SVRG and Katyusha 
%% Coded by:     Fei Feng 
%% Last update:  01/07/2019
%% Content:      main algorithms
%% Details:      Prox-SVRG/Katyusha

function InexactPrecdnTest(data)
%% Regulator Parameters
params.LAMBDA = 1e-4;           % penalty parameter
params.RIDGE = 1e-6;            % ridge regression parameter
%% Algorithm Parameters
params.VERBOSE = 1;             % choose 1 to print details
params.MAX_EPOCH = 100000;      % max number of outer loop in SVRG
params.MAX_ITER = 100;          % max number of inner loop in SVRG
params.CHECK_STEP = 100;        % number of iterations to check
params.TOL = 1e-10;             % accuracy tolerance
params.BATCH_SIZE = 1;          % batch-size for stochastic gradient
params.KATYUSHA = 0;            % choose 1 to run Katyusha, else Prox-SVRG
params.TAU = 0.26;               % Katyusha-X: momentum weight, tau=0.5 is PSVRG
%% Stepsize Parameters
params.L = 9;                  % gradient's Lipshitz for subproblem
params.GAMMA = 2/params.L;      % proximal parameter
%% Preconditioner Parameters
params.PRECDN = 0;              % choose 1 to use preconditioner
params.FISTA = 1;
params.MAX_SUB_ITER =20;         % max number of subproblem
params.BUILD = 1;               % choose 1 to formulate preconditioner
params.BCD_SIZE = 1;          % block-size for BCD
params.M_BLOCK_SIZE = 1;        % block-size for preconditioner
params.ETA = 1E-1;              % stepsize for subproblem
params.EPS = 0.3;           % preconditioner = \epsilon*L + 1/n*\approx A^TA, 0.5 is equivalent to no preconditioner.
params.SCALE = 50;
%% Problem settings
prob = lasso(data, params);
%prob = logistic(data, params);
%prob = pca(data,params);    
prob.min_value = 0.2549690153;
x = zeros(prob.p,1);  
w = zeros(prob.p,1);
if(params.KATYUSHA)
    y_old = zeros(prob.p,1);
    y_new = zeros(prob.p,1);
end

%% Inexact Preconditioner Solver
fprintf('\nCalling Inexact Preconditioner Solver_MATLAB 12/19/2018\n');
fprintf('-----------------------------------------------\n');
if(params.VERBOSE)
    fprintf('PRECDN = %d\n', params.PRECDN);
    fprintf('KATYUSHA = %d\n', params.KATYUSHA);
    fprintf('L = %d\n', params.L);
    fprintf('EPS = %d\n',params.EPS);
    fprintf('ETA = %d\n', params.ETA);
    fprintf('SCALE = %d\n', params.SCALE);
end
fprintf('Time\t,Epoch\t,Error\n');
tic
for i = 0:params.MAX_EPOCH
    % full gradient at x
    g = grad(prob, x, prob.n);
    % inner loop
    for j = 1:params.MAX_ITER
        % a variance-reduced stochastic gradient
        tilde_g = g + scGradDiff(prob, w, x, params.BATCH_SIZE);
        % solve non-smooth part
        w = blockDiagonalProx(prob, w, tilde_g); 
    end
   
    if(params.KATYUSHA)
        y_old = y_new;
        y_new = w;
        x = (1.5*y_new + 0.5*x - (1-params.TAU)*y_old)/(1+params.TAU);
    else
        x = w;
    end
    
    if(mod(i,params.CHECK_STEP)==0)
        time = toc;
        error = checkError(prob, x);
        if(params.VERBOSE)
            fprintf('%.3f,%d,%.10f\n', time, i, error);
        end
        if(abs(error) < params.TOL)
            break;
        end
    end
end





