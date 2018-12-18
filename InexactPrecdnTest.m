%% <<-- CODE ARCHIVE -->>
%% Project name: Inexact Preconditioning on Prox-SVRG and Katyusha 
%% Coded by:     Fei Feng 
%% Last update:  12/17/2018
%% Content:      main algorithms
%% Details:      Prox-SVRG/Katyusha

function InexactPrecdnTest(A, b)
%% hyperparameters settings
params.MAX_EPOCH = 100;        % max number of outer loop in SVRG
params.MAX_ITER = 100;         % max number of inner loop in SVRG
params.MAX_SUB_ITER = 5;       % max number of subproblem
params.BATCH_SIZE = 50;        % batch-size for stochastic gradient
params.VERBOSE = 1;            % choose 1 to print details
params.PRECDN = 1;             % choose 1 to use preconditioner
params.KATYUSHA = 0;           % choose 1 to run Katyusha, else Prox-SVRG
params.TAU = 0.5;              % Katyusha-X: momentum weight
params.BLOCK_SIZE = 50;        % block-size for precdn
params.LAMBDA = 1;             % penalty parameter
params.GAMMA = 1;              % proximal parameter
params.L = 1e4;                % gradient's Lipshitz for subproblem
params.Check_step = 10;

%% variables settings
prob = lasso(A, b, params);
DIM = prob.p;
x = zeros(DIM);  
if(params.KATYUSHA)
    y_old = zeros(DIM);
    y_new = zeros(DIM);
end

%% Inexact Preconditioner Solver
fprintf('\nCalling Inexact Preconditioner Solver_MATLAB 12/13/2018\n');
fprintf('-----------------------------------------------\n');
if(params.VERBOSE)
    disp(struct2table(params));
end

tic
for i = 1:params.MAX_EPOCH
    % full gradient at x
    g = grad(prob, x, DIM);
    % inner loop
    for j = 1:params.MAX_ITER
        % a variance-reduced stochastic gradient
        tilde_g = g + scGradDiff(prob, w, x, params.BATCH_SIZE);
        % solve non-smooth part
        w = BlockDiagonalProx(prob, w, tilde_g); 
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
        error = checkError(prob, w);
        if(params.VERBOSE)
            fprintf('Time = %f, Iter = %d, Error = %f\n', time, i, error);
        end
    end
end





