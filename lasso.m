%% <<-- CODE ARCHIVE -->>
%% Project name: Inexact Preconditioning on Prox-SVRG and Katyusha 
%% Coded by:     Fei Feng 
%% Last update:  12/17/2018
%% Content:      test example -- Lasso
%% Details:      min_x (1/2n)*\|Ax-b\|^2+\lambda \|x\|_1

classdef lasso
   properties
      A         % an n-by-p feature matrix
      b         % label vector
      params    % hyperparameters
      n         % number of data
      p         % dimension of x
      min_value % minimum value
   end
   
   methods
      function prob = Lasso(A_, b_, params_)
          prob.A = A_;
          prob.b = b_;
          prob.params = params_;
          [prob.n, prob.p] = size(A_);
          prob.min_value = 0;
      end
      
      % compute gradient
      function g = grad(prob, x, batch_size)
          % full gradient
          if(batch_size == prob.n)
              g = prob.A'*(prob.A*x-prob.b)/prob.n;
          % mini-batched gradient
          else
              y = randsample(prob.n, batch_size);
              select_A = zeros(batch_size, prob.p);
              select_b = zeros(batch_size, 1);
              for i=1:batch_size
                  select_A(i,:) = prob.A(y(i),:);
                  select_b(i) = prob.b(y(i));
              end
              g = select_A'*(select_A*x-select_b)/batch_size;
          end
      end
      
      % compute gradient difference 
      function g = scGradDiff(prob, w, x, batch_size)
          % randomly select #batch_size numbers from [n]
          y = randsample(prob.n,batch_size);
          select_A = zeros(batch_size, prob.p);
          select_b = zeros(batch_size, 1);
          for i=1:batch_size
              select_A(i,:) = prob.A(y(i),:);
              select_b(i) = prob.b(y(i));
          end
          g = select_A'*(select_A*(w-x)-select_b)/batch_size;
      end
      
      % L1 proximal operator
      function y = proximalL1(x, gamma)
          y = zeros(length(x));
          for i=1:length(x)
              y(i) = sign(x(i))*max(abs(x(i))-gamma, 0);
          end
      end
      
      % proximal with block diagonal preconditioner
      function y = blockDiagonalProx(prob, w, tilde_g)
          % set initial point as w
          y = w;       
          gamma = prob.params.GAMMA;
          % trivial preconditioner
          if(prob.params.PRECDN == 0) 
              y = proximalL1(w-gamma*tilde_g, gamma);
          % non-trivial preconditioner
          else         
              block_size =  prob.params.BLOCK_SIZE;
              block_num = ceiling(prob.p / block_size);
              step_size = 2/prob.params.L;
              % solve with random proximal BCD 
              for i = 1:prob.params.MAX_SUB_ITER
                  % select a block
                  block_id = randi(block_num,1);
                  block_start = 1+(block_id-1)*block_size;
                  block_end = min(prob.p, block_id*block_size);
                  % label corresponding block matrix
                  sub_A = prob.A(:, block_start:block_end);
                  y_block = y(block_start:block_end);
                  w_block = w(block_start:block_end);
                  tilde_g_block = tilde_g(block_start:block_end);
                  % forward operator
                  temp = y - step_size * (1/gamma * sub_A' * (sub_A * (y_block-w_block))+tilde_g_block);
                  % backward operator
                  for j = block_start:block_end
                      y(j) = sign(temp(j-block_start+1))*max(abs(temp(j-block_start+1))-gamma, 0);
                  end
              end
          end
      end
      
      % sub-optimality
      function error = checkError(prob, x)
          error = .5/prob.n * norm(prob.A*x-prob.b)^2 + prob.lambda * norm(x,1)-prob.min_value;
      end
   end
end