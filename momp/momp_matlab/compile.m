clc;

% Data loading   
rng(10, 'twister'); % set random seed
T = cumsum(smooth(randn(1,2^16)));  % make data 


m = 512; %setting the motif length
[momp_out, momp_loc] = momp_v9(T, m, 1, 1, 0);
