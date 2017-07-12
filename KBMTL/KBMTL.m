function KBMTL(X_train, Y_train, X_test, Y_test, out_file, alpha, beta, gamma, seed)

maxNumCompThreads(1);

%initalize the parameters of the algorithm
parameters = struct();

%set the hyperparameters of gamma prior used for intermediate noise
parameters.alpha_upsilon = 1;
parameters.beta_upsilon = 1;

%set the hyperparameters of gamma prior used for bias
parameters.alpha_gamma = 1;
parameters.beta_gamma = 1;

%set the hyperparameters of gamma prior used for kernel weights
parameters.alpha_omega = 1;
parameters.beta_omega = 1;

%set the hyperparameters of gamma prior used for output noise
parameters.alpha_epsilon = 1;
parameters.beta_epsilon = 1;

%set the number of iterations
parameters.iteration = 200;

%determine whether you want to calculate and store the lower bound values
parameters.progress = 0;

%set the seed for random number generator used to initalize random variables
parameters.seed = seed;

K_train = rbf_dot(X_train, X_train, gamma);
K_test  = rbf_dot(X_test, X_train, gamma);
parameters.alpha_lambda = alpha;
parameters.beta_lambda = beta;

%set the number of tasks (e.g., the number of compounds in Nature Biotechnology paper)
T = size(Y_train, 2);
%set the number of kernels (e.g., the number of views in Nature Biotechnology paper)
P = 1;

%initialize the kernels and outputs of each task for training
Ktrain = cell(1, T);
ytrain = cell(1, T);
for t = 1:T
    y = Y_train(:, t);
    K = K_train(~isnan(y), ~isnan(y));
    y = y(~isnan(y));
    Ktrain{t} = K; %should be an Ntra x Ntra x P matrix containing similarity values between training samples of task t
    ytrain{t} = y; %should be an Ntra x 1 matrix containing target outputs of task t
end

%perform training
state = bayesian_multitask_multiple_kernel_learning_train(Ktrain, ytrain, parameters);

%initialize the kernels of each task for testing
Ktest = cell(1, T);
for t = 1:T
    y = Y_train(:, t);    
    K = K_test(:, ~isnan(y));    
    Ktest{t} = K'; %should be an Ntra x Ntest x P matrix containing similarity values between training and test samples of task t
end

%perform prediction
prediction = bayesian_multitask_multiple_kernel_learning_test(Ktest, state);

Y_true = Y_test;
Y_pred = zeros(size(Y_true));
for t = 1:T
   Y_pred(:, t) = prediction.y{t}.mu;
end
save(out_file, 'Y_true', 'Y_pred', '-v6');

