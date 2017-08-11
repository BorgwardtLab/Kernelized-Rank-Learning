addpath(genpath([pwd '/yamlmatlab/']));
addpath(genpath([pwd '/KBMTL']));

config = yaml.ReadYaml('config.yaml');
cv = config.cv;
keepk = config.keepk;
seeds = cell2mat(config.seeds);
data_name = config.data;
analysis = config.analysis;
alphas = cell2mat(config.kbmtl_alphas);
betas = cell2mat(config.kbmtl_betas);
gammas = cell2mat(config.kbmtl_gammas);
keepk_ratios = cell2mat(config.keepk_ratios);
sample_ratios = cell2mat(config.sample_ratios);

directory = 'result';
mkdir(directory);
directory = strcat(directory, '/', data_name);
mkdir(directory);
directory = strcat(directory, '/', analysis);
mkdir(directory);
directory = strcat(directory, '/', 'KBMTL');
mkdir(directory);

if strcmp(analysis, 'FULL')
    for s=1:length(seeds)
        seed = seeds(s);
        for i=1:cv
            [X_train, X_test, Y_train, Y_test] = read_FULL(data_name, seed, i-1, -1);
            for a=1:length(alphas)
                alpha = alphas(a);
                for b=1:length(betas)
                    beta = betas(b);
                    for g=1:length(gammas)
                        gamma = gammas(g);
                        out_file = sprintf('result/%s/FULL/KBMTL/KBMTL_FULL_seed%d_cv%d_Alpha%s_Beta%s_Gamma%s.mat', data_name, seed, i-1, num2str(alpha), num2str(beta), num2str(gamma));
                        KBMTL(X_train, Y_train, X_test, Y_test, out_file, alpha, beta, gamma, seed)
                    end
                end
            end
            for j=1:cv
                [X_train, X_test, Y_train, Y_test] = read_FULL(data_name, seed, i-1, j-1);
                for a=1:length(alphas)
                    alpha = alphas(a);
                    for b=1:length(betas)
                        beta = betas(b);
                        for g=1:length(gammas)
                            gamma = gammas(g);
                            out_file = sprintf('result/%s/FULL/KBMTL/KBMTL_FULL_seed%d_cv%d.%d_Alpha%s_Beta%s_Gamma%s.mat', data_name, seed, i-1, j-1, num2str(alpha), num2str(beta), num2str(gamma));
                            KBMTL(X_train, Y_train, X_test, Y_test, out_file, alpha, beta, gamma, seed)
                        end
                    end
                end                    
            end
        end
    end    
end

if strcmp(analysis, 'SAMPLE')
    for s=1:length(seeds)
        seed = seeds(s);
        for sr=1:length(sample_ratios)
            sample_ratio = sample_ratios(sr);
            for i=1:cv
                [X_train, X_test, Y_train, Y_test] = read_SAMPLE(data_name, seed, sample_ratio, i-1, -1);
                for a=1:length(alphas)
                    alpha = alphas(a);
                    for b=1:length(betas)
                        beta = betas(b);
                        for g=1:length(gammas)
                            gamma = gammas(g);
                            out_file = sprintf('result/%s/SAMPLE/KBMTL/KBMTL_SAMPLE_seed%d_cv%d_ratio%s_Alpha%s_Beta%s_Gamma%s.mat', data_name, seed, i-1, num2str(sample_ratio), num2str(alpha), num2str(beta), num2str(gamma));
                            KBMTL(X_train, Y_train, X_test, Y_test, out_file, alpha, beta, gamma, seed)
                        end
                    end
                end
                for j=1:cv
                    [X_train, X_test, Y_train, Y_test] = read_SAMPLE(data_name, seed, sample_ratio, i-1, j-1);
                    for a=1:length(alphas)
                        alpha = alphas(a);
                        for b=1:length(betas)
                            beta = betas(b);
                            for g=1:length(gammas)
                                gamma = gammas(g);
                                out_file = sprintf('result/%s/SAMPLE/KBMTL/KBMTL_SAMPLE_seed%d_cv%d.%d_ratio%s_Alpha%s_Beta%s_Gamma%s.mat', data_name, seed, i-1, j-1, num2str(sample_ratio), num2str(alpha), num2str(beta), num2str(gamma));
                                KBMTL(X_train, Y_train, X_test, Y_test, out_file, alpha, beta, gamma, seed)
                            end
                        end
                    end                    
                end
            end
        end
    end    
end

if strcmp(analysis, 'KEEPK')
    for s=1:length(seeds)
        seed = seeds(s);
        for kr=1:length(keepk_ratios)
            keepk_ratio = keepk_ratios(kr);
            for i=1:cv
                [X_train, X_test, Y_train, Y_test] = read_KEEPK(data_name, seed, keepk_ratio, keepk, i-1, -1);
                for a=1:length(alphas)
                    alpha = alphas(a);
                    for b=1:length(betas)
                        beta = betas(b);
                        for g=1:length(gammas)
                            gamma = gammas(g);
                            out_file = sprintf('result/%s/KEEPK/KBMTL/KBMTL_KEEPK_seed%d_cv%d_ratio%s_keep%s_Alpha%s_Beta%s_Gamma%s.mat', data_name, seed, i-1, num2str(keepk_ratio), num2str(keepk), num2str(alpha), num2str(beta), num2str(gamma));
                            KBMTL(X_train, Y_train, X_test, Y_test, out_file, alpha, beta, gamma, seed)
                        end
                    end
                end
                for j=1:cv
                    [X_train, X_test, Y_train, Y_test] = read_KEEPK(data_name, seed, keepk_ratio, keepk, i-1, j-1);
                    for a=1:length(alphas)
                        alpha = alphas(a);
                        for b=1:length(betas)
                            beta = betas(b);
                            for g=1:length(gammas)
                                gamma = gammas(g);
                                out_file = sprintf('result/%s/KEEPK/KBMTL/KBMTL_KEEPK_seed%d_cv%d.%d_ratio%s_keep%s_Alpha%s_Beta%s_Gamma%s.mat', data_name, seed, i-1, j-1, num2str(keepk_ratio), num2str(keepk), num2str(alpha), num2str(beta), num2str(gamma));
                                KBMTL(X_train, Y_train, X_test, Y_test, out_file, alpha, beta, gamma, seed)
                            end
                        end
                    end                    
                end
            end
        end
    end    
end

exit()

function [X_train, X_test, Y_train, Y_test] = read_FULL(data_name, seed, i, j)
    if j == -1
        X_file = sprintf('data/%s/X/X_seed%d_cv%d.mat', data_name, seed, i);
        Y_file = sprintf('data/%s/FULL_Y/FULL_Y_seed%d_cv%d.mat', data_name, seed, i);
    else
        X_file = sprintf('data/%s/X/X_seed%d_cv%d.%d.mat', data_name, seed, i, j);
        Y_file = sprintf('data/%s/FULL_Y/FULL_Y_seed%d_cv%d.%d.mat', data_name, seed, i, j);
    end
    X = load(X_file);
    X_train = X.X_train;
    X_test = X.X_test;
    X_train = double(X_train);
    X_test = double(X_test);    
    Y = load(Y_file);
    Y_train = Y.Y_train;
    Y_test = Y.Y_test;
end

function [X_train, X_test, Y_train, Y_test] = read_SAMPLE(data_name, seed, sample_ratio, i, j)
    if j == -1
        X_file = sprintf('data/%s/X/X_seed%d_cv%d.mat', data_name, seed, i);
        Y_file = sprintf('data/%s/SAMPLE_Y/SAMPLE_Y_seed%d_cv%d_sr%1.1f.mat', data_name, seed, i, sample_ratio);
    else
        X_file = sprintf('data/%s/X/X_seed%d_cv%d.%d.mat', data_name, seed, i, j);
        Y_file = sprintf('data/%s/SAMPLE_Y/SAMPLE_Y_seed%d_cv%d.%d_sr%1.1f.mat', data_name, seed, i, j, sample_ratio);
    end
    X = load(X_file);
    X_train = X.X_train;
    X_test = X.X_test;
    X_train = double(X_train);
    X_test = double(X_test);       
    Y = load(Y_file);
    Y_train = Y.Y_train;
    Y_test = Y.Y_test;
end

function [X_train, X_test, Y_train, Y_test] = read_KEEPK(data_name, seed, keepk_ratio, keepk, i, j)
    if j == -1
        X_file = sprintf('data/%s/X/X_seed%d_cv%d.mat', data_name, seed, i);
        Y_file = sprintf('data/%s/KEEPK_Y/KEEPK_Y_seed%d_cv%d_kr%1.1f_keep%s.mat', data_name, seed, i, keepk_ratio, num2str(keepk));
    else
        X_file = sprintf('data/%s/X/X_seed%d_cv%d.%d.mat', data_name, seed, i, j);
        Y_file = sprintf('data/%s/KEEPK_Y/KEEPK_Y_seed%d_cv%d.%d_kr%1.1f_keep%s.mat', data_name, seed, i, j, keepk_ratio, num2str(keepk));
    end
    X = load(X_file);
    X_train = X.X_train;
    X_test = X.X_test;
    X_train = double(X_train);
    X_test = double(X_test);       
    Y = load(Y_file);
    Y_train = Y.Y_train;
    Y_test = Y.Y_test;
    
end


