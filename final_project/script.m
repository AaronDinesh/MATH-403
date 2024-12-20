clear
close all

function isToolboxInstalled(toolboxName)
    % Get a list of installed toolboxes
    v = ver;
    isInstalled = any(strcmp(toolboxName, {v.Name}));
    err_str = sprintf("The %s Toolbox is not installed", toolboxName);
    assert(isInstalled, err_str)
end

isToolboxInstalled('Statistics and Machine Learning Toolbox')

function A = genGaussianKernelrand(sigma, n, mu)
    % X = rand(n, n)*2 -1;
    % Y = rand(n,n)*2 -1;
    % A = exp(-((X.^2 + Y.^2) / (2 * sigma^2)));

    x = rand(n, 1);
    A = exp(-((x - x').^2) / (2 * sigma^2));

    A = A + mu*eye(n);
end

function [S, A_hat, U, L] = rpCholeskey(A, k)
    N = size(A, 1);
    F = zeros(N, k);
    d = diag(A);
    S = zeros(1, k); % Initialize pivot set
    for i=1:k
        probs = d/sum(d);
        pivot = randsample(1:N, 1, true, probs);
        S(i) = pivot;
        g = A(:, pivot);
        g = g - F(:, 1:i-1)*F(pivot, 1:i-1)';
        F(:, i) = g / sqrt(g(pivot));
        d = d - (abs(F(:, i)).^2);
        d = max(d, 0);
    end
    A_hat = F * F';
    [U, L, ~] = svd(F, 0);
    L = L.^2;
end


function [A_hat, U, L] = nystromApprox(A, r)
    N = size(A, 1);
    omega = randn(N, r);
    [omega, ~] = qr(omega, 0);
    Y = A*omega;
    nu = eps(norm(Y, 'fro'));
    Y_nu = Y + nu*omega;
    C = chol(omega'*Y_nu);
    B = Y_nu/C;
    [U, Sigma, ~] = svd(B, 0);
    L = max(0, Sigma.^2 - nu*eye(r));
    A_hat = U * L * U';

end

function P = genPreconditioner(A_hat, mu)
    N = size(A_hat, 1);
    [U, D] = eig(A_hat);
    lastLambda = D(end);
    P = (1/(lastLambda + mu)) * U*(D+mu*eye(N))*U' + (eye(N) - U*U'); 
end

function P = genPreconditionerUL(U, L, mu)
    N = size(U, 1);
    dL = diag(L);
    lastLambda = min(dL(dL>0));
    P = (1/(lastLambda + mu)) * U*(L + mu*eye(size(L, 1)))*U' + (eye(N) - U*U'); 
end


function E_spec = rand_power_method(A, U, L, q)
    N = size(A, 1);
    g = randn(N, 1);
    v0 = g/norm(g);
    for i=1:1:q
        v = A * v0 - U*(L*(U'*v0));
        E_spec = v0'*v;
        v = v / norm(v, 2);
        v0 = v;
    end

end

function [A_hat, U, L, l] = adaptive_rand_nystrom_approx(A, l0, lm, q, tol)
    Y = [];
    Omega = [];
    E = inf;
    m = l0;
    N = size(A, 1);
    while (E > tol)
        omega0 = randn(N, m);
        [omega0, ~] = qr(omega0, 0);
        Y0 = A*omega0;
        Omega = [Omega, omega0];
        Y = [Y, Y0];
        nu = sqrt(N)*eps(norm(Y, 2));
        Ynu = Y +nu*Omega;
        C = chol(Omega'*Y);
        B = Ynu / C;
        [U, S, ~] = svd(B, 0);
        L = max(0, S.^2 - nu*eye(size(S, 1)));
        E = rand_power_method(A, U, L, q);
        m = l0;
        l0 = 2*l0;

        if l0 > lm 
            l0 = l0 - m;
            m = lm-l0;
            omega0 = randn(N, m);
            [omega0, ~] = qr(omega0, 0);
            Y0 = A*omega0;
            Omega = [Omega, omega0];
            Y = [Y, Y0];
            nu = sqrt(N)*eps(norm(Y, 2));
            Ynu = Y +nu*Omega;
            C = chol(Omega'*Y);
            B = Ynu / C;
            [U, S, ~] = svd(B, 0);
            L = max(0, S.^2 -nu*eye(size(S, 1)));
            break
        end
    end
    
    A_hat = U * L * U';
    l = nnz(L);
end

%% Running RPCholesky and Nystrom
clear
close all
% Experimentation parameters
num_el = 40;
sigmas = logspace(-4, 4, num_el); % Different scale parameters (logarithmic scale)
mus = logspace(-4, 4, num_el); % Different regularization parameters (logarithmic scale)
N = 1000;
max_pcg_iters = 5000;
rank = 30;
pcg_tol = 1e-12;
time_rpc = zeros(length(sigmas), length(mus));
time_nys = zeros(length(sigmas), length(mus));
time_direct = zeros(length(sigmas), length(mus));

residual_rpc = zeros(length(sigmas), length(mus));
residual_nys = zeros(length(sigmas), length(mus));
residual_direct = zeros(length(sigmas), length(mus));


iter_rpc = zeros(length(sigmas), length(mus));
iter_nys = zeros(length(sigmas), length(mus));
iter_direct = zeros(length(sigmas), length(mus));


run_avg = 2;
totalIterations = length(sigmas) * length(mus)*run_avg;
hWait = waitbar(0, 'Initializing...', 'Name', 'Processing Sigma and Mu');
startTime = tic;



for avg=1:run_avg
    for s=1:length(sigmas)
        for m=1:length(mus)
            sigma = sigmas(s);
            mu = mus(m);
    
            % Update waitbar message
            currentIteration = ((avg - 1) * length(sigmas) * length(mus)) + (s - 1) * length(mus) + m;
            elapsedTime = toc(startTime);
            estimatedTotalTime = elapsedTime / currentIteration * totalIterations;
            remainingTime = estimatedTotalTime - elapsedTime;
            waitbar(currentIteration / totalIterations, hWait, ...
            sprintf('Sigma: %.2e, Mu: %.2e\nEstimated Time Left: %.2fs   Elapsed Time: %.2fs', sigma, mu, remainingTime, elapsedTime));
    
            A = genGaussianKernelrand(s, N, m);
            b = randn(N, 1);
            % fprintf("Running RPCholesky...\n")
            tic
            [~, ~, U, L] = rpCholeskey(A, rank);
            P = genPreconditionerUL(U, L, mu);
            [x_rpc, ~, rel_res, iter] = pcg(A, b, pcg_tol, max_pcg_iters, P);
            residual_rpc(s, m) = rel_res;
            time_rpc(s,m) = time_rpc(s,m) + toc;
            iter_rpc(s,m) = iter;
            
            % fprintf("Running Nystrom...\n")
            tic
            [~, U, L] = nystromApprox(A, rank);
            P = genPreconditionerUL(U, L, mu);
            [x_nys, ~, rel_res, iter] = pcg(A, b, pcg_tol, max_pcg_iters, P);
            residual_nys(s, m) = rel_res;
            iter_nys(s,m) = iter;
            time_nys(s, m) = time_nys(s, m) + toc;
            
            % fprintf("Using Matlab Solver...\n")
            tic
            
            [x_dir, ~, rel_res, iter] = pcg(A, b, pcg_tol, max_pcg_iters);
            residual_direct(s, m) = rel_res;
            iter_direct(s,m) = iter;
            time_direct(s, m) = time_direct(s, m) + toc;
        end
    end
end
% Close waitbar
close(hWait);

time_rpc = time_rpc ./ run_avg;
time_nys = time_nys ./ run_avg;
time_direct = time_direct ./ run_avg;



cmap = spring;

% Create heatmaps for runtimes
figure;
imagesc(log10(mus), log10(sigmas), log10(time_rpc)); set(gca, 'YDir', 'normal');
colormap(cmap); 
colorbar;
clim([min(log10(time_rpc(:))), max(log10(time_rpc(:)))]);
xlabel('log_{10}(\mu)');
ylabel('log_{10}(\sigma)');
title(sprintf('Log time for RPC (n=%d, rank=%d, tol=%e)', N, rank, pcg_tol));
saveas(gcf, fullfile('plots', 'time_rpc.png'));

figure;
imagesc(log10(mus), log10(sigmas), log10(time_nys)); set(gca, 'YDir', 'normal');
colormap(cmap); 
colorbar;
clim([min(log10(time_nys(:))), max(log10(time_nys(:)))]);
xlabel('log_{10}(\mu)');
ylabel('log_{10}(\sigma)');
title(sprintf('Log time for NYS (n=%d, rank=%d, tol=%e)', N, rank, pcg_tol));
saveas(gcf, fullfile('plots', 'time_nys.png'));

figure;
imagesc(log10(mus), log10(sigmas), log10(time_direct)); set(gca, 'YDir', 'normal');
colormap(cmap); 
colorbar;
clim([min(log10(time_direct(:))), max(log10(time_direct(:)))]);
xlabel('log_{10}(\mu)');
ylabel('log_{10}(\sigma)');
title(sprintf('Log time for CGS Solve (n=%d, rank=%d, tol=%e)', N, rank, pcg_tol));
saveas(gcf, fullfile('plots', 'time_direct.png'));

figure;
imagesc(log10(mus), log10(sigmas), residual_direct); set(gca, 'YDir', 'normal');
colormap(cmap); 
colorbar;
clim([min(residual_direct(:)), max(residual_direct(:))]);
xlabel('log_{10}(\mu)');
ylabel('log_{10}(\sigma)');
title(sprintf('Residul for CGS Solve (n=%d, rank=%d, tol=%e)', N, rank, pcg_tol));
saveas(gcf, fullfile('plots', 'residual_direct.png'));

figure;
imagesc(log10(mus), log10(sigmas), residual_nys); set(gca, 'YDir', 'normal');
colormap(cmap); 
colorbar;
clim([min(residual_nys(:)), max(residual_nys(:))]);
xlabel('log_{10}(\mu)');
ylabel('log_{10}(\sigma)');
title(sprintf('Residul for Nys Solve (n=%d, rank=%d, tol=%e)', N, rank, pcg_tol));
saveas(gcf, fullfile('plots', 'residual_nyspng'));

figure;
imagesc(log10(mus), log10(sigmas), residual_rpc); set(gca, 'YDir', 'normal');
colormap(cmap);
colorbar;
clim([min(residual_rpc(:)), max(residual_rpc(:))]);
xlabel('log_{10}(\mu)');
ylabel('log_{10}(\sigma)');
title(sprintf('Residul for RPC Solve (n=%d, rank=%d, tol=%e)', N, rank, pcg_tol));
saveas(gcf, fullfile('plots', 'residual_rpc.png'));

figure;
imagesc(log10(mus), log10(sigmas), iter_direct); set(gca, 'YDir', 'normal');
colormap(cmap);
colorbar;
clim([min(iter_direct(:)), max(iter_direct(:))]);
xlabel('log_{10}(\mu)');
ylabel('log_{10}(\sigma)');
title(sprintf('Iteration for CGS (n=%d, rank=%d, tol=%e)', N, rank, pcg_tol));
saveas(gcf, fullfile('plots', 'iter_cgs.png'));

figure;
imagesc(log10(mus), log10(sigmas), iter_nys); set(gca, 'YDir', 'normal');
colormap(cmap);
colorbar;
clim([min(iter_nys(:)), max(iter_nys(:))]);
xlabel('log_{10}(\mu)');
ylabel('log_{10}(\sigma)');
title(sprintf('Iteration for NYS Solve (n=%d, rank=%d, tol=%e)', N, rank, pcg_tol));
saveas(gcf, fullfile('plots', 'iter_nys.png'));

figure;
imagesc(log10(mus), log10(sigmas), iter_rpc); set(gca, 'YDir', 'normal');
colormap(cmap); 
colorbar;
clim([min(iter_rpc(:)), max(iter_rpc(:))]);
xlabel('log_{10}(\mu)');
ylabel('log_{10}(\sigma)');
title(sprintf('Iteration for RPC Solve (n=%d, rank=%d, tol=%e)', N, rank, pcg_tol));
saveas(gcf, fullfile('plots', 'iter_rpc.png'));
%% Running Kernel Ridge Regression
%This code performs KRR on the MNIST dataset to see if it can 
%differentiate between 2 and 7. We use RPCholesky to reduce the number
%of iterations to solve the system
clear
close all

function K = genGaussianKernel(X, Y, sigma)

    dist = pdist2(X, Y).^2;
    K = exp(-dist/(2*sigma^2));
end


load("mnist-matlab\mnist.mat");

trainImages = training.images;
trainImages = double(trainImages) / 255.0;

testImages = test.images;
testImages = double(testImages) / 255.0;

testLabels = test.labels;
trainLabels = training.labels;

width = test.width;
height = test.height;

numTrain = 10000;
numTest = 500;
trainImages = trainImages(:, : ,1:numTrain);
testImages = testImages(:,:, 1:numTest);

trainImages = reshape(trainImages, [], size(trainImages, 3)); %Flatten the Image dimension
testImages = reshape(testImages, [], size(testImages, 3));

%Reshape for compatability in pdist2 in genGaussianKernel
trainImages = trainImages';
testImages = testImages';

trainLabels = trainLabels(1:numTrain);
testLabels = testLabels(1:numTest);


% Trying to see if KRR can figure out 2 and 7
bool27 = or(testLabels == 2, testLabels == 7);
testLabels = double(testLabels(bool27) > 2);
testImages = testImages(bool27, :);

bool27 = or(trainLabels == 2, trainLabels == 7);
trainLabels = double(trainLabels(bool27) > 2);
trainImages = trainImages(bool27, :);


mu = 1e-7;
sigma = 5;
pcg_tol = 1e-16;

train_kernel = genGaussianKernel(trainImages, trainImages, sigma);
train_kernel = train_kernel + mu*eye(size(train_kernel, 1));
fprintf("Running RPCholesky...\n")
[~, ~, U, L] = rpCholeskey(train_kernel, 500);
fprintf("Constructing Preconditioner...\n")
P = genPreconditionerUL(U, L, mu);
fprintf("Performimg PCG...\n")
alpha = pcg(train_kernel, trainLabels, pcg_tol, 1000, P);

test_kernel = genGaussianKernel(testImages, trainImages ,sigma);
prediction = test_kernel * alpha;
predictedClasses = prediction > 0.5;
accuracy = mean(predictedClasses == testLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

