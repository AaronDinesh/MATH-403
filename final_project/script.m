function isToolboxInstalled(toolboxName)
    % Get a list of installed toolboxes
    v = ver;
    isInstalled = any(strcmp(toolboxName, {v.Name}));
    err_str = sprintf("The %s Toolbox is not installed", toolboxName);
    assert(isInstalled, err_str)
end

isToolboxInstalled('Statistics and Machine Learning Toolbox')


function A = genGaussianKernel(sigma, n, mu)
    assert(sigma > 0 && sigma <= 1, "Sigma is not in the range 0-1")
    % X = rand(n, n)*2 -1;
    % Y = rand(n,n)*2 -1;
    % A = exp(-((X.^2 + Y.^2) / (2 * sigma^2)));

    x = rand(n, 1);
    A = exp(-((x - x').^2) / (2 * sigma^2));

    A = A + mu*eye(n);
end

function [S, A_hat] = rpCholeskey(A, k)
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
    A_hat = F*F';
end

%Following notation in L6-S58
function A_hat = nystromApprox(A, r)
    p = 5;
    N = size(A, 2);
    omega = randn(N, r+p);
    Y = A * omega;
    A_hat = Y*pinv(omega'*Y)*Y';
end

function P = genPreconditioner(A_hat, mu)
    N = size(A_hat, 1);
    [U, D] = eig(A_hat);
    lastLambda = D(end);
    P = (1/(lastLambda + mu)) * U*(D+mu*eye(N))*U' + (eye(N) - U*U'); 
end

A = genGaussianKernel(0.5, 100, 0.1);
nystromApprox(A, 80);
P = genPreconditioner(A, 0.1)






