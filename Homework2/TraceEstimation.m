function TraceEstimation()
    f = waitbar(0,'Starting Computation...');
    cs = [0.5, 1.0, 1.5, 2];
    number_of_steps = 100;
    % This nu
    log_stop_number = 4;
    err_gh = zeros(length(cs), number_of_steps);
    err_igh = zeros(length(cs),number_of_steps);

    for i=1:length(cs)
        rand_mat = randn(1000, 1000);
        [Q, ~] = qr(rand_mat);
        A = Q' * diag((1:1000).^-cs(i))*Q;
        tra = trace(A);
        iteration_arr = round(logspace(1, log_stop_number, number_of_steps));
        for k=1:length(iteration_arr)
            gh_tr_est = gh(A, iteration_arr(k));
            igh_tr_est = igh(A, iteration_arr(k));
            err_gh_val = abs(tra - gh_tr_est)/tra;
            err_igh_val = abs(tra - igh_tr_est)/tra;
            
            if (err_gh_val==0); err_gh(i, k) = 1e-16; else; err_gh(i, k) = err_gh_val; end
            if(err_gh_val ==0); err_igh(i, k) = 1e-16; else; err_igh(i, k) = err_igh_val; end
        end
        
        fig = figure;
        loglog(iteration_arr, err_gh(i, :),'-s', 'DisplayName','GH');
        hold on
        loglog(iteration_arr, err_igh(i, :),'-o', 'DisplayName', 'IGH');
        ax = gca;
        ax.YGrid = 'on';
        ax.YMinorGrid = 'on';
        ax.XGrid = 'on';
        ax.XMinorGrid = 'on';
        legend('Location','northeast');
        xlabel('Number of Iterations (N)');
        ylabel('Relative Error');
        title(sprintf('Trace Estimation Algorithm Comparison for c=%g', cs(i)));
        hold off;
        saveas(gcf, sprintf('tr_est_plot_c_%g.png', cs(i)))
        waitbar(i/length(cs), f, sprintf("Completed c=%g", cs(i)))
        close(fig)
    end
    close(f)
end

function tr = gh(A, N)
    h = waitbar(0, "Running GH Trace Estimation...");
    trace = 0;
    for i=1:1:N
        rand_vec = randn(size(A, 1), 1);
        trace = trace + rand_vec' * A * rand_vec;
        waitbar(i/N, h, sprintf('GH Iteration: %g/%g', i, N))
    end
    tr = trace/N;
    close(h);
end

function tr = igh(A, N)
    g = waitbar(0, "Running Improved Trace Estimation...");
    k = N;
    p = k + 1;
    l = 2*(k+1);
    OMEGA = randn(size(A, 1), k+p);
    PSI = randn(size(A, 1), k+p+l);
    [Q, ~] = qr(A*OMEGA, 0);
    %Calculating the pinv by using the \ operator and solving the linear
    %system
    P_AOmegaPsi = Q*((PSI'*Q) \ PSI');
    
    %Clearing some variables to save memory
    clear OMEGA;
    clear PSI;
    clear Q;
    X = P_AOmegaPsi*A;
    trx = trace(X);
    trace_est_mat = A - X;
    trace_est = 0;
    for i=1:1:N
        rand_vec = randn(size(trace_est_mat, 1), 1);
        trace_est = trace_est + rand_vec' * trace_est_mat * rand_vec;
        waitbar(i/N, g, sprintf('Improved GH Iteration: %g/%g', i, N))
    end
    tr = (trace_est/N) + trx; 
    close(g);
end