import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm


def gh_worker(A, N, queue):
    trace = 0
    for i in range(N):
        test_vec = np.random.normal(0, 1, (A.shape[0], 1))
        trace += test_vec.T @ A @ test_vec
    queue.put(trace[0, 0])

def gh(A, N):
    num_workers = mp.cpu_count()
    iterations_per_worker = N // num_workers
    queue = mp.Queue()
    processes = []

    for _ in range(num_workers):
        p = mp.Process(target=gh_worker, args=(A, iterations_per_worker, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    total_trace = sum([queue.get() for _ in range(num_workers)])
    return total_trace / N

def better_gh_worker(to_est, N, queue):
    trace = 0
    for i in range(N):
        test_vec = np.random.normal(0, 1, (to_est.shape[0], 1))
        trace += test_vec.T @ to_est @ test_vec
    queue.put(trace[0, 0])

def better_gh(A, N):
    k = N
    p = k + 1
    l = 2 * (k + 1)
    omega = np.random.normal(1, 0, (A.shape[0], k + p))
    psi = np.random.normal(1, 0, (A.shape[0], k + p + l))
    psiAOProduct = psi.T @ A @ omega
    psiAOTransProduct = psiAOProduct.T @ psiAOProduct
    psiAOPinv = np.linalg.solve(psiAOTransProduct, psiAOProduct.T)
    proj = (A@omega) @ psiAOPinv @ psi.T
    X = proj @ A
    to_est = A - X
    tr_x = np.trace(X)

    num_workers = mp.cpu_count()
    iterations_per_worker = N // num_workers
    queue = mp.Queue()
    processes = []

    for _ in range(num_workers):
        p = mp.Process(target=better_gh_worker, args=(to_est, iterations_per_worker, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    total_trace = sum([queue.get() for _ in range(num_workers)])
    tr = (total_trace / N) + tr_x
    return tr

def main():
    cs = [0.5, 1.0, 1.5, 2.0]
    logspace_number = 100    
    logspace_stop = 2
    err_gh = np.zeros((len(cs), logspace_number))
    err_bgh = np.zeros((len(cs), logspace_number))
    for i, c in enumerate(cs):
        random_mat = np.random.normal(0, 1, (1000, 1000))
        Q, _ = np.linalg.qr(random_mat)
        A = Q.T @ np.diag([np.power(x, -c) for x in np.arange(1, 1001, dtype=np.float64)]) @ Q
        del Q
        tr_a = np.trace(A)

        for j, n in enumerate(tqdm(np.logspace(1, logspace_stop, logspace_number), desc=f"Processing c={c}")):
            gh_estimate = gh(A, int(n))
            improved_gh_estimate = better_gh(A, int(n))
            err_gh[i, j] = abs(tr_a - gh_estimate)/gh_estimate
            err_bgh[i, j] = abs(tr_a - improved_gh_estimate)/improved_gh_estimate
        del A

    fig, ax = plt.subplots(2,2, sharex=True)
    lines = []

    i = 0;
    for h in range(2):
        for k in range(2):
            line_gh, = ax[h, k].loglog([int(x) for x in np.logspace(1, logspace_stop, logspace_number)], err_gh[i, :], label="GH")
            line_bgh, = ax[h, k].loglog([int(x) for x in np.logspace(1, logspace_stop, logspace_number)], err_bgh[i, :], label="BGH")
            ax[h, k].set_title(f"Trace Estimation Relative Error c={cs[i]}")
            ax[h, k].legend(loc="upper left")
            ax[h, k].autoscale(enable=True, axis='y', tight=True)
            ax[h, k].grid(True, which="both", linestyle="-", linewidth=0.5)
            ax[h, k].set(xlabel="Iteration Count", ylabel="Relative Error")
            lines.extend([line_gh, line_bgh])
            i+= 1
 

    INTERACTION = False
    if INTERACTION:
        # Add interactive checkbox to toggle line visibility
        from matplotlib.widgets import CheckButtons
        rax = plt.axes([0.8, 0.4, 0.15, 0.2])
        labels = [line.get_label() for line in lines]
        visibility = [line.get_visible() for line in lines]
        check = CheckButtons(rax, labels, visibility)

        def toggle_lines(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            plt.draw()

        check.on_clicked(toggle_lines)

    plt.show()

if __name__ == '__main__':
    main()
