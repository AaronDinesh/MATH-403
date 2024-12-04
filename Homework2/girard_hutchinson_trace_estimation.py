import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def gh(A, N):
    trace = 0 
    for i in range(N):
        test_vec = np.random.normal(size=(A.shape[0],1))
        trace += test_vec.T @ A @ test_vec
        #print("Iteration ", i, ": ", trace[0, 0]/(i+1))
    return trace[0, 0]/N

def better_gh(A, N):
    k = N
    p = k+1
    l = 2*(k+1)
    omega = np.random.rand(A.shape[0], k+p)
    psi = np.random.rand(A.shape[0], k+p+l)
    Q, _ = np.linalg.qr(A@omega)
    del omega
    proj = Q @ np.linalg.pinv(psi.T @ Q) @ psi.T
    del psi
    del Q
    X = proj @ A
    to_est = A - X
    tr_x = np.trace(X)
    del X

    trace = 0
    
    for i in range(N):
        test_vec = np.random.normal(size=(A.shape[0],1))
        trace += test_vec.T @ to_est @ test_vec
        #print("Iteration ", i, ": ", trace[0, 0]/(i+1) + tr_x)
    tr = (trace[0, 0]/N) + tr_x
    return tr
        



def main():
    cs = [0.5, 1.0, 1.5, 2.0]
    #cs = [0.5]
    logspace_number = 50
    logspace_stop = 4
    err = np.zeros((len(cs)*2, logspace_number))
    for i, c in enumerate(cs):
        random_mat = np.random.rand(1000, 1000)
        Q, _ = np.linalg.qr(random_mat)
        A = Q.T @ np.diag([np.power(x, -c) for x in np.arange(1, 1001, dtype=np.float64)]) @ Q 
        print(A.shape)
        del Q
        tr_a = np.trace(A)
        
        for j, n in enumerate(tqdm(np.logspace(1, logspace_stop, logspace_number), desc=f"Processing c:{c}")):
            gh_estimate = gh(A, int(n))
            improved_gh_estimate = better_gh(A, int(n))
            err[i, j] = abs(tr_a-gh_estimate)/gh_estimate
            err[i+1, j] = abs(tr_a-improved_gh_estimate)/improved_gh_estimate
        del A
    
 
    fig, ax = plt.subplots(2,2, sharex=True)
    i = 0;
    for h in range(2):
        for k in range(2):
            line_gh, = ax[h, k].loglog([int(x) for x in np.logspace(1, logspace_stop, logspace_number)], err[i, :], label="GH")
            line_bgh, = ax[h, k].loglog([int(x) for x in np.logspace(1, logspace_stop, logspace_number)], err[i + 1, :], label="BGH")
            ax[h, k].set_title(f"Trace Estimation Relative Error c={c[i]}")
            ax[h, k].legend(loc="upper left")
            ax[h, k].autoscale(enable=True, axis='y', tight=True)
            ax[h, k].grid(True, which="both", linestyle="-", linewidth=0.5)
            ax[h, k].set(xlabel="Iteration Count", ylabel="Relative Error")
            i+= 1


    plt.legend(loc="upper left") 
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.show() 



if __name__ == '__main__':
    main()
