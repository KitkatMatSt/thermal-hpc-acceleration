from os.path import join
import sys
import numpy as np
from numba import jit

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@jit(nopython=True) 
def jacobi_numba(u, interior_mask, max_iter, atol=1e-6):
    u_new = u.copy()
    rows, cols = u.shape

    for n in range(max_iter):
        delta = 0.0
        # Row-major access (i then j) is cache efficient
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if interior_mask[i-1, j-1]:
                    tmp = 0.25 * (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])
                    
                    diff = abs(tmp - u[i, j])
                    if diff > delta:
                        delta = diff
                    
                    u_new[i, j] = tmp
        
        u[:] = u_new[:]
        if delta < atol:
            break
            
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp': u_interior.mean(),
        'std_temp': u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = building_ids[:N]

    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
       u0, interior_mask = load_data(LOAD_DIR, bid)
       all_u0[i] = u0
       all_interior_mask[i] = interior_mask

    # --- ADDED: Warm-up call to compile the JIT function ---
    # This prevents compilation time from skewing your results
    dummy_u = np.zeros((10, 10))
    dummy_mask = np.zeros((8, 8), dtype=bool)
    jacobi_numba(dummy_u, dummy_mask, 1, 1e-4)

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        
        u = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL) 
        all_u[i] = u

    stat_keys = ['mean_temp','std_temp','pct_above_18','pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))