from os.path import join
import sys
import numpy as np
import multiprocessing as mp

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol: # convergence check, see eq (2) to ensure we don't run more iterations than necessary, very important!
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

def process_building_batch(batch_ids, load_dir, max_iter, abs_tol):
    results = []
    for bid in batch_ids:
        u, mask = load_data(load_dir, bid)
        u_final = jacobi(u, mask, max_iter, abs_tol)
        stats = summary_stats(u_final, mask)
        results.append((bid, stats))
    return results

def process_single_building(bid, load_dir, max_iter, abs_tol):
    """Worker function to process exactly one building."""
    u, mask = load_data(load_dir, bid)
    u_final = jacobi(u, mask, max_iter, abs_tol)
    stats = summary_stats(u_final, mask)
    return bid, stats

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    MAX_ITER = 20_000  
    ABS_TOL = 1e-4    
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    if len(sys.argv) < 3:
        print("Usage: python script.py <N> <num_workers>")
        sys.exit(1)
        
    N = int(sys.argv[1])
    num_workers = int(sys.argv[2])

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    # --- Dynamic Scheduling Implementation ---
    # We use pool.starmap on the individual building_ids. 
    # chunksize=1 ensures that each task is handled independently, 
    # providing the best load balancing for varying workloads.
    with mp.Pool(processes=num_workers) as pool:
        # Prepare a list of tuples, one for each building
        task_args = [(bid, LOAD_DIR, MAX_ITER, ABS_TOL) for bid in building_ids]
        
        # starmap distributes these tasks dynamically
        results = pool.starmap(process_single_building, task_args, chunksize=1)

    # Output results in CSV format
    print('building_id,' + ','.join(stat_keys))
    for bid, stats in results:
        row = [str(stats[k]) for k in stat_keys]
        print(f"{bid}," + ",".join(row))