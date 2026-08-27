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
        if delta < atol: 
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

if __name__ == '__main__':
    # 1. function parameters
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    MAX_ITER = 20_000  
    ABS_TOL = 1e-4    
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    # 2. parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <N> <num_workers>")
        sys.exit(1)
        
    N = int(sys.argv[1])
    num_workers = int(sys.argv[2])

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    # 3. Static Scheduling
    # assign same amount of building_ids to worker 
    chunks = np.array_split(building_ids, num_workers)

    # 4. Each worker processes its chunk of building IDs, including loading data, running jacobi, and computing stats 
   
    with mp.Pool(processes=num_workers) as pool:
        args = [(chunk, LOAD_DIR, MAX_ITER, ABS_TOL) for chunk in chunks]
        # starmap wraps args tuples to process_building_batch
        list_of_batches = pool.starmap(process_building_batch, args)

    # 5. flatten the list of batches and print CSV output
    print('building_id,' + ','.join(stat_keys))
    for batch in list_of_batches:
        for bid, stats in batch:
            row = [str(stats[k]) for k in stat_keys]
            print(f"{bid}," + ",".join(row))