import numpy as np
import cupy as cp
import sys
import time
from os.path import join
import pandas as pd


@cp.fuse()
def jacobi_step_fused_with_mask(u_l, u_r, u_u, u_d, u_old, interior_mask):
    """
    Combined kernel to minimize kernel launch overhead and memory allocation.
    Returns u_next for interior points and u_old for boundaries.
    """
    u_next = 0.25 * (u_l + u_r + u_u + u_d)
    # cp.where within a fused kernel is much faster than boolean indexing
    return cp.where(interior_mask, u_next, u_old)

def jacobi_cupy_ultra_optimized(u0, interior_mask, max_iter, atol=1e-4):
    """
    Final optimized solver: uses fusion and reduces synchronization frequency.
    """
    u = cp.asarray(u0)
    mask = cp.asarray(interior_mask)
    
    for i in range(max_iter):
        
        u_l, u_r = u[1:-1, :-2], u[1:-1, 2:]
        u_u, u_d = u[:-2, 1:-1], u[2:, 1:-1]
        u_center = u[1:-1, 1:-1]

     
        u_new_interior = jacobi_step_fused_with_mask(u_l, u_r, u_u, u_d, u_center, mask)
        
        
        if i % 10 == 0:
            
            delta = cp.abs(u_center - u_new_interior).max()
            if delta < atol:
                u[1:-1, 1:-1] = u_new_interior
                break
        
        
        u[1:-1, 1:-1] = u_new_interior
            
    return u.get() 

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(bool)
    return u, interior_mask

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

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        full_list = f.read().splitlines()

    building_ids = full_list[2000:] 

    N = len(building_ids)
    
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    total_start = time.time()

    results_list = []
    
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        
        u = jacobi_cupy_ultra_optimized(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

        stats = summary_stats(u, interior_mask)
        stats['building_id'] = building_ids[i] 
        results_list.append(stats)
        
    total_duration = time.time() - total_start

    
    stat_keys = ['mean_temp','std_temp','pct_above_18','pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    avg_time = total_duration / N

    df = pd.DataFrame(results_list)
    
    cols = ['building_id', 'mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    #df[cols].to_csv('simulation_results.csv', index=False)
    df[cols].to_csv('simulation_results_part2.csv', index=False)
    
    print("\n[Success] All results saved to simulation_results_part2.csv")

    sys.stderr.write(f"\nSummary for Task 10 (Ultra-Fused CuPy):\n")
    sys.stderr.write(f"Processed {N} plans in {total_duration:.2f}s\n")
    sys.stderr.write(f"Average time per plan: {avg_time:.4f}s\n")
    sys.stderr.write(f"Estimated time for 4,571 plans: {(avg_time * 4571) / 3600:.2f} hours\n")