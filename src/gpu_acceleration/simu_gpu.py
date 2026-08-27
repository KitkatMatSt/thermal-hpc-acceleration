from os.path import join
import sys
import numpy as np
from numba import cuda
from time import perf_counter as time

# Force context on the first available device
try:
    cuda.select_device(0)
except Exception as e:
    print(f"CUDA initialization failed: {e}")
    sys.exit(1)

@cuda.jit   # decorator to tell Numba this is a GPU kernel
def jacobi_kernel(u, u_new, mask):  # kernel to perform one Jacobi iteration
    i, j = cuda.grid(2)  # calculate global thread indices in 2D
    if 0 < i < u.shape[0] - 1 and 0 < j < u.shape[1] - 1:
        m_i, m_j = i - 1, j - 1
        if mask[m_i, m_j]:
            u_new[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
        else:
            u_new[i, j] = u[i, j]

def solve_gpu(u_host, mask_host, max_iter, d_u, d_u_new, d_mask): # helper function to manage GPU memory and kernel execution
    d_u.copy_to_device(u_host) # copy initial data to GPU
    d_u_new.copy_to_device(u_host) # we can initialize u_new with the same values as u
    d_mask.copy_to_device(mask_host) # copy mask to GPU
    
    tpb = (16, 16) # threads per block (16x16 = 256 threads per block)
    # blocks per grid to cover the entire array, rounding up to ensure we cover all elements
    bpg = ((u_host.shape[0] + 15) // 16, (u_host.shape[1] + 15) // 16)  # ceil division to get enough blocks to cover the grid
    
    for _ in range(max_iter):
        jacobi_kernel[bpg, tpb](d_u, d_u_new, d_mask) # launch kernel on GPU
        d_u, d_u_new = d_u_new, d_u  # swap references for next iteration
     
    cuda.synchronize() # ensure all GPU work is done before copying back
    return d_u.copy_to_host() # copy final result back to CPU memory

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    target_ids = ids[:N]
    MAX_ITER = 20000

    # These allocations will now succeed in the new environment
    d_u = cuda.device_array((514, 514), dtype=np.float32)  # allocate GPU memory for the temperature grid
    d_u_new = cuda.device_array((514, 514), dtype=np.float32)
    d_mask = cuda.device_array((512, 512), dtype=bool)

    start_time = time()
    for bid in target_ids: # Loop over each building ID and perform the simulation on GPU
        u = np.zeros((514, 514), dtype=np.float32, order='C') # allocate CPU memory for the temperature grid, ensure it's C-contiguous for better GPU performance
        u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy")).astype(np.float32)
        m = np.load(join(LOAD_DIR, f"{bid}_interior.npy")).astype(bool)
        _ = solve_gpu(u, m, MAX_ITER, d_u, d_u_new, d_mask)

    total_time = time() - start_time
    print("-" * 30)
    print(f"Average time per plan: {total_time/N:.4f} s")