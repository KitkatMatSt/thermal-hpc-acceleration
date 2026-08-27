%%time
from os.path import join
import matplotlib.pyplot as plt
import os
import numpy as np

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

# --- 2. Main Logic ---
# The IDs you have downloaded to your Mac
building_ids = ['10000', '10009', '11117', '10082', '10029', '10064']
MAX_ITER = 20_000
ABS_TOL = 1e-4
print("Starting batch visualization process...")

for bid in building_ids:
    if not os.path.exists(f"{bid}_domain.npy"):
        print(f"Warning: Files for building {bid} not found. Skipping...")
        continue
    
    print(f"Simulating Building {bid} (this may take a moment)...")
    u0, mask = load_data('./', bid)
    u_final = jacobi(u0, mask, MAX_ITER, ABS_TOL)

    # --- 3. Plotting the Heatmap ---
    plt.figure(figsize=(7, 6))
    
    # 'RdBu_r' is 'Red-Blue Reversed': Blue is cold (5°C), Red is hot (25°C)
    im = plt.imshow(u_final[1:-1, 1:-1], cmap='RdBu_r', vmin=5, vmax=25)
    
    plt.title(f"Steady-State Heat Distribution: {bid}", fontsize=14)
    plt.colorbar(im, label='Temperature (°C)')
    plt.axis('off') # Remove pixel coordinates for a cleaner look
    
    # Save the plot with high quality (DPI)
    filename = f'task3_heatmap_{bid}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Successfully saved: {filename}")
    plt.close() # Important: Close plot to free up memory

print("\nAll tasks completed. You can now open the .png files in your folder!")