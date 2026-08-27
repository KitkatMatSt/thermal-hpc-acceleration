import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where you saved your .npy files locally
data_dir = './' 

# Get all unique building IDs in the folder
building_ids = sorted(list(set(f.split('_')[0] for f in os.listdir(data_dir) if f.endswith('.npy'))))

for bid in building_ids:
    # Load the domain and interior mask
    domain = np.load(os.path.join(data_dir, f"{bid}_domain.npy")) #
    interior = np.load(os.path.join(data_dir, f"{bid}_interior.npy")) #
    
    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(domain)
    ax[0].set_title(f"Building {bid} - Domain")
    ax[1].imshow(interior, cmap='gray')
    ax[1].set_title(f"Building {bid} - Interior Mask")
    plt.show()