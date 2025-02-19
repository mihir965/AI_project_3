from env_utils import *
from bot_movement import *
import sys
import os
import numpy as np
import random
from Bot_Improved import *
import csv
from collections import defaultdict

n = 30

def run_single_simulation(alpha, simulation_num, seed_value):
    grid = np.loadtxt('grid.txt', dtype=int)
    bot_pos = bot_init(grid, n, 3)
    rat_pos = rat_init(grid, n, 2)

    grid_array = np.array(grid)

# # Define colors for each value in the grid
#     cmap = plt.matplotlib.colors.ListedColormap(['white', 'black', 'green', 'red'])  # Colors for 0, -1, 2, 3
#     bounds = [-1.5, -0.5, 0.5, 1.5, 3.5]  # Boundaries for -1, 0, 2, 3
#     norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

#     # Plot the grid
#     plt.figure(figsize=(10, 10))
#     plt.imshow(grid_array, cmap=cmap, norm=norm)

#     # Add gridlines for better visualization
#     plt.grid(color='gray', linestyle='-', linewidth=0.5)
#     plt.xticks(np.arange(-0.5, len(grid[0]), 1), [])
#     plt.yticks(np.arange(-0.5, len(grid), 1), [])
#     plt.gca().set_xticks(np.arange(-0.5, len(grid[0])), minor=True)
#     plt.gca().set_yticks(np.arange(-0.5, len(grid)), minor=True)
#     plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

#     # Show the plot
#     plt.show()


    grid_for_use = np.copy(grid)
    while True:
        bot_pos, data_log = main_function(grid_for_use, n, bot_pos)
        if not bot_pos:
            return False
        simulation_result = main_improved(grid_for_use, n, bot_pos, rat_pos, alpha, simulation_num, seed_value, data_log, True) #data_log
        if simulation_result==False:
            print("The return was False")
            return False
        if simulation_result:
            return simulation_result
        
def save_simulation_data(seed_value, total_data):
    filename = f"./data/seed_{seed_value}_{random.randint(0,100000)}.npz"
    # filename = f"./data/moving_{random.randint(0,10000)}.npz"
    np.savez_compressed(
        filename,
        bot_grid = [entry["bot_prob_grid"] for entry in total_data],
        rat_grid = [entry["rat_prob_grid"] for entry in total_data],
        time_step_remaining = [entry["remaining_steps"] for entry in total_data],
        blocked_ratio = [entry["blocked_ratio"] for entry in total_data],
        time_step = [entry["t"] for entry in total_data],
        dist_to_target = [entry["dist_to_target"] for entry in total_data],
        target_cell_prob = [entry["prob_target_cell"] for entry in total_data],
        max_prob = [entry["most_probable_cell_prob"] for entry in total_data]
    )
    print(f"Simulation data saved to {filename}")

def run_comparisons(alpha = 0.08, simulations=1000):#5000
    seed_value = 457
    np.random.seed(seed_value)
    total_data = []
    for sim_num in range(1, simulations+1):
        new_data = run_single_simulation(alpha, sim_num, seed_value)
        if new_data == False:
            continue
        total_data.extend(new_data)
    save_simulation_data(seed_value, total_data)

if __name__ == "__main__":
    run_comparisons()
