# from environment_utils import *
from env_utils import *
from bot_movement import *
import sys
import os
import numpy as np
import random
from Bot_Improved import *

n = 30
# alpha = random.uniform(0.02, 0.2)
alpha = 0.1
print(alpha)

seed_value = random.randrange(1, 1000)
# seed_value = 808
# seed_value = 259
# seed_value = 565
# seed_value = 304
seed_value = 457
print(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

grid = grid_init(n)
bot_pos = bot_init(grid, n, 3)
rat_pos = rat_init(grid, n, 2)

frames_bot1 = []

print(grid)

grid_array = np.array(grid)

# Define colors for visualization
cmap = plt.cm.get_cmap('coolwarm', 2)  # 2 discrete colors
bounds = [-1.5, -0.5, 0.5]  # Range of values
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# Create the plot
plt.figure(figsize=(6, 6))
plt.imshow(grid_array, cmap=cmap, norm=norm)

# Add gridlines
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.xticks(range(len(grid[0])))
plt.yticks(range(len(grid)))

# Show the visualization
plt.show()


print("Bot 1")
random.seed(seed_value)
np.random.seed(seed_value)

#Localization Function - main_function - Run this for localization
# bot_pos, data_log = main_function(grid, n, bot_pos)

# print(data_log)


#Baseline bot Function - main_function_catching - Run this for stationary rat, baseline bot
# rat_caught = main_function_catching(grid, n, bot_pos, rat_pos, alpha, 5, seed_value, False)


#Baseline bot Function - main_function_catching_moving_rat - Run this for moving rat with the baseline logic
# rat_caught = main_function_catching_moving_rat(grid, n, bot_pos, rat_pos, alpha, 5, seed_value, False)

#Improved bot Function - main_improved - Run this for stationary rat, Modified bot
rat_caught = main_improved(grid, n, bot_pos, rat_pos, alpha, 5, seed_value, False) #data_log

#Improved bot Function - main_improved_with_moving_rat - Run this for moving rat with modified logic
# rat_caught = main_improved_with_moving_rat(grid, n, bot_pos, rat_pos, alpha, 5, seed_value, False)
