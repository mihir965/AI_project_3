import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import heapq
import matplotlib.animation
import os
import csv

# outlines the parameters that define each cell
class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0

# checking if cell is blocked
def is_blocked(grid, row, col):
    return grid[row][col] == -1

def is_unblocked(grid, row, col):
    return grid[row][col] != -1

# checking if cell is within the defined boundary
def is_valid(row, col, n):
    return (row >= 0) and (row < n) and (col >= 0) and (col < n)

# calculating how far i and j are (manhattan distance)
def calculate_d_value(row, col, dest):
    return abs(row - dest[0]) + abs(col - dest[1])

# if bot has reached the rat
def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

# number of open neighbours each cell has
def count_open_neighbours(grid, i, j):
    movement = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    open_neighbours = 0
    for mi, mj in movement:
        test_i, test_j = i + mi, j + mj
        if grid[test_i][test_j] == 0:
            open_neighbours += 1
    return open_neighbours

def blocked_cells(grid, n):
    blocked_cells = []
    for i in range(1, n-1):
        for j in range(1, n-1):
            if grid[i][j] == -1:
                open_neighbours = count_open_neighbours(grid, i, j)
                if open_neighbours == 1:
                    blocked_cells.append((i, j))
    return blocked_cells

def dead_end_cells(grid, n):
    dead_ends = []
    for i in range(1, n-1):
        for j in range(1, n-1):
            if grid[i][j] == 0 and count_open_neighbours(grid, i, j) == 1:
                dead_ends.append((i, j))
    return dead_ends

def grid_init(n):
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    init_x, init_y = random.randint(1, n-2), random.randint(1, n-2)
    grid[init_x][init_y] = 0
    while True:
        b_cells = blocked_cells(grid=grid, n=n)
        if not b_cells:
            break
        i, j = random.choice(b_cells)
        grid[i][j] = 0

    # d_cells = dead_end_cells(grid, n)
    # num_to_open = len(d_cells) // 2
    # for _ in range(num_to_open):
    #     if not d_cells:
    #         break
    #     i, j = random.choice(d_cells)
    #     d_cells.remove((i, j))
    #     movement = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    #     closed_neighbors = [(i + mi, j + mj) for mi, mj in movement if grid[i + mi][j + mj] == 1]
    #     if closed_neighbors:
    #         ni, nj = random.choice(closed_neighbors)
    #         grid[ni][nj] = 0
    grid_matplot = np.array(grid)
    return grid_matplot

def place_element(grid, n, value):
    if value == 2:
        cannot = 3
    elif value == 3:
        cannot = 2
    while True:
        x, y = random.randint(1, n-2), random.randint(1, n-2)
        if (grid[x][y] != -1 and grid[x][y] != cannot):
            grid[x][y] = value
            return x, y
    
def bot_init(grid, n, v):
    x, y = place_element(grid, n, v)
    return x, y

def rat_init(grid, n, v):
    x, y = place_element(grid, n, v)
    return x, y

def visualize_simulation_1(frames, interval=100):
    cmap = ListedColormap(['black', 'white', 'red', 'blue'])
    boundaries = [-1.5, -0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    fig, ax = plt.subplots()
    ax.set_title('Grid Simulation')
    mat = ax.matshow(frames[0], cmap=cmap, norm=norm)
    def update(frame):
        mat.set_data(frame)
        return [mat]
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation
from matplotlib.colors import ListedColormap, Normalize

def visualize_simulation_2(frames, interval=100):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title('Grid Simulation')

    # Prepare the first frame
    first_frame = frames[0]

    # Mask the blocked cells (-1)
    masked_frame = np.ma.masked_where(first_frame == -1, first_frame)

    # Set the colormap for probabilities
    cmap = plt.cm.viridis  # You can choose any continuous colormap you prefer
    cmap.set_bad(color='black')  # Set the color for masked (blocked) cells

    # Normalize the probabilities between 0 and 1
    norm = Normalize(vmin=0.0, vmax=0.005)

    # Plot the initial frame
    mat = ax.imshow(masked_frame, cmap=cmap, norm=norm)

    # Optionally, add a colorbar to show the probability scale
    fig.colorbar(mat, ax=ax, label='Probability')

    # Function to update each frame
    def update(frame):
        # Mask the blocked cells in the current frame
        masked_frame = np.ma.masked_where(frame == -1, frame)
        mat.set_data(masked_frame)
        return [mat]

    # Create the animation
    ani = matplotlib.animation.FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=True
    )

    plt.show()


# saves all the results for each simulation
def log_results(log_data, filename='/Users/drcrocs22/Developer/Rutgers Projects/Intro To AI/PROJECT_1_FINAL/simulation_results.csv'):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Bot Type','run_id' ,'q', 'bot_pos_init', 'button_pos_init', 'fire_init', 'steps', 'result', 'final_frame','seed_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

def save_final_frame(frame, filename='final_frame.png'):
    cmap = ListedColormap(['white', 'black', 'red', 'blue', 'green'])
    plt.figure()
    plt.title('Final Frame')
    plt.imshow(frame, cmap=cmap, vmin=0, vmax=4)
    plt.axis('off')  # Hide axis ticks and labels
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
