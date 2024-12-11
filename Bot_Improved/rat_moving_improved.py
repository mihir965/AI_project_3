from env_utils import *
from bot_movement import *
import math
import random
import numpy as np

def log_simulation_result(simulation_num, seed, alpha, outcome):
    with open("simulation_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([simulation_num, seed, f"{alpha:.2f}", outcome])

#The rat can only occupy an open cell
def list_possible_cells(grid, n):
    possible_cells = []
    for i in range(n):
        for j in range(n):
            #0 = open 2 = Rat 3 = Bot
            if grid[i][j] == 0 or grid[i][j]==2:
                possible_cells.append((i, j))
    return possible_cells

def get_valid_rat_moves(grid, rat_pos):
    n = len(grid)
    possible_moves = []
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    for dx, dy in directions:
        new_x, new_y = rat_pos[0] + dx, rat_pos[1] + dy
        if (is_valid(new_x, new_y, n) and 
            is_unblocked(grid, new_x, new_y)):
            possible_moves.append((new_x, new_y))
    
    # Include current position as the rat might stay
    possible_moves.append(rat_pos)
    return possible_moves

def simulate_rat_movement(grid, rat_pos):
    valid_moves = get_valid_rat_moves(grid, rat_pos)
    new_pos = random.choice(valid_moves)
    if new_pos != rat_pos:
        grid[rat_pos[0]][rat_pos[1]] = 0
        grid[new_pos[0]][new_pos[1]] = 2
    return grid, new_pos

def is_possible_cell(grid, i, j):
    return grid[i][j] == 0 or grid[i][j] == 2

def manhattan_dist(bot_pos, j):
    return abs(bot_pos[0] - j[0]) + abs(bot_pos[1] - j[1])

#P(ping|d(i,j)) = e^{-a(d(i,j)-1)}
def prob_ping_rat(bot_pos, rat_pos, alpha):
    # dist = manhattan_dist(bot_pos, rat_pos)
    dist = max(1, manhattan_dist(bot_pos, rat_pos))
    exponent = -1*(alpha*(dist-1))
    prob = math.exp(exponent)
    rand_chance = random.uniform(0.0, 1.0)
    # print(prob)
    # print(rand_chance)
    if prob>=rand_chance:
        return True
    else:
        return False
    
def prob_ping_j(bot_pos, j, alpha):
    # dist = manhattan_dist(bot_pos, j)
    dist = max(1, manhattan_dist(bot_pos, j))
    exponent = -1*(alpha*(dist-1))
    prob = math.exp(exponent)
    return prob

def init_prob_cells(grid, n, list_poss_cells):
    new_grid = np.copy(grid)
    num_possible_cells = len(list_poss_cells)
    # print(num_possible_cells)
    init_value = 1/num_possible_cells
    # print(init_value)
    for cell in list_poss_cells:
        new_grid[cell[0]][cell[1]] = init_value
    return new_grid

#We will partition the grid into 4 quadrants
def partition_grid(grid, n):
    quadrants = {
        'Q1': [],
        'Q2': [],
        'Q3': [],
        'Q4': []
    }
    mid = n//2
    for i in range(mid):
        for j in range(mid):
            if is_possible_cell(grid, i, j):
                quadrants['Q1'].append((i, j))

    for i in range(mid):
        for j in range(mid, n):
            if is_possible_cell(grid, i, j):
                quadrants['Q2'].append((i, j))
    
    for i in range(mid, n):
        for j in range(mid):
            if is_possible_cell(grid, i, j):
                quadrants['Q3'].append((i,j))
    
    for i in range(mid, n):
        for j in range(mid, n):
            if is_possible_cell(grid, i, j):
                quadrants['Q4'].append((i, j))
    
    return quadrants

#Updating probabilities of grid cells based on whether the ping was heard or not. We aggregate the probablity for the quadrant and return to pick the maximum one
def update_probabilities(prob_grid, prob_grid_dict, alpha, hear_prob, bot_pos, grid):
    quadrant_probabilities = {quadrant: 0.0 for quadrant in prob_grid_dict.keys()}
    total_prob_sensor = 0.0
    new_prob_grid = np.copy(prob_grid)

    if hear_prob:
        for quadrant, cells in prob_grid_dict.items():
            for cell in cells:
                total_prob_sensor += prob_ping_j(bot_pos, cell, alpha) * new_prob_grid[cell[0]][cell[1]]
    else:
        for quadrant, cells in prob_grid_dict.items():
            for cell in cells:
                total_prob_sensor += (1 - prob_ping_j(bot_pos, cell, alpha)) * new_prob_grid[cell[0]][cell[1]]

    if total_prob_sensor != 0:
        if hear_prob:
            for quadrant, cells in prob_grid_dict.items():
                for cell in cells:
                    p_ping_j_val = prob_ping_j(bot_pos, cell, alpha)
                    updated_p = (p_ping_j_val * new_prob_grid[cell[0]][cell[1]]) / total_prob_sensor
                    new_prob_grid[cell[0]][cell[1]] = updated_p
                    quadrant_probabilities[quadrant] += updated_p
        else:
            for quadrant, cells in prob_grid_dict.items():
                for cell in cells:
                    p_ping_j_val = (1 - prob_ping_j(bot_pos, cell, alpha))
                    updated_p = (p_ping_j_val * new_prob_grid[cell[0]][cell[1]]) / total_prob_sensor
                    new_prob_grid[cell[0]][cell[1]] = updated_p
                    quadrant_probabilities[quadrant] += updated_p

    # Normalize quadrant probabilities
    total_quadrant_prob = sum(quadrant_probabilities.values())
    if total_quadrant_prob > 0:
        for quadrant in quadrant_probabilities:
            quadrant_probabilities[quadrant] /= total_quadrant_prob

    # Update probabilities based on rat movement
    new_prob_grid = update_prob_after_movement(new_prob_grid, grid)

    return quadrant_probabilities, new_prob_grid

#Distributing the probability based on the valid moves the rat can take
def update_prob_after_movement(prob_grid, grid):
    n = len(prob_grid)
    new_prob_grid = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if prob_grid[i][j] > 0:
                valid_moves = get_valid_rat_moves(grid, (i, j))
                # Distribute probability equally among possible next positions
                move_prob = prob_grid[i][j] / len(valid_moves)
                for move in valid_moves:
                    new_prob_grid[move[0]][move[1]] += move_prob
    return new_prob_grid

#This is for selecting the target cell within the target quadrant
def weighted_center(target_quadrant, prob_grid_dict, prob_grid, grid_for_map, n):
    sum_x = 0.0
    sum_y = 0.0
    sum_p = 0.0

    for cell in prob_grid_dict[target_quadrant]:
        i, j = cell
        p = prob_grid[i][j]
        sum_x += p * i
        sum_y += p * j
        sum_p += p

    if sum_p > 0:
        x_c = sum_x/sum_p
        y_c = sum_y/sum_p
        x_final = int(round(x_c))
        y_final = int(round(y_c))
        if grid_for_map[x_final][y_final] == -1:
            print("invalid")
            cardinal_directions = [(0,1), (0,-1), (1,0), (-1,0)]
            for i, j in cardinal_directions:
                    while is_valid(x_final, y_final, n) and not is_unblocked(grid_for_map, x_final, y_final):
                        x_final, y_final = x_final+i, y_final+j
        target_cell = (x_final, y_final)
    else:
        target_cell = random.choice(prob_grid_dict[target_quadrant]) if prob_grid_dict[target_quadrant] else None

    return target_cell   

#We move halfway first, then if consistency is there, we move all the way. t is updated here since we are not alternating between movement and sensing.
def movement(grid_for_map, target_cell, bot_pos, n, frames_grid, rat_pos, t, prob_grid):
    path = plan_path_bot2(grid_for_map, bot_pos, target_cell, n)
    if path and len(path)>1:
        halfway = len(path)//2
        # print(path)
        # print(f"Halfway is :{path[halfway]}")
        halfway_coordinates = path[halfway]
        # print(target_cell)
        # print(path[len(path)-1])
        # final_coordinates = path[len(path)-1]
        while True:
            t+=1
            #The rat will also be moving with every time step
            grid_for_map, rat_pos = simulate_rat_movement(grid_for_map, rat_pos)
            #We need to update the probabilities as well with the movement of the rat
            prob_grid = update_prob_after_movement(prob_grid, grid_for_map)
            # old_pos = bot_pos
            grid_for_map[bot_pos[0]][bot_pos[1]] = 0
            bot_pos = path.pop(1)
            if grid_for_map[bot_pos[0]][bot_pos[1]] == 2:
                print("The rat was caught!")
                grid_for_map[bot_pos[0]][bot_pos[1]] = 3
                break
            grid_for_map[bot_pos[0]][bot_pos[1]] = 3
            # print(f"Moving bot from {old_pos} to {bot_pos}")
            if bot_pos==halfway_coordinates:
                break
            frames_grid.append(np.copy(grid_for_map))
            # print(bot_pos)
        # print(f"Moving bot from {old_pos} to {bot_pos}")
        frames_grid.append(np.copy(grid_for_map))
        # print(bot_pos)
        return bot_pos, frames_grid, t, prob_grid, rat_pos
    else:
        print(f"The position of the rat is {rat_pos}\nThe bot is at {bot_pos}")
        print(f"No valid path found to target cell: {target_cell}")
        return False, frames_grid, t, prob_grid, rat_pos

#Once a quadrant has been selected as consistent, we refine it to narrow down the search and repeat the entire process.
def refine_quadrants(parent_quadrant_name, quadrant_cells, n):
    print("Refine Quadrants ran")
    if not quadrant_cells:
        print("No cells to refine in the quadrant.")
        return {}
    
    avg_row = sum(i for i, j in quadrant_cells) / len(quadrant_cells)
    avg_col = sum(j for i, j in quadrant_cells) / len(quadrant_cells)
    
    refined_quadrants = {
        f"{parent_quadrant_name}_Q1": [],
        f"{parent_quadrant_name}_Q2": [],
        f"{parent_quadrant_name}_Q3": [],
        f"{parent_quadrant_name}_Q4": []
    }

    for cell in quadrant_cells:
        i, j = cell
        if i < avg_row and j < avg_col:
            refined_quadrants[f'{parent_quadrant_name}_Q1'].append(cell)
        elif i < avg_row and j >= avg_col:
            refined_quadrants[f'{parent_quadrant_name}_Q2'].append(cell)
        elif i >= avg_row and j < avg_col:
            refined_quadrants[f'{parent_quadrant_name}_Q3'].append(cell)
        else:
            refined_quadrants[f'{parent_quadrant_name}_Q4'].append(cell)
    
    refined_quadrants = {k: v for k, v in refined_quadrants.items() if v}
    
    print(f"Refined Quadrants: {list(refined_quadrants.keys())}")
    return refined_quadrants

#Sometimes, the bot would be stuck on the target cell, while being very close to the rat, This is a last ditch attempt to get the rat
def last_ditch_check_neighbours(bot_pos, grid_for_map, n, frames_grid, t):
    cardinality = [[1,0],[-1,0],[0,1],[0,-1]]
    for i, j in cardinality:
        new_i, new_j = bot_pos[0]+i, bot_pos[1]+j
        if is_valid(new_i, new_j, n):
            t+=1
            bot_pos = (new_i, new_j)
            if grid_for_map[bot_pos[0]][[bot_pos[1]]]==2:
                grid_for_map[bot_pos[0]][bot_pos[1]] = 3
                print("The rat was caught in a last ditch attempt!")
                frames_grid.append(np.copy(grid_for_map))
                return bot_pos, frames_grid
    return bot_pos, frames_grid

def main_improved_with_moving_rat(grid, n, bot_pos, rat_pos, alpha, simulation_num, seed_value, driver_comparison): # simulation_num, seed_value
    frames_grid = []
    frames_heatmap = []
    grid_for_map = np.copy(grid)
    frames_grid.append(np.copy(grid_for_map))
    grid_for_prob = np.zeros_like(grid, dtype=float)
    init_kb = list_possible_cells(grid_for_map, n)
    prob_grid = init_prob_cells(grid_for_prob, n, init_kb)
    frames_heatmap.append(np.copy(prob_grid))
    quadrants = partition_grid(grid_for_map, n)
    t = 0
    current_quadrant = None
    while True:
        old_rat_pos = rat_pos
        grid_for_map, rat_pos = simulate_rat_movement(grid_for_map, rat_pos)
        print(f"Step {t}: Rat moved from {old_rat_pos} to {rat_pos}")

        frames_grid.append(np.copy(grid_for_map))

        if rat_pos == bot_pos:
            print("The rat ran into the bot!!")
            if not driver_comparison:
                visualize_simulation_1(frames_grid)
            return False
    
        # Update probabilities based on sensor and rat movement
        hear_prob = prob_ping_rat(bot_pos, rat_pos, alpha)
        quadrant_probabilities, prob_grid = update_probabilities(
            prob_grid, quadrants, alpha, hear_prob, bot_pos, grid
        )

        # Determine target quadrant and weighted center
        target_quadrant = max(quadrant_probabilities, key=quadrant_probabilities.get)
        print(f"The target quadrant is: {target_quadrant}")
        target_cell = weighted_center(target_quadrant, quadrants, prob_grid, grid_for_map, n)
        print(f"Weighted center of {target_quadrant}: {target_cell}")
        if target_cell and 0 <= target_cell[0] < n and 0 <= target_cell[1] < n and grid_for_map[target_cell[0]][target_cell[1]] != -1:
            if current_quadrant == target_quadrant:
                print("Consistency within the same quadrant.")
                if bot_pos == target_cell:
                    t+=1
                    bot_pos, frames_grid = last_ditch_check_neighbours(bot_pos, grid_for_map, n, frames_grid, t)
                    if grid_for_map[bot_pos[0]][bot_pos[1]]==2:
                        print("The rat was caught!!")
                        print(f"Steps taken: {t}")
                        log_simulation_result(simulation_num, seed_value, alpha, "Success")
                        frames_grid.append(np.copy(grid_for_map))
                        if not driver_comparison:
                            visualize_simulation_1(frames_grid)
                        return True
                    print("Bot already at the target cell.")
                    if t>2000:
                        print("timeout")
                        log_simulation_result(simulation_num, seed_value, alpha, "Failure")
                        if not driver_comparison:
                            visualize_simulation_1(frames_grid)
                        return False
                    else:
                        continue
                # Move the bot to the target cell
                bot_pos, frames_grid, t, prob_grid, rat_pos = movement(grid_for_map, target_cell, bot_pos, n, frames_grid, rat_pos, t, prob_grid)
                # Refine the target quadrant
                refined_quadrants = refine_quadrants(target_quadrant, quadrants[target_quadrant], n)
                if refined_quadrants:
                    del quadrants[target_quadrant]
                    quadrants.update(refined_quadrants)
                    print(f"Refined {target_quadrant} into {list(refined_quadrants.keys())}")
            else:
                print("Switching to a new quadrant.")
                if bot_pos == target_cell:
                    t+=1
                    bot_pos, frames_grid = last_ditch_check_neighbours(bot_pos, grid_for_map, n, frames_grid, t)
                    if grid_for_map[bot_pos[0]][bot_pos[1]]==2:
                        print("The rat was caught!!")
                        print(f"Steps taken: {t}")
                        log_simulation_result(simulation_num, seed_value, alpha, "Success")
                        frames_grid.append(np.copy(grid_for_map))
                        if not driver_comparison:
                            visualize_simulation_1(frames_grid)
                        return True
                    print("Bot already at the target cell.")
                    if t>2000:
                        print("timeout")
                        log_simulation_result(simulation_num, seed_value, alpha, "Failure")
                        if not driver_comparison:
                            visualize_simulation_1(frames_grid)
                        return False
                    else:
                        continue
                bot_pos, frames_grid, t, prob_grid, rat_pos = movement(grid_for_map, target_cell, bot_pos, n, frames_grid, rat_pos, t, prob_grid)
                if bot_pos is False:
                    print("Bot movement failed.")
                    break
                current_quadrant = target_quadrant
        else:
            print(f"Invalid target cell selected: {target_cell}")
            # print(grid_for_map[target_cell[0]][target_cell[1]])

        # Check if bot catches the rat
        if bot_pos == rat_pos:
            print(f"Bot caught the rat at {bot_pos} in {t} steps!")
            log_simulation_result(simulation_num, seed_value, alpha, "Success")
            frames_grid.append(np.copy(grid_for_map)) 
            if not driver_comparison:
                visualize_simulation_1(frames_grid)
            return True

        # Timeout condition
        if t > 2000:
            print("Timeout: Bot took too long.")
            log_simulation_result(simulation_num, seed_value, alpha, "Failure")
            if not driver_comparison:
                visualize_simulation_1(frames_grid)
            return False