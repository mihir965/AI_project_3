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
            if grid[i][j] == 0 or grid[i][j]==2:
                possible_cells.append((i, j))
    return possible_cells

def is_possible_cell(grid, i, j):
    return grid[i][j] == 0 or grid[i][j] == 2

def manhattan_dist(bot_pos, j):
    return abs(bot_pos[0] - j[0]) + abs(bot_pos[1] - j[1])

#P(ping|d(i,j)) = e^{-a(d(i,j)-1)}
def prob_ping_rat(bot_pos, rat_pos, alpha):
    dist = max(1, manhattan_dist(bot_pos, rat_pos))
    exponent = -1*(alpha*(dist-1))
    prob = math.exp(exponent)
    rand_chance = random.uniform(0.0, 1.0)
    if prob>=rand_chance:
        return True
    else:
        return False
    
def prob_ping_j(bot_pos, j, alpha):
    dist = max(1, manhattan_dist(bot_pos, j))
    exponent = -1*(alpha*(dist-1))
    prob = math.exp(exponent)
    return prob

def init_prob_cells(grid, n, list_poss_cells):
    new_grid = np.copy(grid)
    num_possible_cells = len(list_poss_cells)
    init_value = 1/num_possible_cells
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
def update_probabilities(prob_grid, prob_grid_dict, alpha, hear_prob, bot_pos):
    quadrant_probabilities = {quadrant: 0.0 for quadrant in prob_grid_dict.keys()}
    total_prob_sensor = 0.0
    new_prob_grid = np.copy(prob_grid)

    if hear_prob:
        for quadrant, cells in prob_grid_dict.items():
            for cell in cells:
                total_prob_sensor += prob_ping_j(bot_pos, cell, alpha) * new_prob_grid[cell[0]][cell[1]]
    elif not hear_prob:
        for quadrant, cells in prob_grid_dict.items():
            for cell in cells:
                total_prob_sensor += (1-prob_ping_j(bot_pos, cell, alpha)) * new_prob_grid[cell[0]][cell[1]]

    if total_prob_sensor != 0:
        if hear_prob:
            for quadrant, cells in prob_grid_dict.items():
                for cell in cells:
                    p_ping_j_val = prob_ping_j(bot_pos, cell, alpha)
                    updated_p = (p_ping_j_val * new_prob_grid[cell[0]][cell[1]])/total_prob_sensor
                    new_prob_grid[cell[0]][cell[1]] = updated_p
                    quadrant_probabilities[quadrant] += updated_p
        else:
            for quadrant, cells in prob_grid_dict.items():
                for cell in cells:
                    p_ping_j_val = (1 - prob_ping_j(bot_pos, cell, alpha))
                    updated_p = (p_ping_j_val * new_prob_grid[cell[0]][cell[1]])/total_prob_sensor
                    new_prob_grid[cell[0]][cell[1]] = updated_p
                    quadrant_probabilities[quadrant] += updated_p

    #Normalization
    total_quadrant_prob = sum(quadrant_probabilities.values())
    if total_quadrant_prob > 0:
        for quadrant in quadrant_probabilities:
            quadrant_probabilities[quadrant] /= total_quadrant_prob
    else:
        print("Total quadrant probability is zero. Normalization skipped.")
    
    return quadrant_probabilities, new_prob_grid

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
def movement(grid_for_map, target_cell, bot_pos, n, frames_grid, rat_pos, t, moving):
    path = plan_path_bot2(grid_for_map, bot_pos, target_cell, n)
    if path and len(path)>1:
        halfway = len(path)//2
        halfway_coordinates = path[halfway]
        while True:
            t+=1
            grid_for_map[bot_pos[0]][bot_pos[1]] = 0
            bot_pos = path.pop(1)
            moving+=1
            if grid_for_map[bot_pos[0]][bot_pos[1]] == 2:
                print("The rat was caught!")
                grid_for_map[bot_pos[0]][bot_pos[1]] = 3
                break
            grid_for_map[bot_pos[0]][bot_pos[1]] = 3
            if bot_pos==halfway_coordinates:
                break
            frames_grid.append(np.copy(grid_for_map))
        frames_grid.append(np.copy(grid_for_map))
        return bot_pos, frames_grid, t, moving
    else:
        print(f"The position of the rat is {rat_pos}\nThe bot is at {bot_pos}")
        print(f"No valid path found to target cell: {target_cell}")
        return False, frames_grid, t, moving

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
def last_ditch_check_neighbours(bot_pos, grid_for_map, n, frames_grid, t, moving):
    cardinality = [[1,0],[-1,0],[0,1],[0,-1]]
    for i, j in cardinality:
        new_i, new_j = bot_pos[0]+i, bot_pos[1]+j
        if is_valid(new_i, new_j, n):
            t+=1
            moving+=1
            bot_pos = (new_i, new_j)
            if grid_for_map[bot_pos[0]][[bot_pos[1]]]==2:
                grid_for_map[bot_pos[0]][bot_pos[1]] = 3
                print("The rat was caught in a last ditch attempt!")
                frames_grid.append(np.copy(grid_for_map))
                return bot_pos, frames_grid, t, moving
    return bot_pos, frames_grid, t, moving

def save_simulation_data(simulation_num, seed_value, data_log):
    filename = f"simulation_{simulation_num}_seed_{seed_value}.npz"
    np.savez_compressed(
        filename,
        bot_grid = [entry["bot_prob_grid"] for entry in data_log],
        rat_grid = [entry["rat_prob_grid"] for entry in data_log],
        time_step = [entry["t"] for entry in data_log]
    )
    print(f"Simulation data saved to {filename}")

def main_improved(grid, n, bot_pos, rat_pos, alpha, simulation_num, seed_value, driver_comparison, data_log):
    frames_grid = []
    grid_for_map = np.copy(grid)
    frames_grid.append(np.copy(grid_for_map))
    grid_for_prob = np.zeros_like(grid, dtype=float)
    init_kb = list_possible_cells(grid_for_map, n)
    prob_grid = init_prob_cells(grid_for_prob, n, init_kb)
    quadrants = partition_grid(grid_for_map, n)
    t=0
    current_quadrant = None
    bot_prob_grid = data_log[-1]['bot_prob_grid']
    sensing = 0
    moving = 0
    waiting = 0
    while True:
        hear_prob_rat = prob_ping_rat(bot_pos, rat_pos, alpha)
        print(hear_prob_rat)
        sensing+=1
        quadrant_probabilities, prob_grid = update_probabilities(prob_grid, quadrants, alpha, hear_prob_rat, bot_pos)
        
        target_quadrant = max(quadrant_probabilities, key=quadrant_probabilities.get)
        print(f"The target quadrant is:\n {target_quadrant}")
        target_cell = weighted_center(target_quadrant, quadrants, prob_grid, grid_for_map, n)
        print(f"Weighted center of {target_quadrant}: {target_cell}")

        data_log.append({
            "t": t,
            "bot_prob_grid": bot_prob_grid,
            "rat_prob_grid": np.copy(prob_grid)
        })

        if target_cell and 0 <= target_cell[0] < n and 0 <= target_cell[1] < n and grid_for_map[target_cell[0]][target_cell[1]] != -1:
            if target_quadrant==current_quadrant:
                print("Consistency")
                if bot_pos==target_cell:
                    # t+=1
                    bot_pos, frames_grid, t, moving = last_ditch_check_neighbours(bot_pos, grid_for_map, n, frames_grid, t, moving)
                    if grid_for_map[bot_pos[0]][bot_pos[1]]==2:
                        print("The rat was caught!!")
                        print(f"Steps taken: {t}")
                        log_simulation_result(simulation_num, seed_value, alpha, "Success")
                        frames_grid.append(np.copy(grid_for_map))
                        if not driver_comparison:
                            visualize_simulation_1(frames_grid)
                        return True
                    print("The bot is already at the target cell")
                    if t>2000:
                        print("timeout")
                        log_simulation_result(simulation_num, seed_value, alpha, "Failure")
                        if not driver_comparison:
                            visualize_simulation_1(frames_grid)
                        return False
                    else:
                        continue
                bot_pos, frames_grid, t, moving = movement(grid_for_map, target_cell, bot_pos, n, frames_grid, rat_pos, t, moving)
                refined_quadrants = refine_quadrants(target_quadrant, quadrants[target_quadrant], n)
                if refined_quadrants:
                    del quadrants[target_quadrant]
                    quadrants.update(refined_quadrants)
                    print(f"Refined {target_quadrant} into {list(refined_quadrants.keys())}")
                else:
                    print(f"No sub-quadrants to refine in {target_quadrant}")
            else:
                print("Else ran")
                if bot_pos==target_cell:
                    # t+=1
                    bot_pos, frames_grid, t, moving = last_ditch_check_neighbours(bot_pos, grid_for_map, n, frames_grid, t, moving)
                    if grid_for_map[bot_pos[0]][bot_pos[1]]==2:
                        print("The rat was caught!!")
                        print(f"Steps taken: {t}")
                        log_simulation_result(simulation_num, seed_value, alpha, "Success")
                        frames_grid.append(np.copy(grid_for_map))
                        if not driver_comparison:
                            visualize_simulation_1(frames_grid)
                        return True
                    print("The bot is already at the target cell")
                    if t>2000:
                        print("timeout")
                        log_simulation_result(simulation_num, seed_value, alpha, "Failure")
                        if not driver_comparison:
                            visualize_simulation_1(frames_grid)
                        return False
                    else:
                        continue
                bot_pos, frames_grid, t, moving = movement(grid_for_map, target_cell, bot_pos, n, frames_grid, rat_pos, t, moving)
                if bot_pos==False:
                    break
                current_quadrant = target_quadrant
        else:
            print(f"Invalid target cell selected: {target_cell}")
            t+=1
            waiting+=1

        if bot_pos==rat_pos:
            print("Bot has caught the rat!")
            print(f"Steps taken: {t}")
            print(f"The moving steps were: {moving}")
            print(f"The sensing steps were: {sensing}")
            print(f"The waiting steps were: {waiting}")
            log_simulation_result(simulation_num, seed_value, alpha, "Success")
            frames_grid.append(np.copy(grid_for_map))
            if not driver_comparison:
                visualize_simulation_1(frames_grid)
            total_steps = len(data_log)
            for i, entry in enumerate(data_log):
                entry["remaining_steps"] = total_steps - i - 1

            save_simulation_data(simulation_num, seed_value, data_log)
            return data_log
        
        if t>2000:
            print("timeout")
            log_simulation_result(simulation_num, seed_value, alpha, "Failure")
            if not driver_comparison:
                visualize_simulation_1(frames_grid)
            return False
        
        #Removed frames grid from return to get only True or false