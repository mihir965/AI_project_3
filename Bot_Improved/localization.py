import env_utils
import numpy as np
import random

def list_open_cells(grid, n):
    open_list = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0 or grid[i][j] == 3:
                open_list.append((i,j))
    return open_list

def list_rat_poss_cells(grid, n):
    rat_list = []
    for i in range(n):
        for j in range(n):
            if grid[i][j]==0:
                rat_list.append((i,j))
    return rat_list

def init_prob_cells(grid, n, list_poss_cells):
    new_grid = np.zeros((n,n), dtype=float)
    num_possible_cells = len(list_poss_cells)
    init_value = 1.0 / num_possible_cells
    for cell in list_poss_cells:
        new_grid[cell[0], cell[1]] = init_value
    return new_grid

def sensing_neighbours_blocked(grid, bot_pos, n):
    cardinality = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    blocked_cells = 0
    for ci, cj in cardinality:
        test_i, test_j = bot_pos[0]+ci, bot_pos[1]+cj
        if 0 <= test_i < n and 0 <= test_j < n:
            if grid[test_i][test_j]==-1:
                blocked_cells+=1
        else:
            blocked_cells+=1
    return blocked_cells

def update_kb_blocked(bot_kb, blocked, grid, n):
    new_bot_kb = []
    for i in bot_kb:
        blocked_cells = sensing_neighbours_blocked(grid, (i[0],i[1]), n)
        if blocked_cells==blocked:
            new_bot_kb.append(i)
    return new_bot_kb

def check_common_direction(bot_kb, grid, last_move_direction, n):
    directions = {
        'north': 0,
        'south': 0,
        'east': 0,
        'west': 0
    }
    direction_offset = {
        'north':(-1,0),
        'south':(1,0),
        'east':(0,1),
        'west':(0,-1)      
    }
    opposite_direction = {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east'
    }
    for i in bot_kb:
        for dir_name, (ci, cj) in direction_offset.items():
            test_i, test_j = i[0]+ci , i[1]+cj
            if 0 <= test_i < n and 0 <= test_j < n:
                if grid[test_i][test_j]==0:
                    directions[dir_name]+=1
    if last_move_direction:
        forbidden_dir = opposite_direction[last_move_direction]
        if len([d for d in directions if directions[d]>0])>1:
            directions[forbidden_dir] = -1
    return max(directions, key=directions.get)

def attempt_movement(dir_check, grid, bot_pos, n):
    direction_offset = {
        'north':(-1,0),
        'south':(1,0),
        'east':(0,1),
        'west':(0,-1)      
    }
    move = direction_offset[dir_check]
    test_i, test_j = bot_pos[0]+move[0], bot_pos[1]+move[1]
    if not (0 <= test_i < n and 0 <= test_j < n):
        return False, bot_pos
    if grid[test_i][test_j] == 1:
        return False, bot_pos
    else:
        grid[bot_pos[0]][bot_pos[1]] = 0
        bot_pos = (test_i, test_j)
        if grid[test_i][test_j] == 2:
            print("The bot caught the rat!")
        grid[test_i][test_j] = 3
        return True, bot_pos

def update_kb_movement(move_check, dir_check, bot_kb, grid, n):
    updated_kb_moves = []
    direction_offset = {
        'north':(-1,0),
        'south':(1,0),
        'east':(0,1),
        'west':(0,-1)      
    }
    direction = direction_offset[dir_check]
    for i in bot_kb:
        test_i, test_j = i[0] + direction[0], i[1] + direction[1]
        if 0 <= test_i < n and 0 <= test_j < n:
            movement_possible = (grid[test_i][test_j] != 1)
        else:
            movement_possible = False
        if move_check:
            if movement_possible:
                updated_kb_moves.append((test_i, test_j))
        else:
            if not movement_possible:
                updated_kb_moves.append(i)
    return updated_kb_moves

def compute_weighted_center(prob_grid):
    rows, cols = prob_grid.shape
    total_prob = np.sum(prob_grid)
    if total_prob == 0:  # Avoid division by zero
        return rows // 2, cols // 2  # Default to grid center

    # Compute weighted average positions
    weighted_row = sum(i * np.sum(prob_grid[i, :]) for i in range(rows)) / total_prob
    weighted_col = sum(j * np.sum(prob_grid[:, j]) for j in range(cols)) / total_prob
    
    return int(round(weighted_row)), int(round(weighted_col))   


def main_function(grid, n, bot_pos):
    print(f"Original bot position that simulation knows: {bot_pos}")
    
    # Initial sets of possible cells
    init_open_list = list_open_cells(grid, n)
    rat_cells = list_rat_poss_cells(grid, n)

    # Create bot_prob_grid as a full n x n array with probabilities only in open cells
    bot_prob_grid = np.zeros((n,n), dtype=float)
    if len(init_open_list) > 0:
        initial_prob = 1.0 / len(init_open_list)
        for cell in init_open_list:
            bot_prob_grid[cell[0], cell[1]] = initial_prob

    # Create rat_prob_grid similarly
    rat_prob_grid = init_prob_cells(grid, n, rat_cells)

    weighted_center = compute_weighted_center(bot_prob_grid)
    grid_center = (rat_prob_grid.shape[0] // 2, rat_prob_grid.shape[1] // 2)
    dist_to_center = abs(weighted_center[0] - grid_center[0]) + abs(weighted_center[1] - grid_center[1])
    prob_at_center = bot_prob_grid[weighted_center]
    most_probable_cell_prob = np.max(bot_prob_grid)

    data_log = []
    t = 0
    blocked_sensing = 0
    direction_sensing = 0
    last_move_direction = None
    bot_kb = init_open_list
    print(f"The length of the Initial Bot Knowledge base: {len(bot_kb)}")

    blocked_check = True
    while len(bot_kb) > 1 and t < 1000:
        if blocked_check:
            # Perform blocked sensing logic
            blocked = sensing_neighbours_blocked(grid, bot_pos, n) 
            bot_kb = update_kb_blocked(bot_kb, blocked, grid, n)
            
            # Update bot_prob_grid based on the new bot_kb
            bot_prob_grid[:] = 0.0
            if len(bot_kb) > 0:
                new_prob = 1.0 / len(bot_kb)
                for cell in bot_kb:
                    bot_prob_grid[cell[0], cell[1]] = new_prob

            blocked_sensing += 1
            print(f"Number of blocked cells: {blocked}")
            print(f"Length of kb after sensing blocked neighbours: {len(bot_kb)}")
            print("End of blocked check")
            print(f"Knowledge base after block check:\n{bot_kb}")

        else:
            # Perform direction checking and movement logic
            print("Checking in directions")
            dir_check = check_common_direction(bot_kb, grid, last_move_direction, n)
            print(f"The most common dir: {dir_check}")
            move_check, bot_pos = attempt_movement(dir_check, grid, bot_pos, n)
            print(f"Movement in {dir_check} is {move_check}")
            print(f"New pos: {bot_pos}")
            bot_kb = update_kb_movement(move_check, dir_check, bot_kb, grid, n)

            # Update bot_prob_grid based on new bot_kb
            bot_prob_grid[:] = 0.0
            if len(bot_kb) > 0:
                new_prob = 1.0 / len(bot_kb)
                for cell in bot_kb:
                    bot_prob_grid[cell[0], cell[1]] = new_prob

            print(f"New length of kb: {len(bot_kb)}")
            if move_check:
                last_move_direction = dir_check
            print(f"Knowledge base after direction check:\n{bot_kb}")
            direction_sensing += 1

        if len(bot_kb) == 0:
            print("Error: No possible positions remain in the knowledge base.")
            break

        # Switch between blocked and direction checks
        blocked_check = not blocked_check
        print(f"Before next time step: length kb: {len(bot_kb)}")

        # Log the current probability grids
        data_log.append({
            "t": t,
            "bot_prob_grid": bot_prob_grid.copy(),
            "rat_prob_grid": rat_prob_grid.copy(),
            "dist_to_target": dist_to_center,
            "prob_target_cell": prob_at_center,
            "most_probable_cell_prob": most_probable_cell_prob
        })

        t += 1
        print(f"Time step: {t}")

    if t==1000 or t>1000:
        return False, False

    if len(bot_kb) == 1:
        print(f"Remaining KB: {bot_kb[0]}\n Bot Pos: {bot_pos}")

    print(f"Time Steps taken: {t}")
    print(f"Number of Blocked cells sensing actions: {blocked_sensing}")
    print(f"Number of direction sensing actions: {direction_sensing}")
    return bot_kb[0], data_log