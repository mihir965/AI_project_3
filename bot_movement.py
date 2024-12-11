from env_utils import *

def plan_path_bot2(grid, bot_pos, dest, n):
    closed_list = [[False for _ in range(n)] for _ in range(n)]
    cell_details = [[Cell() for _ in range(n)] for _ in range(n)]
    open_list = []
    i, j = bot_pos
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j
    heapq.heappush(open_list, (0.0, (i, j)))
    found_dest = False
    cell_details, found_dest = bot_planning_bot2(closed_list, cell_details, open_list, bot_pos, dest, grid, found_dest, n)
    print(len(cell_details))
    if found_dest:
        print(f"Yaayyy\nThe destination is {dest}")
        return track_path_bot2(cell_details, bot_pos, dest, n)
    else:
        print("Oops")
        return None
    
def track_path_bot2(cell_details, src, dest, n):
    path = []
    i, j = dest
    visited = set()
    t = 0
    while not (i == src[0] and j == src[1]):
        path.append((i, j))
        if (i, j) in visited:
            print("Detected loop in track_path.")
            break
        visited.add((i, j))
        temp_i = cell_details[i][j].parent_i
        temp_j = cell_details[i][j].parent_j
        if not is_valid(temp_i, temp_j, n):
            print(f"Invalid parent cell: ({temp_i}, {temp_j})")
            break
        i, j = temp_i, temp_j
        t+=1
    path.append((src[0], src[1]))
    path.reverse()
    # print(path)
    # print("Function almost done")
    return path

def bot_planning_bot2(closed_list, cell_details, open_list, src, dest, grid, found_dest, n):
    while len(open_list) > 0:
        p = heapq.heappop(open_list)[1]
        i, j = p
        closed_list[i][j] = True
        if is_destination(i, j, dest):
            print("Destination found")
            found_dest = True
            break
        for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if is_valid(i + x, j + y, n):
                if is_unblocked(grid, i + x, j + y) and not closed_list[i + x][j + y]:
                    g_new = cell_details[i][j].g + 1
                    h_new = calculate_d_value(i + x, j + y, dest)
                    f_new = g_new + h_new
                    if cell_details[i + x][j + y].f > f_new:
                        cell_details[i + x][j + y].f = f_new
                        cell_details[i + x][j + y].g = g_new
                        cell_details[i + x][j + y].h = h_new
                        cell_details[i + x][j + y].parent_i = i
                        cell_details[i + x][j + y].parent_j = j
                        heapq.heappush(open_list, (f_new, (i + x, j + y)))
    return cell_details, found_dest