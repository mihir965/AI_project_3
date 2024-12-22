import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_data(is_test):
    data = np.load("data/seed_457_67152.npz")

    bot_grid = np.array(data['bot_grid'])
    rat_grid = np.array(data['rat_grid'])
    time_step_remaining = np.array(data['time_step_remaining'])
    blocked_ratio = np.array(data['blocked_ratio'])
    time_step = np.array(data['time_step'])
    dist_to_target = np.array(data['dist_to_target'])
    target_cell_prob = np.array(data['target_cell_prob'])
    max_prob = np.array(data['max_prob'])

    #Flattening the grids
    bot_grid = bot_grid.reshape(bot_grid.shape[0], -1)
    rat_grid = rat_grid.reshape(rat_grid.shape[0], -1)

    # Concatenate all features
    X = np.hstack([
        bot_grid,
        rat_grid,
        blocked_ratio.reshape(-1, 1),
        time_step.reshape(-1, 1),
        dist_to_target.reshape(-1, 1),
        target_cell_prob.reshape(-1, 1),
        max_prob.reshape(-1, 1)
    ])

    if is_test==True:
        y = np.array(data['time_step_remaining'])  
    else:
        # Target output
        y = time_step_remaining


    #Normalizing data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y