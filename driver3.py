from env_utils import *
from bot_movement import *
import sys
import os
import numpy as np
import random
from Bot import *
from Bot_Improved import *
import csv
from collections import defaultdict

n = 30

def initialize_comparison_log():
    with open("comparison_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Simulation Number", "Seed Value", "Alpha", "Baseline Outcome", "Improved Outcome", "Baseline Move Outcome", "Improved Move Outcome"])

def run_single_comparison(alpha, simulation_num):

    seed_value = random.randint(1, 1000)
    print(f"Alpha: {alpha:.2f}, Simulation: {simulation_num}, Seed: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    # Initialize the grid and bot/rat positions
    grid = grid_init(n)
    bot_pos_baseline = bot_init(grid, n, 3)
    bot_pos_improved = bot_pos_baseline
    rat_pos = rat_init(grid, n, 2)
    
    # Run the baseline bot
    grid_baseline = np.copy(grid)
    baseline_caught = main_function_catching(grid_baseline, n, bot_pos_baseline, rat_pos, alpha, simulation_num, seed_value, True)

    #Run the baseline with moving rat
    grid_base_move = np.copy(grid)
    baseline_moving = main_function_catching_moving_rat(grid_base_move, n, bot_pos_baseline, rat_pos, alpha, simulation_num, seed_value, True)


    # Run the improved bot
    grid_improved = np.copy(grid)
    improved_caught = main_improved(grid_improved, n, bot_pos_improved, rat_pos, alpha, simulation_num, seed_value, True)

    #Run the improved bot with moving rat
    grid_improved_move = np.copy(grid)
    improved_moving = main_improved_with_moving_rat(grid_improved_move, n, bot_pos_improved, rat_pos, alpha, simulation_num, seed_value, True)
    
    # Log the results
    with open("comparison_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([simulation_num, seed_value, f"{alpha:.2f}", "Success" if baseline_caught else "Failure", "Success" if improved_caught else "Failure", "Success" if baseline_moving else "Failure", "Success" if improved_moving else "Failure"])
    
    return baseline_caught, improved_caught, baseline_moving, improved_moving

def run_comparisons(alpha_start=0.0, alpha_end=0.2, alpha_step=0.01, simulations=50):

    initialize_comparison_log()
    
    alpha_values = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
    
    # Track results for each bot
    baseline_success = defaultdict(int)
    baseline_move_success = defaultdict(int)
    improved_success = defaultdict(int)
    improved_move_success = defaultdict(int)
    total_simulations = defaultdict(int)
    
    for alpha in alpha_values:
        for sim_num in range(1, simulations + 1):
            # Run a single comparison for both bots
            baseline_caught, improved_caught, baseline_moving, improved_moving = run_single_comparison(alpha, sim_num)
            baseline_success[alpha] += int(baseline_caught)
            baseline_move_success[alpha] += int(baseline_moving)
            improved_success[alpha] += int(improved_caught)
            improved_move_success[alpha] += int(improved_moving)
            total_simulations[alpha] += 1
    
    # Calculate and print average accuracies
    print("\nAverage Accuracies per Alpha Value:")
    with open("comparison_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Alpha", "Baseline Accuracy","Baseline Move Accuracy", "Improved Accuracy", "Improved Move Accuracy"])
        for alpha in sorted(total_simulations.keys()):
            baseline_acc = baseline_success[alpha] / total_simulations[alpha]
            baseline_acc_move = baseline_move_success[alpha] / total_simulations[alpha]
            improved_acc = improved_success[alpha] / total_simulations[alpha]
            improved_acc_move = improved_move_success[alpha] / total_simulations[alpha]
            print(f"Alpha: {alpha:.2f} | Baseline: {baseline_acc:.2%} | Baseline Move: {baseline_acc_move:.2%} | Improved: {improved_acc:.2%} | Improved Move: {improved_acc_move:.2%}")
            writer.writerow([f"{alpha:.2f}", f"{baseline_acc:.2%}", f"{baseline_acc_move:.2%}" , f"{improved_acc:.2%}", f"{improved_acc_move:.2%}"])

if __name__ == "__main__":
    run_comparisons()
