# Genetic Algorithm Solver

Welcome to the Genetic Algorithm Solver! This project implements a genetic algorithm to solve three classic optimization problems: 
the Traveling Salesman Problem (TSP), Job-Shop Scheduling Problem (JSSP), and N-Queens Problem. 
The solver includes both a genetic algorithm approach and a brute-force method for comparison.

## Features

- Traveling Salesman Problem (TSP): Finds the shortest possible route visiting a list of cities and returning to the origin city.
- Job-Shop Scheduling Problem (JSSP): Optimizes job schedules to minimize total completion time (makespan) across multiple machines.
- N-Queens Problem: Places N queens on a chessboard such that no two queens threaten each other, with the aim of finding all valid configurations.
- Genetic Algorithm Implementation: Efficiently solves each problem using genetic algorithms with crossover, mutation, and selection strategies.
- Brute Force Comparison: Provides brute-force solutions for benchmarking against genetic algorithm results.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required libraries: `itertools`, `random`, `time`

### Installation

1. Clone the repository
2. Navigate to the project directory
3. Run the Python script

After running the script, you will see results for each optimization problem, including:

- Best solutions found by the genetic algorithm
- Comparisons with brute-force solutions
- Execution times for both methods

## Algorithms

### 1. Traveling Salesman Problem (TSP)
- Genome Representation: Random order of cities
- Fitness Function: Negative total distance of the tour
- Crossover: Combination of parent tours
- Mutation: Randomly swap cities in the tour

### 2. Job-Shop Scheduling Problem (JSSP)
- Genome Representation: Random job-machine assignments
- Fitness Function: Negative makespan of job completion
- Crossover: Combination of job-machine assignments
- Mutation: Randomly swap jobs or machines

### 3. N-Queens Problem
- Genome Representation: Random positions of queens on the board
- Fitness Function: Negative number of conflicts
- Crossover: Combination of parent queen positions
- Mutation: Randomly swap queen positions
