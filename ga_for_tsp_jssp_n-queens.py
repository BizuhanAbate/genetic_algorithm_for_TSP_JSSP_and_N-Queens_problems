import time
from typing import List, Tuple, Callable, Optional
from random import sample, randint, random
from itertools import permutations
from contextlib import contextmanager

# Typing definitions
Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]

# Timer context
@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed Time: {end - start:.4f} seconds")

# Generate Population
def generate_population(size: int, genome_func: Callable[[], Genome]) -> Population:
    return [genome_func() for _ in range(size)]

# Crossover function (generic)
def crossover(parent_a: Genome, parent_b: Genome) -> Tuple[Genome, Genome]:
    size = len(parent_a)
    start, end = sorted(sample(range(size), 2))
    offspring_a = parent_a[:start] + parent_b[start:end] + parent_a[end:]
    offspring_b = parent_b[:start] + parent_a[start:end] + parent_b[end:]
    return offspring_a, offspring_b

# Mutation function (generic)
def mutation(genome: Genome, probability: float = 0.5) -> Genome:
    for i in range(len(genome)):
        if random() < probability:
            j = randint(0, len(genome) - 1)
            genome[i], genome[j] = genome[j], genome[i]
    return genome

# Sort population by fitness
def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)

# General Evolution Algorithm
def run_evolution(
    population_size: int,
    genome_func: Callable[[], Genome],
    fitness_func: FitnessFunc,
    fitness_limit: int,
    generation_limit: int
) -> Tuple[Genome, int]:
    population = generate_population(population_size, genome_func)

    for generation in range(generation_limit):
        population = sort_population(population, fitness_func)
        if fitness_func(population[0]) >= fitness_limit:
            return population[0], generation

        next_generation = population[:2]  # Elitism
        for _ in range(len(population) // 2 - 1):
            parents = sample(population, 2)
            offspring_a, offspring_b = crossover(parents[0], parents[1])
            next_generation += [mutation(offspring_a), mutation(offspring_b)]

        population = next_generation

    return population[0], generation

# ------------- TSP ---------------
distances = [
    [0, 5, 10, 20],
    [10, 0, 5, 25],
    [13, 25, 0, 30],
    [11, 25, 20, 0]
]

def generate_tsp_genome() -> Genome:
    return sample(range(len(distances)), len(distances))

def tsp_fitness(genome: Genome) -> int:
    total_distance = sum(distances[genome[i]][genome[i + 1]] for i in range(len(genome) - 1))
    total_distance += distances[genome[-1]][genome[0]]
    return -total_distance

def tsp_bruteforce() -> Tuple[Tuple[int], int]:
    best_route = None
    min_distance = float('inf')
    for route in permutations(range(len(distances))):
        distance = sum(distances[route[i]][route[i + 1]] for i in range(len(route) - 1))
        distance += distances[route[-1]][route[0]]
        if distance < min_distance:
            min_distance = distance
            best_route = route
    return best_route, min_distance

def run_tsp():
    print("\nRunning Genetic Algorithm for TSP")
    with timer():
        best_tsp_solution, tsp_generations = run_evolution(
            population_size=10,
            genome_func=generate_tsp_genome,
            fitness_func=tsp_fitness,
            fitness_limit=-80,
            generation_limit=100
        )
    print(f"Best GA TSP solution: {best_tsp_solution} (Distance: {-tsp_fitness(best_tsp_solution)})")
    print(f"Generations: {tsp_generations}")

    print("\nRunning Brute Force for TSP")
    with timer():
        best_tsp_bf_solution, best_tsp_bf_distance = tsp_bruteforce()
        print(f"Best Brute Force TSP solution: {best_tsp_bf_solution} (Distance: {best_tsp_bf_distance})")

# ------------- JSSP ---------------
jobs = [
    [3, 2, 2],  # Job 1 (time on machine 1, 2, 3)
    [2, 1, 4],  # Job 2 (time on machine 1, 2, 3)
    [4, 3, 1],  # Job 3 (time on machine 1, 2, 3)
]
num_machines = 3

def generate_jssp_genome() -> Genome:
    return sample(range(len(jobs) * num_machines), len(jobs) * num_machines)

def jssp_fitness(genome: Genome) -> int:
    return -calculate_makespan(genome)

def calculate_makespan(genome: Genome) -> int:
    machine_times = [0] * num_machines
    job_times = [0] * len(jobs)

    for gene in genome:
        job_id = gene // num_machines
        machine_id = gene % num_machines
        start_time = max(machine_times[machine_id], job_times[job_id])
        end_time = start_time + jobs[job_id][machine_id]
        machine_times[machine_id] = end_time
        job_times[job_id] = end_time

    return max(machine_times)

def jssp_bruteforce() -> Tuple[List[int], int]:
    best_makespan = float('inf')
    best_schedule = None

    for genome in permutations(range(len(jobs) * num_machines)):
        makespan = calculate_makespan(genome)
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = genome

    return best_schedule, best_makespan

def run_jssp():
    print("\nRunning Genetic Algorithm for JSSP")
    with timer():
        best_jssp_solution, jssp_generations = run_evolution(
            population_size=10,
            genome_func=generate_jssp_genome,
            fitness_func=jssp_fitness,
            fitness_limit=-50,
            generation_limit=100
        )
    best_makespan = -jssp_fitness(best_jssp_solution)
    print(f"Best GA JSSP solution: {best_jssp_solution}")
    print(f"Makespan (total time): {best_makespan}")
    print(f"Generations: {jssp_generations}")

    print("\nRunning Brute Force for JSSP")
    with timer():
        best_schedule, best_makespan = jssp_bruteforce()
        print(f"Best JSSP schedule: {best_schedule}")
        print(f"Makespan (total time): {best_makespan}")

# ------------- N-Queens ---------------
def generate_nqueens_genome(n: int) -> Genome:
    return sample(range(n), n)

def nqueens_fitness(genome: Genome) -> int:
    n = len(genome)
    horizontal_collisions = sum([genome.count(queen) - 1 for queen in genome]) // 2
    diagonal_collisions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == abs(genome[i] - genome[j]):
                diagonal_collisions += 1
    total_collisions = horizontal_collisions + diagonal_collisions
    return -(total_collisions)

def brute_force_nqueens(n: int):
    best_solution = None
    best_conflicts = float('inf')
    for solution in permutations(range(n)):
        conflicts = nqueens_fitness(list(solution))
        if conflicts == 0:
            best_solution = list(solution)
            best_conflicts = conflicts
            break
    return best_solution, best_conflicts

def run_nqueens():
    n = 8
    print("\nRunning Genetic Algorithm for N-Queens")
    with timer():
        best_nqueens_solution, nqueens_generations = run_evolution(
            population_size=10,
            genome_func=lambda: generate_nqueens_genome(n),
            fitness_func=nqueens_fitness,
            fitness_limit=0,
            generation_limit=100
        )
    print(f"Best GA N-Queens solution: {best_nqueens_solution} (Conflicts: {-nqueens_fitness(best_nqueens_solution)})")
    print(f"Generations: {nqueens_generations}")

    print("\nRunning Brute Force for N-Queens")
    with timer():
        best_solution, best_conflicts = brute_force_nqueens(n)
    print(f"Best Brute Force N-Queens solution: {best_solution} (Conflicts: {best_conflicts})")

# Main function
def main():
    print("Starting Genetic Algorithm and Brute Force Comparison...\n")
    run_tsp()      # Run TSP
    run_jssp()     # Run JSSP
    run_nqueens()  # Run N-Queens

if __name__ == "__main__":
    main()
