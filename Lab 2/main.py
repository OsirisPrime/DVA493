import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Load the data
def import_file():
    locations = {}
    with open('berlin52.tsp', "r") as file:
        lines = file.readlines()
        node_section = False
        for line in lines:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                node_section = True
                continue
            if line == "EOF" or not node_section:
                continue
            parts = line.split()
            if len(parts) == 3:
                node_id = int(parts[0])
                x, y = map(float, parts[1:])
                locations[node_id] = (x, y)
    return locations

# Calculate the Euclidean distance
def calculate_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# Create the distance matrix
def create_distance_matrix(locations):
    ids = list(locations.keys())
    size = len(ids)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = calculate_distance(locations[ids[i]], locations[ids[j]])
    return matrix

# Fitness function
def fitness(route, distance_matrix):
    total_distance = sum(distance_matrix[route[i-1]][route[i]] for i in range(len(route)))
    total_distance += distance_matrix[route[-1]][route[0]]  # Return to start
    return total_distance

# Initialize the population
def initialize_population(size, num_locations):
    population = []
    for _ in range(size):
        route = list(range(1, num_locations))  # Exclude starting location
        random.shuffle(route)
        route = [0] + route  # Add starting location at the beginning
        population.append(route)
    return population

# Tournament selection for selecting parents
def tournament_selection(population, fitnesses, tournament_size):
    tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
    tournament_winner = min(tournament, key=lambda x: x[1])  # Select based on fitness (lower is better)
    return tournament_winner[0]

# Perform crossover between two parents
def crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    for loc in parent2:
        if loc not in child:
            child[child.index(-1)] = loc
    return child

# Mutate a route
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(route)), 2)  # Avoid mutating start location
        route[i], route[j] = route[j], route[i]

# Genetic algorithm with termination based on distance threshold
def genetic_algorithm(locations, distance_threshold, population_size, elitism_rate, tournament_size, max_generations, mutation_rate):
    distance_matrix = create_distance_matrix(locations)
    num_locations = len(locations)
    population = initialize_population(population_size, num_locations)

    best_route = None
    best_distance = float('inf')
    elitism_count = int(elitism_rate * population_size)
    generation = 0

    while best_distance > distance_threshold and generation < max_generations:
        # Calculate fitness values
        fitness_values = [fitness(route, distance_matrix) for route in population]
        fitnesses = [1 / f for f in fitness_values]  # Inverse, since lower distance is better

        # Sort population by fitness (ascending order of distance)
        sorted_population = [route for _, route in sorted(zip(fitness_values, population))]

        # Elitism: Carry over the best individuals to the next generation
        new_population = sorted_population[:elitism_count]

        # Generate the rest of the population
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        # Trim excess individuals if the population exceeds the size
        population = new_population[:population_size]

        # Update the best solution found so far
        for route in population:
            dist = fitness(route, distance_matrix)
            if dist < best_distance:
                best_distance = dist
                best_route = route

        generation += 1

    return best_route, best_distance, distance_matrix

# Plot the route
def plot_route(locations, route):
    x = [locations[i+1][0] for i in route]  # Adjust index as locations are 1-indexed
    y = [locations[i+1][1] for i in route]
    x.append(x[0])  # Return to start
    y.append(y[0])  # Return to start

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', color='b', linestyle='-')
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(route[i % len(route)] + 1), fontsize=8, color="red")
    plt.title("Optimal TSP Route")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.show()

# Compute the total distance of a route
def compute_total_distance(route, distance_matrix):
    total_distance = sum(distance_matrix[route[i-1]][route[i]] for i in range(len(route)))
    total_distance += distance_matrix[route[-1]][route[0]]  # Return to start
    return total_distance

#----------------------------------------------------------------------------------------------------#

# Import locations
locations = import_file()
print(f"All locations: {locations}")

# Model parameters
distance_threshold = 8000
population_size = 500
elitism_rate = 0.4
mutation_rate = 0.2
tournament_size = 8
max_generations = 1000  # Fallback to avoid infinite loop

# Train the model to find the best route
best_route, best_distance, distance_matrix = genetic_algorithm(
    locations, distance_threshold, population_size, elitism_rate, tournament_size, max_generations, mutation_rate
)

# Compute and print the total distance
total_distance = compute_total_distance(best_route, distance_matrix)
print(f"Best route: {best_route}")
print(f"Best distance: {best_distance}")
print(f"Total distance of the route: {total_distance}")

# Visualize the best route
plot_route(locations, best_route)
