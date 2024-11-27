import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Load the data
def import_file():
    locations = {} # Empty location list

    with open('berlin52.tsp', "r") as file:
        lines = file.readlines()
        node_section = False

        for line in lines:
            line = line.strip()

            if line == "NODE_COORD_SECTION": # Start of coordinates
                node_section = True
                continue

            if line == "EOF" or not node_section: # End of coordinates
                continue
            parts = line.split()

            if len(parts) == 3:
                node_id = int(parts[0]) # Take the first integer as ID
                x, y = map(float, parts[1:]) # Take the rest as x & y coordinates
                locations[node_id] = (x, y) # Insert the x & y coordinate to the id

    return locations

# Calculate the Euclidean distance
def calculate_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# Create the distance matrix
def create_distance_matrix(locations):
    ids = list(locations.keys()) #Extract locations IDs
    size = len(ids)
    matrix = np.zeros((size, size)) # Make an empty matrix of (size x size)

    # Go through all row and columns, and put in the distances between the cities
    for i in range(size):
        for j in range(size):
            matrix[i][j] = calculate_distance(locations[ids[i]], locations[ids[j]])
    return matrix

# Fitness function
def fitness(route, distance_matrix):
    total_distance = sum(distance_matrix[route[i-1]][route[i]] for i in range(len(route))) #Calculate the total distances
    total_distance += distance_matrix[route[-1]][route[0]]  # Return to start
    return total_distance

# Initialize the population
def initialize_population(size, num_locations):
    population = []
    for _ in range(size):
        route = list(range(1, num_locations))  # Exclude starting location
        random.shuffle(route) # Randomize the cities order
        route = [0] + route  # Add starting location at the beginning
        population.append(route)
    return population

# Tournament selection for selecting parents with probabilistic choice
def tournament_selection(population, fitnesses, tournament_size):
    # Randomly select individuals for the tournament
    tournament = random.sample(list(zip(population, fitnesses)), tournament_size)

    # Sort the tournament participants by fitness. Lower is better
    tournament = sorted(tournament, key=lambda x: x[1])

    # Assign probabilities to participants based on rank
    probabilities = [1 / (i + 1) for i in range(tournament_size)]
    probabilities = [p / sum(probabilities) for p in probabilities]  # Normalize to sum to 1

    # Choose a participant probabilistically based on rank
    selected_index = np.random.choice(range(tournament_size), p=probabilities)
    return tournament[selected_index][0]

# Perform crossover between two parents
def crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size # Put negative placeholder
    start, end = sorted(random.sample(range(size), 2)) # Select two random points to copy from parent 1
    child[start:end] = parent1[start:end]

    # Insert the remaining cities from parent 2
    for loc in parent2:
        if loc not in child:
            child[child.index(-1)] = loc
    return child

# Mutate a route
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(route)), 2)  # Avoid mutating start location
        route[i], route[j] = route[j], route[i] # Switch places

# Genetic algorithm with termination based on distance threshold
def genetic_algorithm(locations, distance_threshold, population_size, elitism_rate, tournament_size, max_generations):
    distance_matrix = create_distance_matrix(locations) # Create distance matrix
    num_locations = len(locations)
    population = initialize_population(population_size, num_locations) # Initialize the population

    best_route = None
    best_distance = float('inf')
    elitism_count = int(elitism_rate * population_size)
    generation = 0
    best_distances = []

    # Look for the best route until distance < 8000 or until max generation
    while best_distance > distance_threshold and generation < max_generations:
        # Calculate fitness values
        fitness_values = [fitness(route, distance_matrix) for route in population]

        # Sort population by fitness (ascending order of distance)
        sorted_population = [route for _, route in sorted(zip(fitness_values, population))]

        # Elitism: Carry over the best individuals to the next generation
        new_population = sorted_population[:elitism_count]

        # Dynamic mutation rate. Higher early (diversity), low later (exploitation)
        mutation_rate = 0.5 * (1 - (generation / max_generations))

        # Generate the rest of the population
        while len(new_population) < population_size:
            # Select two parent
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)

            # Perform crossover with both parents
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)

            # Mutate the children
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        # Trim excess individuals if the population exceeds the size
        population = new_population[:population_size]

        # Update the best solution found so far
        for route in population:
            dist = fitness(route, distance_matrix)
            if dist < best_distance: # Found new best distance
                best_distance = dist
                best_route = route

        print(f"Generation {generation}: New best route found with distance {best_distance:.2f}")
        best_distances.append(best_distance)
        generation += 1

    return best_route, best_distance, best_distances


# Plot the route
def plot_route(locations, route, distance):
    x = [locations[i+1][0] for i in route]  # Adjust index as locations are 1-indexed
    y = [locations[i+1][1] for i in route]
    x.append(x[0])  # Return to start
    y.append(y[0])  # Return to start

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', color='blue', linestyle='-')
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(route[i % len(route)] + 1), fontsize=8, color="red")
    plt.title(f"Optimal TSP Route: {distance:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.show()

# Plot distance improvement over generations
def plot_distance_progress(best_distances):
    plt.figure(figsize=(10, 6))
    plt.plot(best_distances, color='blue', label='Best Distance')
    plt.title("Best Distance Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.grid()
    plt.show()

#----------------------------------------------------------------------------------------------------#

# Import locations
locations = import_file()
print(f"All locations: {locations}")

# Model parameters
distance_threshold = 8000
population_size = 2000
elitism_rate = 0.1
tournament_size = 8
max_generations = 10000  # Fallback to avoid infinite loop

# Train the model to find the best route
best_route, best_distance, best_distances = genetic_algorithm(
    locations, distance_threshold, population_size, elitism_rate, tournament_size, max_generations
)

# Compute and print the route and best distance
print()
print(f"Best route: {best_route}")
print(f"Best distance: {best_distance:.2f}")

# Visualize the best route
plot_route(locations, best_route, best_distance)

# Visualize the result
plot_distance_progress(best_distances)
