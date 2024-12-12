import heapq
from collections import defaultdict

"Reads the city data from the input file and constructs the graph."
def import_city_data():
    graph = defaultdict(list)
    with open('city 1.txt', 'r') as file:
        lines = file.readlines()
        start_parsing = False

        for line in lines:
            line = line.strip()
            if line == "LOCATIONS AND DISTANCES:":
                start_parsing = True
                continue

            if start_parsing and line:
                location1, location2, distance = line.split()
                distance = int(distance)
                graph[location1].append((location2, distance))
                graph[location2].append((location1, distance))
    return graph

"Calculates the shortest path using Dijkstra's algorithm."
def dijkstra(graph, destination):
    pq = [(0, destination)]  # Priority queue (distance, city)
    distances = {destination: 0} # Dictionary to store the shortest distance to each city
    previous_nodes = {destination: None} # Dictionary to track the path

    while pq:
        current_distance, current_city = heapq.heappop(pq) # Get city with the smallest distance

        # Explore neighbors of the current city
        for neighbor, weight in graph[current_city]:
            distance = current_distance + weight # Calculate distance to neighbor

            # If a shorter path is found
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_city
                heapq.heappush(pq, (distance, neighbor)) # Add the neighbor to the queue

    return distances, previous_nodes

"Reconstructs the path from start to destination using the previous nodes dictionary"
def reconstruct_path(previous_nodes, start):
    path = []
    current = start
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    return path

#-------------------------------------------------------------------------------#

destination = 'F'

# Read the graph from the file
graph = import_city_data()

# Find the shortest distances and paths
distances, previous_nodes = dijkstra(graph, destination)

# Output the results
print(f"Shortest distances to {destination}:")
for city, distance in sorted(distances.items()):
    print(f"{city}: {distance}")

print("\nShortest paths to F:")
for city in sorted(distances.keys()):
    path = reconstruct_path(previous_nodes, city)
    print(f"Path from {city} to {destination}: {' -> '.join(path)}")


