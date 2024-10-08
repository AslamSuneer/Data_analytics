import math
import pandas as pd

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

# K-means clustering function
def k_means_clustering(data, k, max_iterations=100):
    # Step 1: Randomly initialize centroids (in this case, we take the first k data points)
    centroids = data[:k]

    for iteration in range(max_iterations):
        # Step 2: Assign labels to each data point based on the closest centroid
        labels = [min(range(k), key=lambda i: euclidean_distance(point, centroids[i])) for point in data]

        # Step 3: Calculate new centroids based on the assigned labels
        new_centroids = []
        for i in range(k):
            cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
            if len(cluster_points) > 0:
                new_centroid = [sum(point[dim] for point in cluster_points) / len(cluster_points) for dim in range(len(data[0]))]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[i])  # If no points assigned, keep the previous centroid

        print(f"Iteration {iteration + 1} clusters:")
        for i in range(k):
            cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
            print(f"Cluster {i + 1}: {cluster_points}")

        # Step 4: Check for convergence (if centroids do not change)
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return labels, centroids

# Read data from a CSV file
csv_file = "kmean.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Convert data to a list of lists (assuming numerical data)
data = df.values.tolist()

# Number of clusters (k)
k = int(input("Enter the number of clusters (k): "))

# Run the K-means clustering
labels, centroids = k_means_clustering(data, k)

# Output the final clusters and centroids
print("\nFinal clusters:")
for i in range(k):
    cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
    print(f"Cluster {i + 1}: {cluster_points}")

print("\nFinal centroids:")
print(centroids)
