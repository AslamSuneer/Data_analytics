import math
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def k_means_clustering(data, k, max_iterations=100, outlier_threshold=2.0):
    centroids = data[:k]

    for iteration in range(max_iterations):
        labels = []
        distances = []

        for point in data:
            # Find the closest centroid
            min_dist, label = min((euclidean_distance(point, centroids[i]), i) for i in range(k))
            labels.append(label)
            distances.append(min_dist)

        new_centroids = []
        outliers = []

        for i in range(k):
            cluster_points = [data[j] for j in range(len(data)) if labels[j] == i and distances[j] <= outlier_threshold]
            
            if len(cluster_points) > 0:
                # Update the centroid as the mean of non-outlier points in the cluster
                new_centroid = [sum(point[dim] for point in cluster_points) / len(cluster_points) for dim in range(len(data[0]))]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[i])

            # Collect points considered outliers based on the threshold distance
            outliers.extend([data[j] for j in range(len(data)) if labels[j] == i and distances[j] > outlier_threshold])

        print(f"\nIteration {iteration + 1} clusters (excluding outliers):")
        for i in range(k):
            cluster_points = [data[j] for j in range(len(data)) if labels[j] == i and distances[j] <= outlier_threshold]
            print(f"Cluster {i + 1}: {cluster_points}")

        if new_centroids == centroids:
            break

        centroids = new_centroids

    return labels, centroids, outliers

def plot_clusters(data, labels, centroids, outliers):
    plt.figure(figsize=(8, 6))

    for i in range(max(labels) + 1):
        cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
        cluster_points = list(zip(*cluster_points))  
        plt.scatter(cluster_points[0], cluster_points[1], label=f'Cluster {i + 1}')

    # Plot outliers
    if outliers:
        outliers_x, outliers_y = zip(*outliers)
        plt.scatter(outliers_x, outliers_y, color='red', marker='x', s=50, label='Outliers')

    # Plot centroids
    centroids_x, centroids_y = zip(*centroids)
    plt.scatter(centroids_x, centroids_y, color='black', marker='*', s=100, label='Centroids')

    plt.title('K-means Clustering with Outliers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.show()

csv_file = "kmean.csv"  
df = pd.read_csv(csv_file)
data = df.values.tolist()

k = int(input("Enter the number of clusters (k): "))
outlier_threshold = float(input("Enter the outlier threshold distance: "))

labels, centroids, outliers = k_means_clustering(data, k, outlier_threshold=outlier_threshold)

print("\nFinal clusters (excluding outliers):")
for i in range(k):
    cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
    print(f"Cluster {i + 1}: {cluster_points}")

print("\nFinal centroids:")
print(centroids)

print("\nIdentified outliers:")
print(outliers)

plot_clusters(data, labels, centroids, outliers)
