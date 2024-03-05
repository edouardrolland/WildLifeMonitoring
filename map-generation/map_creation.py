import numpy as np
import matplotlib.pyplot as plt

# Dictionnary of the number of animals present in Ol Pejeta Conservancy
# Source: https://www.olpejetaconservancy.org/aerial-count-reveals-8-increase-in-ol-pejetas-wildlife/

animals_ol_pejeta = {
    "Plains Zebra": 3042,
    "Impala": 2495,
    "Buffalo": 2800,
    "Gazelle": 1670,
    "Elephant": 269,
    "Eland": 239,
    "Warthdog": 198,
    "Giraffe": 138,
    "Waterbuck": 79,
    "Reedbuck": 29,
    "Ostrich": 9,
}

# Calculate the total number of animals
TOTAL_ANIMALS = sum(animals_ol_pejeta.values())
print("Total number of animals:", TOTAL_ANIMALS)

def generate_similar_points(mean, std_deviation, covariance_matrix, num_points):
    """
    Generate similar points based on mean, standard deviation, and covariance matrix.
    
    Parameters:
        mean: 1D array_like, mean of the distribution for each dimension.
        std_deviation: 1D array_like, standard deviation of the distribution for each dimension.
        covariance_matrix: 2D array_like, covariance matrix.
        num_points: int, number of points to generate.
        
    Returns:
        points: 2D array, generated points with shape (num_points, len(mean)).
    """
    # Generate random samples for each dimension (x, y)
    samples = np.random.randn(num_points, len(mean))
    
    # Use Cholesky decomposition to get a transformation matrix
    # that respects the desired covariance
    cholesky_matrix = np.linalg.cholesky(covariance_matrix)
    
    # Apply the transformation to get points with the desired covariance
    transformed_samples = np.dot(samples, cholesky_matrix.T) + mean
    
    return transformed_samples

# Données de nuage de points (remplacez ces données par les vôtres)
data = np.random.randn(100, 2)  # 100 points 2D

# Calculer la moyenne
mean = np.mean(data, axis=0)

# Calculer la covariance
covariance_matrix = np.cov(data, rowvar=False)

# Générer des points similaires
num_points = 100  # Nombre de points à générer
new_points = generate_similar_points(mean, np.std(data, axis=0), covariance_matrix, num_points)

# Afficher les points originaux et les points générés
plt.figure(figsize=(10, 5))

plt.scatter(data[:, 0], data[:, 1], color='blue', label='Original Points')
plt.scatter(new_points[:, 0], new_points[:, 1], color='red', marker='x', label='Generated Points')

plt.title('Original Points vs Generated Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.show()


