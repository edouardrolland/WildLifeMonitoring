import numpy as np
import json
import math
import random
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pyMetaheuristic.algorithm import elephant_herding_optimization

class Drone:
    def __init__(self, altitude, fov, position):
        # Non tunable parameters
        MAX_ALTITUDE = 150
        # Tunable parameters
        self.max_altitude = MAX_ALTITUDE
        self.altitude = altitude
        self.fov = fov
        self.x, self.y = position
        self.radius = self.altitude * math.tan(math.radians(self.fov / 2))

    def plot(self):
        plt.scatter(self.x, self.y, marker='x', s=20, c='r')
        circle = plt.Circle((self.x, self.y), self.radius,
                            color='r', fill=False)
        plt.gca().add_patch(circle)

    def find_visible_animals(self, animal_positions):
        
        drone_position = np.array([[self.x], [self.y]])
        animal_positions = np.array(animal_positions)
        distances = np.linalg.norm(animal_positions - drone_position, axis=0)
        visible_animals_mask = distances < self.radius
        visible_animals = animal_positions[:, visible_animals_mask]
        
        return visible_animals.tolist()

class AnimalPosition:
    
    def __init__(self):
        self.all_x_coords = []
        self.all_y_coords = []

    def pixel_to_meter(self, x, y, resolution, fov, altitude):
        """
        Convert pixel coordinates to meters based on image resolution, camera FOV, and drone altitude.
        """
        width = resolution[0]
        x_meters = 2 * altitude * math.tan(math.radians(fov / 2)) * x / width
        y_meters = 2 * altitude * math.tan(math.radians(fov / 2)) * y / width
        return x_meters, y_meters

    def generate_points(self, N):
        # Generate random x and y coordinates between 0 and 1000
        buffer = 60
        x = np.random.randint(buffer, 1000 - buffer, N)
        y = np.random.randint(buffer, 1000 - buffer, N)
        return x, y

    def plot_points(self):
        # Plot the spatial distribution of points
        plt.scatter(self.all_x_coords, self.all_y_coords, s=5)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Spatial Distribution of Animal Positions')
        plt.xlim(0, 1000)  # Set the X axis range from 0 to 1000
        plt.ylim(0, 1000)  # Set the Y axis range from 0 to 1000
        # Impose the same scale on X and Y axis
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.plot([0, 1000, 1000, 0, 0], [0, 0, 1000, 1000, 0], 'k-')
        plt.show()

    def OlPejeta_Animal_Density(self):
        # Stick to the Ol Pejeta Animal Density
        Number_of_Animals_OP = 10968
        Area_OP = 360
        print(f"The Animal Density in OP is", {Number_of_Animals_OP / Area_OP})

    def attribute_points(self, x_centroids, y_centroids):
        # Chemin vers le fichier JSON contenant les données des annotations
        json_file_path = '/home/edr/Desktop/Animal Herding Database/Goats-Geladas/kenyan-ungulates/ungulate-annotations/annotations-clean-name-pruned/annotations-clean-name-pruned.json'
        FOV = 84
        ALTITUDE = 30
        N = len(x_centroids)

        with open(json_file_path, 'r') as file:
            data = json.load(file)

        selected_images = random.sample(data['images'], N)
        centroid_number = 0
        animal_number = 0

        for image_data in selected_images:
            image_id = image_data['id']
            resolution = image_data['width'], image_data['height']
            image_file_name = image_data['file_name']
            print(f"Processing image {image_file_name}...")
            image_annotations = [
                annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]

            x_coords, y_coords = [], []

            for annotation in image_annotations:
                bbox = annotation['bbox']
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                x_center, y_center = self.pixel_to_meter(
                    x_center, y_center, resolution, FOV, ALTITUDE)
                x_coords.append(x_center)
                y_coords.append(y_center)

            # Perform a random rotation on the data to make it more random
            rotation_angle = random.uniform(0, 2 * math.pi)
            rotation_matrix = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle)],
                                        [math.sin(rotation_angle), math.cos(rotation_angle)]])

            rotated_coords = np.dot(
                rotation_matrix, np.array([x_coords, y_coords]))
            x_coords = rotated_coords[0]
            y_coords = rotated_coords[1]

            point_list = list(zip(x_coords, y_coords))
            centroid = np.mean(point_list, axis=0)

            for k in range(len(x_coords)):
                x_coords[k] = x_coords[k] + \
                    x_centroids[centroid_number] - centroid[0]
                y_coords[k] = y_coords[k] + \
                    y_centroids[centroid_number] - centroid[1]

            self.all_x_coords.extend(x_coords)
            self.all_y_coords.extend(y_coords)

            animal_number += len(x_coords)
            centroid_number += 1

    def generate_herd(self):
        N = 50
        x_centroids, y_centroids = self.generate_points(N)
        self.attribute_points(x_centroids, y_centroids)

    def get_all_points(self):
        return self.all_x_coords, self.all_y_coords
    
    
    
#variable_values = (x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,.........,xn,yn)

animal_attr_points = AnimalPosition()
animal_position = animal_attr_points.get_all_points()

def function_to_optimize(variables_values):
    
    monitored_animals = []
    drones = []
    for k in range(0, len(variables_values)-1):
        drones.append(Drone(ALTITUDE, FOV, (variables_values[k], variables_values[k+1])))    
    
    for drone in drones:
        visible_animals = drone.find_visible_animals(animal_position)
        monitored_animals.extend(visible_animals[0])
            
    unique_monitored_animals = np.unique(monitored_animals, axis=0)
    num_monitored_animals = len(unique_monitored_animals)
    
    percentage = (num_monitored_animals / len(variables_values)) * 100
    
    print("The percentage is: ", percentage)
    
    if percentage < 1:
        return 1000
    
    return 1/percentage

if __name__ == "__main__":

    FOV = 48.8
    ALTITUDE = 100
    N = 3
    
    min_values = [0 for _ in range(2*N)]
    max_values = [1000 for _ in range(2*N)]
 
    # EHO - Parameters
    parameters = {
    'size': 50,
    'min_values': min_values,
    'max_values': max_values,
    'generations': 1000,
    'alpha': 0.5,
    'beta': 0.1,
        'verbose': True,
        'start_init': None,
        'target_value': None
    }
    
    animal_attr_points = AnimalPosition()
    animal_attr_points.generate_herd()
    
    # EHO - Algorithm
    eho = elephant_herding_optimization(target_function = function_to_optimize, **parameters)
    
    # EHO - Solution
    variables = eho[:-1]
    minimum   = eho[ -1]
    print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4))
