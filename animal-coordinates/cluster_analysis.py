import numpy as np
import matplotlib.pyplot as plt
import json
import math
import random

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

    def plot_points(self, x, y):
        # Plot the spatial distribution of points
        plt.scatter(x, y)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Spatial Distribution of Points')
        plt.show()

    def attribute_points(self, x_centroids, y_centroids):
        # Chemin vers le fichier JSON contenant les données des annotations
        json_file_path = '/home/edr/Desktop/Animal Herding Database/Goats-Geladas/kenyan-ungulates/ungulate-annotations/annotations-clean-name-pruned/annotations-clean-name-pruned.json'
        FOV = 84
        ALTITUDE = 80
        N = len(x_centroids)

        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        selected_images = random.sample(data['images'], N)
        centroid_number = 0
        animal_number = 0
        
        # Stick to the Ol Pejeta Animal Density
        Number_of_Animals_OP = 10968
        Area_OP = 360 # 360 km^2
        Density_OP = Number_of_Animals_OP / Area_OP
        
        print(f"The Animal Density in OP is", {Density_OP})
        
        for image_data in selected_images:
            image_id = image_data['id']
            resolution = (image_data['width'], image_data['height'])
            image_file_name = image_data['file_name']
            print(f"Processing image {image_file_name}...")
            image_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]

            x_coords, y_coords = [], []
        
            for annotation in image_annotations:
                bbox = annotation['bbox']            
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                x_center, y_center = self.pixel_to_meter(x_center, y_center, resolution, FOV, ALTITUDE)
                x_coords.append(x_center)
                y_coords.append(y_center)
                
            # Perform a random rotation on the data to make it more random
            rotation_angle = random.uniform(0, 2*math.pi)
            rotation_matrix = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle)],
                                        [math.sin(rotation_angle), math.cos(rotation_angle)]])

            rotated_coords = np.dot(rotation_matrix, np.array([x_coords, y_coords]))
            x_coords = rotated_coords[0]
            y_coords = rotated_coords[1]
            
            point_list = list(zip(x_coords, y_coords))
            centroid = np.mean(point_list, axis=0)
            
            for k in range(len(x_coords)):
                x_coords[k] = x_coords[k] + x_centroids[centroid_number] - centroid[0]
                y_coords[k] = y_coords[k] + y_centroids[centroid_number] - centroid[1]
                
            self.all_x_coords.extend(x_coords)
            self.all_y_coords.extend(y_coords)
            
            plt.scatter(x_coords, y_coords, s=5)
            animal_number += len(x_coords)
            centroid_number += 1
        
        print("Total number of animals:", animal_number)    
        
        plt.xlim(0, 1000)  # Set the X axis range from 0 to 1000
        plt.ylim(0, 1000)  # Set the Y axis range from 0 to 1000
        plt.gca().set_aspect('equal', adjustable='box')  # Impose the same scale on X and Y axis
        plt.grid()
        plt.show()
                
    def main(self):
        N = 10
        x_centroids, y_centroids = self.generate_points(N)
        self.attribute_points(x_centroids, y_centroids)

    def get_all_points(self):
        return self.all_x_coords, self.all_y_coords

if __name__ == "__main__":
    
    animal_attr_points = AnimalPosition()
    animal_attr_points.main()
    all_x_coords, all_y_coords = animal_attr_points.get_all_points()