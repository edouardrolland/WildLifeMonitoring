import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import math
import cv2
import os
import sys


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

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Print a progress bar to track the progress of a loop.
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\033[F%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()

def pixel_to_meter(x, y, resolution, fov, altitude):
    """
    Convert pixel coordinates to meters based on image resolution, camera FOV, and drone altitude.
    """
    width = resolution[0]
    x_meters = 2 * altitude * math.tan(math.radians(fov / 2)) * x / width
    y_meters = 2 * altitude * math.tan(math.radians(fov / 2)) * y / width
    return x_meters, y_meters

def generate_video(images_folder):
    # Create the video from the saved images
    images = [img for img in os.listdir(images_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(images_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

    for image in images:
        img_path = os.path.join(images_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
        os.remove(img_path)  # Remove the image from memory after writing it to the video

    cv2.destroyAllWindows()
    video.release()
    
def generate_csv_file():
    
    txt_file_path = f"animal_positions_global.csv"

    with open(txt_file_path, 'w') as csv_file:
        csv_file.write('Animal Herd Position Database - Edouard George Alain Rolland - SDU Drone Center\n')
        csv_file.write('categories,[{"id":1,"name":"zebra"},{"id":2,"name":"gazelle"},{"id":3,"name":"wbuck"},{"id":4,"name":"buffalo"},{"id":5,"name":"other"}]\n')
        csv_file.write('image_file_name,[x,y,category]\n')
        
    with open(txt_file_path, 'a') as csv_file:
        csv_file.write(f'{image_file_name},{csv_list}\n')
        

def main(plot_clusters=True):
    
    FOV = 84
    ALTITUDE = 80
    json_file_path = '/home/edr/Desktop/Animal Herding Database/Goats-Geladas/kenyan-ungulates/ungulate-annotations/annotations-clean-name-pruned/annotations-clean-name-pruned.json'

    image_folder = "images_folder"
    os.makedirs(image_folder, exist_ok=True)
    
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
    number_of_clusters, number_of_animals, number_of_animals_per_cluster = [], [], []
    
    
    
    for image_data in data['images']:
        image_id = image_data['id']
        resolution = (image_data['width'], image_data['height'])
        image_file_name = image_data['file_name']
        print(f"Processing image {image_file_name}...")

        image_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]

        # Perform DBSCAN clustering
        x_coords, y_coords = [], []
        csv_list = []

        for annotation in image_annotations:
            bbox = annotation['bbox']
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2
            x_center, y_center = pixel_to_meter(x_center, y_center, resolution, FOV, ALTITUDE)
            x_coords.append(x_center)
            y_coords.append(y_center)
            category = annotation['category_id']
            csv_list.append([x_center, y_center, category])

        eps = 20
        min_samples = 3
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(list(zip(x_coords, y_coords)))

        # Visualize the clusters
        unique_labels = set(labels)
        visible_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
        label_colors = {label: color for label, color in zip(unique_labels, visible_colors)}

        cluster_sizes = [np.sum(labels == k) for k in unique_labels]
        
        # Place this code inside the for loop to print the progress bar
        print_progress_bar(image_id + 1, len(data['images']), prefix='Progress:', suffix='Complete', length=50, fill='█')


        if plot_clusters:
            mean_cluster_size = np.mean(cluster_sizes)
            std_cluster_size = np.std(cluster_sizes)

            plt.xlabel(' X (m) ')
            plt.ylabel(' Y (m) ')
            plt.title('DBSCAN Clustering')

            for k in unique_labels:
                if k == -1:
                    col = 'black'
                else:
                    col = label_colors.get(k, 'gray')
                class_member_mask = (labels == k)
                xy = np.array(list(zip(x_coords, y_coords)))[class_member_mask]
                plt.scatter(xy[:, 0], xy[:, 1], color=col, alpha=0.5)

            plt.axis([0, 2 * ALTITUDE * math.tan(math.radians(FOV / 2)), 0, 2 * ALTITUDE * math.tan(math.radians(FOV / 2)) * resolution[1] / resolution[0]])
            #plt.savefig(f"{image_folder}/{image_file_name[:-4]}.png", dpi=300)
            #plt.clf()


    
if __name__ == "__main__":
    main(False)
