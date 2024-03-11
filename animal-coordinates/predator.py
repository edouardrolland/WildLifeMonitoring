import numpy as np
import pygame

class Predator():

    def __init__(self, window):

        self.x = 500
        self.y = 50
        self.vx = 0
        self.vy = 2
        self.visual_predation = 100
        self.direction = np.arctan2(self.vy, self.vx)
        self.predation_detected = False
        self.centroid = [self.x, self.y]
        self.eating = False
        self.eating_duration = 0


    def tracking_behaviour(self, kdtree, preys):
        
        speed_norm = np.sqrt(self.vx**2 + self.vy**2)
        visual_indices = kdtree.query_ball_point((self.x, self.y), self.visual_predation)
        self.centroid = [self.x, self.y]
        
        if len(visual_indices) == 0 and self.predation_detected == True :
            # Any prey can't be seen, so the predator is randomly moving
            self.vx = self.vx + np.random.uniform(-0.1, 0.1)
            self.vy = self.vy + np.random.uniform(-0.1, 0.1)

        if len(visual_indices) != 0:
            self.predation_detected = True #If a prey is detected, the flag is set to True 
            # Find the closest prey
            closest_prey_index = min(visual_indices, key=lambda i: np.linalg.norm(np.array([preys[i].x, preys[i].y]) - np.array([self.x, self.y])))
            closest_prey = preys[closest_prey_index]
            
            # Compute the closest prey's direction
            direction = np.arctan2(closest_prey.y - self.y, closest_prey.x - self.x)

            # Update the predator's speed
            self.vx = speed_norm * np.cos(direction)
            self.vy = speed_norm * np.sin(direction)

            self.direction = direction
            self.closest_prey = closest_prey

            self.centroid = [closest_prey.x, closest_prey.y]
            #If the prey is closed to the predator, the prey is eaten
            if np.linalg.norm(np.array([self.x, self.y]) - np.array([closest_prey.x, closest_prey.y])) < 2:
                preys.remove(closest_prey)
                self.eating = True
            
                    
    def potential_repulsion(self, window, turning_factor):
        
        if self.x != 0 and self.x != window[0]:
            self.ax = turning_factor*(1/(self.x**2) - 1/((self.x - window[0])**2))
        if self.y != 0 and self.y != window[1]:
            self.ay = turning_factor*(1/(self.y**2) - 1/((self.y - window[1])**2))
        if self.x < 0 or self.x > window[0]:
            self.vx = -self.vx
        if self.y < 0 or self.y > window[1]:
            self.vy = -self.vy


    def draw_triangle(self):
        center = (self.x, self.y)
        side_length = 8
        angle_radians = np.arctan2(self.vy, self.vx) + np.pi/2
        triangle = np.array([
            [-side_length / 2, side_length / 2],
            [side_length / 2, side_length / 2],
            [0, -side_length / 1]])
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]])
        
        rotated_triangle = np.dot(triangle, rotation_matrix.T) + center

        return [(int(point[0]), int(point[1])) for point in rotated_triangle]
    
    def speed_limit(self):
        v_max = 1.68
        v_min = 1.68
        vel_norm = np.sqrt(self.vx**2 + self.vy**2)        
        
        if vel_norm > v_max:
            self.vx = (self.vx/vel_norm)*v_max
            self.vy = (self.vy/vel_norm)*v_max
            
        if vel_norm < v_min:
            self.vx = (self.vx/vel_norm)*v_min
            self.vy = (self.vy/vel_norm)*v_min

        if self.eating:
            self.vx, self.vy = 0,0
        
    def uptate(self, window, turnfactor, kdtree, boids):

        self.tracking_behaviour(kdtree, boids)
        self.potential_repulsion(window, turnfactor)
        
        self.vx += self.ax
        self.vy += self.ay
        self.x += self.vx
        self.y += self.vy

        self.speed_limit()

        if self.eating:
            if self.eating_duration < 1000:
                self.eating_duration += 1
                self.eating = True
                self.vx = (self.vx)*0.0
                self.vy = (self.vy)*0.0
            else:
                self.eating = False
                self.eating_duration = 0


if __name__ == "__main__":
    predator = Predator((1000,1000))