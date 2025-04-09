import numpy as np
import matplotlib.pyplot as plt

NEGATIVE = 0
POSITIVE = 1
UNKNOWN = 2

class Env:
    def __init__(self,
                 start_x: float = 0.0,
                 start_y: float = 0.0,
                 end_x: float = 1.0,
                 end_y: float = 1.0,
                 dl: float = 0.001):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.dl = dl

        # The env contains a 2D grid of points
        self._x = np.arange(start_x, end_x, dl)
        self._y = np.arange(start_y, end_y, dl)
        self._X, self._Y = np.meshgrid(self._x, self._y)
        self._Z = np.zeros(self._X.shape).astype(np.float32)

        # Default source origin is the center of the map
        self._sx = (start_x + end_x) / 2
        self._sy = (start_y + end_y) / 2
        self.set_source_origin(self._sx, self._sy)

        # Initialize observation map with UNKNOWN values
        self._observation_map = np.full(self._X.shape, UNKNOWN).astype(np.int32)
    
    def set_source_origin(self,
                          sx: float,
                          sy: float):
        # For each position in the map, the z value is exp(-100*(np.sqrt((x-sx)^2 + (y-sy)^2)-0.2)^2)
        # This creates a Gaussian-like distribution centered at (sx, sy)
        self._sx = sx
        self._sy = sy
        self._Z = np.exp(-100 * (np.sqrt((self._X - self._sx) ** 2 + (self._Y - self._sy) ** 2) - 0.2) ** 2)

    def register_observation(self,
                        x: float,
                        y: float):
        # Randomly sample a value between 0 and 1
        value = np.random.rand()
        # Compare the value with the Z value at the given coordinates
        # If the value is less than the Z value, return POSITIVE
        if value < self._Z[int((x - self.start_x) / self.dl), int((y - self.start_y) / self.dl)]:            
            self._observation_map[int((x - self.start_x) / self.dl), int((y - self.start_y) / self.dl)] = POSITIVE
            return POSITIVE
        # If the value is greater than the Z value, return NEGATIVE
        elif value > self._Z[int((x - self.start_x) / self.dl), int((y - self.start_y) / self.dl)]:
            self._observation_map[int((x - self.start_x) / self.dl), int((y - self.start_y) / self.dl)] = NEGATIVE
            return NEGATIVE
    
    def clear_observation_map(self):
        # Clear the observation map
        self._observation_map = np.full(self._X.shape, UNKNOWN).astype(np.int32)

    def visualize_map(self,
                      show_observation_map: bool = False):
        # Visualize the map
        plt.cla()
        plt.imshow(self._Z,
                   extent=(self.start_x, self.end_x, self.start_y, self.end_y),
                   origin='lower',
                   cmap='gray',
                   interpolation='nearest')
        if show_observation_map:
            # The source origin is marked with a blue cross
            plt.scatter(self._sx, self._sy, c='blue', s=100, marker='x', linewidths=5, label='Source')  # Darker blue color
            # Gree dots for POSITIVE, Red dots for NEGATIVE
            plt.scatter(self._X[self._observation_map == POSITIVE],
                        self._Y[self._observation_map == POSITIVE],
                        c='#32CD32',  # Lighter green color
                        s=10,
                        label='Positive Signal')
            plt.scatter(self._X[self._observation_map == NEGATIVE],
                        self._Y[self._observation_map == NEGATIVE],
                        c='red',                        
                        s=10,
                        label='Negative Signal')
            # Add legends
            plt.legend(loc='upper right')

        plt.colorbar(mappable=plt.cm.ScalarMappable(cmap='gray'), label='Z Value')
        plt.title('Visualization')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
