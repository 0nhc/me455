import numpy as np
import matplotlib.pyplot as plt

from env import Env

def main():
    """
    Question 1
    """
    # Create an instance of the Env class
    env = Env(start_x=0.0, start_y=0.0, end_x=1.0, end_y=1.0, dl=0.001)
    # Set the source origin
    env.set_source_origin(sx=0.3, sy=0.4)
    # Generate 100 random points within the boundaries to measure the Z value
    random_points = np.random.rand(100, 2)
    random_points[:, 0] = random_points[:, 0] * (env.end_x - env.start_x) + env.start_x
    random_points[:, 1] = random_points[:, 1] * (env.end_y - env.start_y) + env.start_y
    # Get observations for each random point
    for point in random_points:
        x, y = point
        env.register_observation(x, y)
    # Visualize the observation map
    env.visualize_map(show_observation_map=True)

    """
    Question 2
    """
    

if __name__ == "__main__":
    main()