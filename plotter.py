import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
radius = 1.0  # Radius of the sphere
num_points = 1000  # Number of points to generate

# Generate random angles
phi = np.random.uniform(0, 2 * np.pi, num_points)
theta = np.arccos(np.random.uniform(-1, 1, num_points))

# Calculate 3D coordinates
x = radius * np.sin(theta) * np.cos(phi)
y = radius * np.sin(theta) * np.sin(phi)
z = radius * np.cos(theta)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere points
ax.scatter(x, y, z, s=1)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set axis limits
ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([-radius, radius])

# Show the plot
plt.show()
