'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from nappe import nappe_clotures, nappe_versements

# ==========================================
# graph pour les taux de clotures
# ==========================================

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.array(nappe_clotures.index)
Y = np.array(nappe_clotures.columns)
Y, X = np.meshgrid(Y, X)
Z = nappe_clotures.values

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, label='taux clotures')

# Customize the z axis.
ax.set_zlim(Z.min(), Z.max())
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()


# ==========================================
# graph pour les taux de versements
# ==========================================

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.array(nappe_versements.index)
Y = np.array(nappe_versements.columns)
Y, X = np.meshgrid(Y, X)
Z = nappe_versements.values

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, label='taux clotures')

# Customize the z axis.
ax.set_zlim(Z.min(), Z.max())
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()
