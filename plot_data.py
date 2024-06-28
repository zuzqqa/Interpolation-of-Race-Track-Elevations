import matplotlib.pyplot as plt
import numpy as np
import main


def plot_elevation(elevations, track_name):
    """Plots the elevation profile of a route."""
    plt.figure(figsize=(10, 6))
    plt.plot(elevations, marker='', linestyle='-', color='red')
    plt.xlabel('Racetrack')
    plt.ylabel('Elevation')
    plt.title(f'{track_name} track elevation profile')
    plt.grid(True)
    plt.savefig(f'{track_name}_elevation.png')

    plt.show()


def plot_track_3d(coordinates, elevations, track_name):
    """Plots a 3D track of the route."""
    latitudes = [coord[0] for coord in coordinates]
    longitudes = [coord[1] for coord in coordinates]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(longitudes, latitudes, elevations, c='b', marker='o')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation')
    ax.set_title(f'{track_name} Elevation Profile')

    plt.savefig(f'{track_name}_elevation3D.png')
    plt.show()


def plot_interpolated_lagrange_chybechev(coordinates, elevations, track_name, n_nodes):
    """Creates a plot of Lagrange interpolation with Chebyshev nodes."""
    chybechev_idx = main.chebyshev_nodes(len(coordinates), n_nodes)
    chybechev_idx.reverse()

    x_indexes = np.array(chybechev_idx) / (len(coordinates) - 1)
    y_indexes = elevations[chybechev_idx]

    x_dense = np.linspace(0, 1, len(coordinates))
    y_plot = [main.lagrange_function(x, x_indexes, y_indexes) for x in x_dense]

    plt.plot(x_dense, y_plot, marker='', linestyle='-', label=f'Interpolation curve ({n_nodes} nodes)')
    plt.scatter(x_indexes, y_indexes, color='green', label=f'Interpolation nodes ({n_nodes})')
    plt.xlabel('Normalized distance')
    plt.ylabel('Elevation')
    plt.title(f'Interpolation plot {track_name}')
    plt.legend()
    plt.grid(True)


def plot_interpolated_lagrange(coordinates, elevations, track_name, n_nodes):
    """Creates a plot of Lagrange interpolation with uniformly distributed nodes."""
    uniform_idx = np.linspace(0, len(coordinates) - 1, n_nodes).astype(int)

    x_indexes = np.linspace(0, 1, n_nodes)
    y_indexes = elevations[uniform_idx]

    x_dense = np.linspace(0, 1, len(coordinates))
    y_plot = [main.lagrange_function(x, x_indexes, y_indexes) for x in x_dense]

    plt.plot(x_dense, y_plot, marker='', linestyle='-', label=f'Interpolation curve ({n_nodes} nodes)')
    plt.scatter(x_indexes, y_indexes, color='green', label=f'Interpolation nodes ({n_nodes})')
    plt.xlabel('Normalized distance')
    plt.ylabel('Elevation')
    plt.title(f'Interpolation plot {track_name}')
    plt.legend()
    plt.grid(True)


def plot_interpolated_cubic_spline(coordinates, elevations, race_track, n_nodes):
    """Creates a plot of interpolation using cubic splines with uniformly distributed nodes."""
    uniform_idx = np.linspace(0, len(coordinates) - 1, n_nodes).astype(int)

    x_indexes = np.linspace(0, 1, n_nodes)
    y_indexes = elevations[uniform_idx]

    a, b, c, d = main.cubic_spline_interpolation(x_indexes, y_indexes)

    x_dense = np.linspace(0, 1, len(coordinates))
    y_plot = []
    for x in x_dense:
        for i in range(len(x_indexes) - 1):
            if x_indexes[i] <= x < x_indexes[i + 1]:
                y = a[i] + b[i] * (x - x_indexes[i]) + c[i] * (x - x_indexes[i]) ** 2 + d[i] * (x - x_indexes[i]) ** 3
                y_plot.append(y)
                break
        else:
            i = len(x_indexes) - 2
            y = a[i] + b[i] * (x - x_indexes[i]) + c[i] * (x - x_indexes[i]) ** 2 + d[i] * (x - x_indexes[i]) ** 3
            y_plot.append(y)

    plt.plot(x_dense, y_plot, marker='', linestyle='-', label=f'Interpolation curve (Cubic Spline, {n_nodes} nodes)')
    plt.scatter(x_indexes, y_indexes, color='green', label=f'Interpolation nodes ({n_nodes})')
    plt.xlabel('Normalized distance')
    plt.ylabel('Elevation')
    plt.title(f'Interpolation plot for {race_track}')
    plt.legend()
    plt.grid(True)