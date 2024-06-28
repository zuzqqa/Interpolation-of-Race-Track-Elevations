import os

import numpy as np
from matplotlib import pyplot as plt

import data_extractor
import plot_data

files = [('SingaporeGP.txt', [9, 10, 11, 12]), ('Monza.txt', [8, 9, 10]), ('Mugello.txt', [8, 9, 10]), ('SilverstoneGP.txt', [8, 9, 10]), ('SpaFrancorchampsGP.txt', [8, 9, 10])]
output_dir = 'plots'


def chebyshev_nodes(x_data, nodes_num):
    """Generates Chebyshev nodes for the given data size and number of nodes."""
    indices = []

    for k in range(nodes_num):
        index = int(((x_data - 1) / 2) * (1 + np.cos(np.pi * (2 * k + 1) / (2 * nodes_num))))
        indices.append(index)

    return indices


def lagrange_function(val, x_inter, y_inter):
    """Calculates the Lagrange interpolation for a given value based on the provided x and y interpolation nodes."""
    out = 0
    n = len(x_inter)
    for i in range(n):
        term = 1
        for j in range(n):
            if i != j:
                if abs(x_inter[i] - x_inter[j]) < 1e-10:
                    continue
                term *= (val - x_inter[j]) / (x_inter[i] - x_inter[j])
        out += term * y_inter[i]
    return out


def cubic_spline_interpolation(x, y):
    """Performs cubic spline interpolation on the given x and y data points and returns the coefficients of the cubic spline functions."""
    n = len(x) - 1
    h = [x[i+1] - x[i] for i in range(n)]
    alpha = [3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]) for i in range(1, n)]
    alpha.insert(0, 0)

    l = [1] * (n + 1)
    mu = [0] * (n + 1)
    z = [0] * (n + 1)

    for i in range(1, n):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[n] = 1
    z[n] = 0
    c = [0] * (n + 1)
    b = [0] * n
    d = [0] * n

    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])

    return y, b, c, d


if __name__ == '__main__':
    for file in files:
        coordinates = data_extractor.extract_coordinates(file[0])
        elevations = data_extractor.get_elevation(coordinates)
        sorted_coords, sorted_elevations = data_extractor.nearest_neighbor_sort(coordinates, elevations)

        sorted_coords = np.array(sorted_coords)
        sorted_elevations = np.array(sorted_elevations)

        plot_data.plot_elevation(sorted_elevations, file[0])
        plot_data.plot_track_3d(sorted_coords, sorted_elevations, file[0])

        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, len(sorted_elevations)), sorted_elevations, marker='', linestyle='-', color='red',
                 label='Original data')

        for n_nodes in file[1]:
            plot_data.plot_interpolated_lagrange(sorted_coords, sorted_elevations, file[0], n_nodes)

        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(output_dir, f'{file[0]}_interpolation_lagrange.png'))
        plt.show()

        plt.plot(np.linspace(0, 1, len(sorted_elevations)), sorted_elevations, marker='', linestyle='-', color='red',
                 label='Original data')
        plot_data.plot_interpolated_cubic_spline(sorted_coords, sorted_elevations, file[0], 20)
        plt.savefig(os.path.join(output_dir, f'{file[0]}_cubic_spline_interpolation_20_nodes.png'))
        plt.show()

        plt.plot(np.linspace(0, 1, len(sorted_elevations)), sorted_elevations, marker='', linestyle='-', color='red',
                 label='Original data')
        plot_data.plot_interpolated_cubic_spline(sorted_coords, sorted_elevations, file[0], 60)
        plt.savefig(os.path.join(output_dir, f'{file[0]}_cubic_spline_interpolation_60_nodes.png'))
        plt.show()

        plt.plot(np.linspace(0, 1, len(sorted_elevations)), sorted_elevations, marker='', linestyle='-', color='red',
                 label='Original data')
        plot_data.plot_interpolated_cubic_spline(sorted_coords, sorted_elevations, file[0], 100)
        plt.savefig(os.path.join(output_dir, f'{file[0]}_cubic_spline_interpolation_100nodes.png'))

        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, len(sorted_elevations)), sorted_elevations, marker='', linestyle='-', color='red', label='Original data')

        plot_data.plot_interpolated_lagrange_chybechev(sorted_coords, sorted_elevations, file[0], 15)
        plt.savefig(os.path.join(output_dir, f'{file[0]}_lagrange_chybechev_15_nodes.png'))
        plt.show()
