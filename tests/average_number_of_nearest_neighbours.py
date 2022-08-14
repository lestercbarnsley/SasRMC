#%%

import numpy as np
import matplotlib.pyplot as plt

import sas_rmc
from sas_rmc import Vector

smeared_positions = np.genfromtxt(
    fname = r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Smeared_particle_positions_200.txt",
    skip_header = 1
)

positions = [Vector(row[0], row[1]) for row in smeared_positions]

average_radius = 137+50

def average_nearest_numbers(average_radius):
    a = []
    for i, p in enumerate(positions):
        a_ave = np.sum([p.distance_from_vector(p_2) < (average_radius) for j, p_2 in enumerate(positions) if j != i])
        a.append(a_ave)
    return np.average(a)

r = np.linspace(50, 1200, num = 200)
average_nns = [average_nearest_numbers(r_i) for r_i in r]
plt.plot(r,average_nns )

r_a = np.array([(r_i, average) for r_i, average in zip(r, average_nns)])

a_dash = np.gradient(average_nns) / np.gradient(r)
a_dash_2 = np.gradient(a_dash) / np.gradient(r)

plt.xlabel(r'Radius ($\AA$)')
plt.ylabel('Average nearest neighbours')
plt.show()

plt.plot(r, a_dash)
plt.show()
plt.plot(r, a_dash_2)
plt.show()

#%%

