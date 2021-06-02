import numpy as np
import time
from halotools.mock_observables import mean_radial_velocity_vs_r

n_halos = 1_000_000
n_threads = 1
boxsize = 250.
rbins = [0.,2.,4.,6.,8.,10.]
Lbox = [boxsize,boxsize, boxsize]
positions = boxsize*np.random.random((n_halos,3))
velocities = np.random.random((n_halos,3))
print(positions.shape)
t0 = time.time()
v_12_halotools = mean_radial_velocity_vs_r(
    positions, 
    velocities, 
    rbins_absolute=rbins,
    period=Lbox,
    num_threads=1,
)
t_halotools = time.time()
print(f"Halotools took {t_halotools-t0} seconds")


