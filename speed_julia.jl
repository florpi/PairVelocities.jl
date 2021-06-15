using Profile, PProf
using PairwiseVelocities
using BenchmarkTools

n_halos = 100_000
boxsize = 250.
Lbox = [boxsize,boxsize, boxsize]
positions = boxsize .* rand(Float64, (n_halos,3))
println(size(positions))
velocities = rand(Float64, (n_halos,3))
rbins = [0.,2.,4.,6.,8.,10.]

@btime PairwiseVelocities.get_pairwise_velocity_radial_mean(positions,  velocities, rbins, Lbox)

@btime PairwiseVelocities.get_pairwise_velocity_radial_mean_cell_lists(positions,  velocities, rbins, Lbox)

#idxs = PairwiseVelocities.get_objects_in_rmax(positions,  10., Lbox)

#@btime PairwiseVelocities.get_objects_in_rmax(positions,  10., Lbox)
#@btime PairwiseVelocities.compute_pairwise_velocities(idxs, positions, velocities, Lbox, rbins)

#@btime PairwiseVelocities.compute_pairwise_velocities(idxs, positions, velocities, Lbox, rbins)
#@btime get_pairwise_velocity_distribution(positions, velocities, rbins, Lbox)
#Profile.clear()
#@profile get_pairwise_velocity_distribution(positions, velocities, rbins, Lbox)
#

