using Profile, PProf
using PairwiseVelocities
using BenchmarkTools

n_halos = 1_000_000
boxsize = 250.
Lbox = [boxsize,boxsize, boxsize]
positions = boxsize .* rand(Float64, (n_halos,3))
println(size(positions))
velocities = rand(Float64, (n_halos,3))
rbins = [0.,2.,4.,6.,8.,10.]

@btime get_pairwise_velocity_distribution(positions, velocities, rbins, Lbox)
#Profile.clear()
#@profile get_pairwise_velocity_distribution(positions, velocities, rbins, Lbox)
#

