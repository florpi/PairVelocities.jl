using Profile, PProf
using Pkg;Pkg.activate(".")
using PairwiseVelocities
using StaticArrays
using BenchmarkTools
using DelimitedFiles

boxsize = 2000.
rbins = LinRange(0.,50,50)
rbins_c = 0.5*(rbins[2:end] + rbins[1:end-1])

filename = "moments.csv"
logn = -5.
run = 101
snapshot = 15

positions, velocities = read_data(run, snapshot, 10^logn, boxsize)
positions = convert(Array{Float64}, positions)
velocities = convert(Array{Float64}, velocities)


moments = PairwiseVelocities.compute_pairwise_velocity_moments(
    positions,
    velocities,
    rbins,
    boxsize,
)
open(filename; write=true) do f
    write(f, "# r_c v_r sigma_r sigma_t skewness_r skewness_rt kurtosis_r kurtosis_t kurtosis_rt \n")
    writedlm(f, 
            zip(rbins_c, moments[2][:], 
            moments[3][:], moments[6][:], 
            moments[4][:], moments[7][:], 
            moments[5][:], moments[8][:], moments[9][:]))
    end

println("Wrote file !")
