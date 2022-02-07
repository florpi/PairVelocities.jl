using Pkg;Pkg.activate(".")
using PairVelocities
using StaticArrays
using BenchmarkTools
using DelimitedFiles

rbins = LinRange(0.,50,50)
rbins_c = 0.5*(rbins[2:end] + rbins[1:end-1])

filename = "moments.csv"

positions, velocities, boxsize, redshift = read_abacus()
println(size(positions))
positions = convert(Array{Float64}, positions)
velocities = convert(Array{Float64}, velocities)


moments = PairVelocities.compute_pairwise_velocity_moments(
    positions,
    velocities,
    rbins,
    boxsize,
)
open(filename; write=true) do f
    write(f, "r_c\tv_r\tsigma_r\tsigma_t\tskewness_r\tskewness_rt\tkurtosis_r\tkurtosis_t\tkurtosis_rt\n")
    writedlm(f, 
            zip(rbins_c, moments[2][:], 
            moments[3][:], moments[6][:], 
            moments[4][:], moments[7][:], 
            moments[5][:], moments[8][:], moments[9][:]))
    end

println("Wrote file !")
