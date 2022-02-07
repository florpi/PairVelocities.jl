using Pkg;Pkg.activate(".")
using PairVelocities
using StaticArrays
using BenchmarkTools
using DelimitedFiles

rbins = LinRange(0.001,150.,150)
rbins_c = 0.5*(rbins[2:end] + rbins[1:end-1])

filename = "moments.csv"

positions, velocities, boxsize, redshift = read_abacus()
println(positions[:][1:20])
N = size(positions)[end]
println("N halos")
println(N)
positions = convert(Array{Float64}, positions)
velocities = convert(Array{Float64}, velocities)


moments = PairVelocities.compute_pairwise_velocity_moments(
    positions,
    velocities,
    rbins,
    boxsize,
)
println(length(moments))
DD = moments[1][:]
println("counts")
println(DD)
RR = (N * (N-1)/boxsize^3 * 4/3 * pi) .* diff(rbins.^3)
xi = DD./RR .- 1.
open(filename; write=true) do f
    write(f, "r_c\txi\tv_r\tsigma_r\tsigma_t\tskewness_r\tskewness_rt\tkurtosis_r\tkurtosis_t\tkurtosis_rt\n")
    writedlm(f, 
             zip(rbins_c, xi, moments[2][:], 
            moments[3][:], moments[6][:], 
            moments[4][:], moments[7][:], 
            moments[5][:], moments[8][:], moments[9][:]))
    end

println("Wrote file !")
