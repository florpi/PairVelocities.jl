using Profile, PProf
using Pkg;Pkg.activate(".")
using PairwiseVelocities
using StaticArrays
using BenchmarkTools
using DelimitedFiles

function readdata(N)
  println("Reading positions...")
  file = open("/cosma/home/dp004/dc-cues1/data_enrique/pos.dat","r")
  positions = Array{Float64}(undef,3,N)
  for i in 1:N
    line = readline(file)
    positions[:,i] = SVector{3,Float64}(parse.(Float64,split(line)))
  end
  close(file)

  println("Reading velocities...")
  file = open("/cosma/home/dp004/dc-cues1/data_enrique/vel.dat","r")
  velocities = Array{Float64}(undef,3,N)
  for i in 1:N
    line = readline(file)
    velocities[:,i] = SVector{3,Float64}(parse.(Float64,split(line)))
  end
  close(file)
  return positions,velocities
end

boxsize = 2000.
rbins = LinRange(0.,200,200)
r_max = maximum(rbins)
rbins_c = 0.5*(rbins[2:end] + rbins[1:end-1])

filename = "moments.csv"

positions, velocities = readdata(1000)

@time moments = PairwiseVelocities.compute_pairwise_velocity_moments(
                            positions,
                            velocities,
                            rbins,
                            boxsize,
)
@time moments = PairwiseVelocities.compute_pairwise_velocity_moments(
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
