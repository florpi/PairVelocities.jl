using Profile, PProf
using Pkg;Pkg.activate(".")
using PairwiseVelocities
using BenchmarkTools
using DelimitedFiles
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--snapshot"
            help = "an option with an argument"
            arg_type = Int
        "--nd"
            help = "an option with an argument"
            arg_type = Int
        "--min_run"
            help = "an option with an argument"
            arg_type = Int
        "--max_run"
            help = "another option with an argument"
            arg_type = Int
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

DATA_DIR = "/cosma7/data/dp004/dc-cues1/DarkQuest/pairwise_velocities/"
snapshot = parsed_args["snapshot"] 
boxsize = 2000.
r_max = 80.

number_densities = readdlm("/cosma6/data/dp004/dc-cues1/DarkQuest/xi/log10density_table.dat", ' ', Float32, '\n')
println(number_densities)
number_density_left = number_densities[parsed_args["nd"],1]
number_density_right = number_densities[parsed_args["nd"],2]
println("number densities = ( ", number_density_left, " , ",number_density_right," ) ")
rbins_c = readdlm("/cosma6/data/dp004/dc-cues1/DarkQuest/xi/separation.dat", '\t', Float64, '\n')
rbins_c = rbins_c[rbins_c .< r_max]

rbins = zeros(Float64, length(rbins_c)+1)

rbins[2:end-1] = rbins_c[1:end-1] + diff(rbins_c)/2.
rbins[1] = 2. *rbins_c[1] - rbins[2]
rbins[end] = 2. *rbins_c[end] - rbins[end-1]

r_max = maximum(rbins)
rbins_c = 0.5*(rbins[2:end] + rbins[1:end-1])

for run in parsed_args["min_run"]:parsed_args["max_run"]
    filename = "run$(run)_nd_$(abs(number_density_left))_$(abs(number_density_right))_snapshot$(snapshot).csv"
    println(filename)
    if number_density_left == number_density_right
        positions, velocities = read_data(
            run, snapshot, 10^number_density_left, boxsize
        )
        positions = convert(Array{Float64}, positions)
        velocities = convert(Array{Float64}, velocities)
        @time moments = PairwiseVelocities.compute_pairwise_velocity_moments(
                            positions,
                            velocities,
                            rbins,
                            boxsize,
        )
    else
        positions_left, velocities_left = read_data(
            run, snapshot, 10^number_density_left, boxsize
        )
        positions_left = convert(Array{Float64}, positions_left)
        velocities_left = convert(Array{Float64}, velocities_left)
        positions_right, velocities_right = read_data(
            run, snapshot, 10^number_density_right, boxsize
        )
        positions_right = convert(Array{Float64}, positions_right)
        velocities_right = convert(Array{Float64}, velocities_right)

        @time moments = PairwiseVelocities.compute_pairwise_velocity_moments(
                            positions_left,
                            velocities_left,
                            positions_right,
                            velocities_right,
                            rbins,
                            boxsize,
        )
    end
    open(DATA_DIR * filename; write=true) do f
        write(f, "# r_c v_r sigma_r sigma_t skewness_r skewness_rt kurtosis_r kurtosis_t kurtosis_rt \n")
        writedlm(f, 
                  zip(rbins_c, moments[2][:], 
                    moments[3][:], moments[6][:], 
                    moments[4][:], moments[7][:], 
                    moments[5][:], moments[8][:], moments[9][:]))
    end

    println("Wrote file for $(run)!")
end
