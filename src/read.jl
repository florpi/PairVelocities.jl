using PyCall

export read_data

function read_data(run, snapshot, number_density, boxsize)
    mimicus = pyimport("mimicus")
    pos, vel, m200c = mimicus.read_raw_data(run=run, snapshot=snapshot)
    pos, vel = mimicus.cut_by_number_density(pos, vel, m200c, number_density, boxsize)
    return permutedims(pos), permutedims(vel)
end

function read_data(run, snapshot, min_mass, max_mass, boxsize)
    mimicus = pyimport("mimicus")
    pos, vel, m200c = mimicus.read_raw_data(run=run, snapshot=snapshot)
    mask = (max_mass .>= m200c .>= min_mass) 
    pos = pos[mask,:]
    vel = vel[mask,:]
    return permutedims(pos), permutedims(vel)
end
