using PyCall

export read_data, read_hod, read_my_hod

function read_hod(run, snapshot)
    mimicus = pyimport("mimicus")
    pos, vel = mimicus.read_raw_data(run=run, snapshot=snapshot,hod=true)
    return permutedims(pos), permutedims(vel)
end

function read_my_hod(run, snapshot, galaxy_type)
    dq = pyimport("dq")
    pos, vel = dq.read_hod_data(run=run, snapshot=snapshot,galaxy_type=galaxy_type)
    return permutedims(pos), permutedims(vel)
end


function read_data(run, snapshot, number_density, boxsize)
    dq = pyimport("dq")
    pos, vel, m200c = dq.read_halo_data(run=run, snapshot=snapshot)
    pos, vel = dq.cut_by_number_density(pos, vel, m200c, number_density, boxsize)
    return permutedims(pos), permutedims(vel)
end

function read_data(run, snapshot, min_mass, max_mass, boxsize)
    dq = pyimport("dq")
    pos, vel, m200c = dq.read_halo_data(run=run, snapshot=snapshot)
    mask = (max_mass .>= m200c .>= min_mass) 
    pos = pos[mask,:]
    vel = vel[mask,:]
    return permutedims(pos), permutedims(vel)
end
