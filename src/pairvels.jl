using Distances
using CellLists
using LinearAlgebra
using NearestNeighbors

export get_pairwise_velocity_distribution
#export get_cross_pairwise_velocity_distribution


function get_periodic_difference( x, y, period)
    delta = abs(x - y)
    return ifelse(delta > 0.5 * period, delta - period, delta )* sign(x-y)
end

function get_pairwise_velocities!(
                    ret, 
                    dv, 
                    pos_pair_left,
                    pos_pair_right,
                    vel_pair_left,
                    vel_pair_right,
                    boxsize
    )
    for i in 1:length(ret)
        dv[i] = vel_pair_left[i] - vel_pair_right[i]
        ret[i] = get_periodic_difference(pos_pair_left[i], pos_pair_right[i], boxsize[i])
    end
    r = LinearAlgebra.norm(ret)
    v_r = LinearAlgebra.dot(dv,ret)/r
    #v_t = sqrt(LinearAlgebra.dot(dv, dv) - v_r*v_r)/sqrt(2.)
    return r, v_r
end

function get_objects_in_rmax(positions, r_max, boxsize)
    tranposed_positions = permutedims(positions)
    metric = PeriodicEuclidean(boxsize)
    balltree = BallTree(tranposed_positions, metric)
    return inrange(balltree, tranposed_positions, r_max, false)
end

function get_objects_in_rmax(left_positions, right_positions, r_max, boxsize)
    left_tranposed_positions = permutedims(left_positions)
    right_tranposed_positions = permutedims(right_positions)
    metric = PeriodicEuclidean(boxsize)
    balltree = BallTree(right_tranposed_positions, metric)
    return inrange(balltree, left_tranposed_positions, r_max, false)
end


function compute_pairwise_velocities(idxs,positions, velocities, boxsize, rbins)
    ret = zeros(Float64, size(positions)[2])
    dv = zeros(Float64, size(positions)[2])
    mean_v_r = zeros(Float64, (length(rbins)-1))
    n_pairs = zeros(Int32, (length(rbins)-1))
    for i in 1:size(positions,1) 
        for j in idxs[i]
            if j > i
                pos_i = @view positions[i,:]
                pos_j = @view positions[j,:]
                vel_i = @view velocities[i,:]
                vel_j = @view velocities[j,:]
                r, v_r = get_pairwise_velocities!(ret, dv, pos_i, pos_j, vel_i, vel_j, boxsize)
                if first(rbins) < r < last(rbins)
                    rbin = searchsortedfirst(rbins, r) - 1
                    mean_v_r[rbin] += v_r
                    n_pairs[rbin] += 1
                    end
            end
        end
    end
    mean_v_r[n_pairs .> 0] = mean_v_r[n_pairs .> 0]./n_pairs[n_pairs .> 0]
    return mean_v_r
end

function get_pairwise_velocity_distribution(
        positions, velocities,
        rbins,
        boxsize
        )
    r_max = maximum(rbins)
    idxs = get_objects_in_rmax(positions, r_max, boxsize)
    return compute_pairwise_velocities(idxs,positions, velocities, boxsize, rbins)
end

function compute_cross_pairwise_velocities(idxs, left_positions, left_velocities, right_positions, right_velocities,boxsize, rbins)
    ret = zeros(Float64, size(left_positions)[2])
    dv = zeros(Float64, size(left_positions)[2])
    mean_v_r = zeros(Float64, (length(rbins)-1))
    n_pairs = zeros(Int32, (length(rbins)-1))
    for i in 1:size(left_positions,1) 
        for j in idxs[i]
            pos_i = @view left_positions[i,:]
            pos_j = @view right_positions[j,:]
            vel_i = @view left_velocities[i,:]
            vel_j = @view right_velocities[j,:]
            r, v_r = get_pairwise_velocities!(ret, dv, pos_i, pos_j, vel_i, vel_j, boxsize)
            if first(rbins) < r < last(rbins)
                rbin = searchsortedfirst(rbins, r) - 1
                mean_v_r[rbin] += v_r
                n_pairs[rbin] += 1
                end
        end
    end
    mean_v_r[n_pairs .> 0] = mean_v_r[n_pairs .> 0]./n_pairs[n_pairs .> 0]
    return mean_v_r
end

function get_cross_pairwise_velocity_distribution(
        left_positions, left_velocities,
        right_positions, right_velocities,
        rbins,
        boxsize
        )
    r_max = maximum(rbins)
    idxs = get_objects_in_rmax(left_positions, right_positions, r_max, boxsize)
    return compute_cross_pairwise_velocities(idxs,left_positions, left_velocities, right_positions, right_velocities,boxsize, rbins)
end


function get_pairwise_velocity_distribution_brute_force(
        positions, velocities,
        rbins,
        boxsize
        )
    mean_v_r = zeros(Float64, (length(rbins)-1))
    n_pairs = zeros(Int32, (length(rbins)-1))
    for i in 1:size(positions,1) 
        for j in i+1:size(positions,1)
            r, v_r = get_pairwise_velocities(
                            positions[i,:], positions[j,:],
                            velocities[i,:], velocities[j,:],
                            boxsize)
            if first(rbins) < r < last(rbins)
                rbin = searchsortedfirst(rbins, r) - 1
                mean_v_r[rbin] += v_r
                n_pairs[rbin] += 1
            end
        end
    end
    mean_v_r[n_pairs .> 0] = mean_v_r[n_pairs .> 0]./n_pairs[n_pairs .> 0]
    return mean_v_r
end
