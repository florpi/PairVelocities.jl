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
    cos_theta = ret[3]/r
    sin_theta = sqrt(ret[1]*ret[1]+ret[2]*ret[2])/r
    cos_phi = ret[1]/sqrt(ret[1]*ret[1]+ret[2]*ret[2])
    sin_phi = ret[2]/sqrt(ret[1]*ret[1]+ret[2]*ret[2])
    if(sqrt(ret[1]*ret[1]+ret[2]*ret[2]) < 1e-10)
        cos_phi = 1.0
        sin_phi = 0.0
    end
    v_t = dv[1] * cos_theta * cos_phi + dv[2] * cos_theta * sin_phi - dv[3] * sin_theta
    return r, v_r, v_t
end

function get_pairwise_velocities_projection(
                    pos_pair_left,
                    pos_pair_right,
                    vel_pair_left,
                    vel_pair_right,
                    los_direction,
                    boxsize,
    )
    r_perp, r_parallel = 0., 0.
    v_los = vel_pair_left[los_direction] - vel_pair_right[los_direction]
    for i in 1:length(pos_pair_left)
        dpos_i = get_periodic_difference(pos_pair_left[i], pos_pair_right[i], boxsize[i])
        if i == los_direction
            r_parallel = dpos_i
        else
            r_perp += dpos_i^2
        end
    end
    return sqrt(r_perp), r_parallel, v_los*sign(r_parallel)
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


function compute_pairwise_velocities_moments(idxs,positions, velocities, boxsize, rbins)
    ret = zeros(Float64, size(positions)[2])
    dv = zeros(Float64, size(positions)[2])
    first_order_r = zeros(Float64, (length(rbins)-1))
    second_order_r = zeros(Float64, (length(rbins)-1))
    second_order_t = zeros(Float64, (length(rbins)-1))
    third_order_r = zeros(Float64, (length(rbins)-1))
    third_order_cross = zeros(Float64, (length(rbins)-1))
    fourth_order_r = zeros(Float64, (length(rbins)-1))
    fourth_order_t = zeros(Float64, (length(rbins)-1))
    fourth_order_cross = zeros(Float64, (length(rbins)-1))
    n_pairs = zeros(Int32, (length(rbins)-1))
    for i in 1:size(positions,1) 
        for j in idxs[i]
            if j > i
                pos_i = @view positions[i,:]
                pos_j = @view positions[j,:]
                vel_i = @view velocities[i,:]
                vel_j = @view velocities[j,:]
                r, v_r, v_t = get_pairwise_velocities!(
                    ret, 
                    dv, 
                    pos_i, 
                    pos_j, 
                    vel_i, 
                    vel_j, 
                    boxsize
                )
                if first(rbins) < r < last(rbins)
                    rbin = searchsortedfirst(rbins, r) - 1
                    first_order_r[rbin] += v_r
                    second_order_r[rbin] += v_r * v_r
                    third_order_r[rbin] += v_r * v_r * v_r
                    fourth_order_r[rbin] += v_r * v_r * v_r * v_r
                    second_order_t[rbin] += v_t * v_t
                    fourth_order_t[rbin] += v_t * v_t * v_t * v_t
                    third_order_cross[rbin] += v_r * v_t * v_t
                    fourth_order_cross[rbin] += v_r * v_r * v_t * v_t
                    n_pairs[rbin] += 1
                    end
            end
        end
    end
    mask = (n_pairs .> 0)
    first_order_r[mask] = first_order_r[mask] ./ n_pairs[mask]
    second_order_r[mask] = second_order_r[mask] ./ n_pairs[mask] - first_order_r[mask].^2
    second_order_t[mask] = second_order_t[mask] ./ n_pairs[mask]
    third_order_r[mask] = (
                    third_order_r[mask] ./ n_pairs[mask] 
                    - 3. * first_order_r[mask] .* second_order_r[mask] 
                    - first_order_r[mask].^3.
    )
    third_order_cross[mask] = (
                               third_order_cross[mask] ./ n_pairs[mask] 
                               - first_order_r[mask] .* second_order_t[mask] 
    )
    fourth_order_r[mask] = (
                        fourth_order_r[mask] ./ n_pairs[mask] 
                        - 4. * first_order_r[mask] .* third_order_r[mask] 
                        - 6. * first_order_r[mask].^2. .* second_order_r[mask] 
                        - first_order_r[mask] .^4.
    )
    fourth_order_t[mask] = fourth_order_t[mask] ./ n_pairs[mask]
    fourth_order_cross[mask] = (
                        fourth_order_cross[mask] ./ n_pairs[mask] 
                        - 2. * third_order_cross[mask] .* first_order_r[mask] 
                        - first_order_r[mask].^2 .* second_order_t[mask]
    )
    return first_order_r, sqrt.(second_order_r), sqrt.(second_order_t), third_order_r ./ second_order_r.^(3. /2.), third_order_cross ./ second_order_r ./ second_order_t .^0.5, fourth_order_r ./ second_order_r .^ 2., fourth_order_t ./ second_order_t .^ 2., fourth_order_cross ./ second_order_r ./ second_order_t
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
    std_v_r = zeros(Float64, (length(rbins)-1))
    n_pairs = zeros(Int32, (length(rbins)-1))
    for i in 1:size(left_positions,1) 
        for j in idxs[i]
            pos_i = @view left_positions[i,:]
            pos_j = @view right_positions[j,:]
            vel_i = @view left_velocities[i,:]
            vel_j = @view right_velocities[j,:]
            r, v_r = get_pairwise_velocities!(
                    ret, 
                    dv, 
                    pos_i, 
                    pos_j, 
                    vel_i, 
                    vel_j, 
                    boxsize
                )
            if first(rbins) < r < last(rbins)
                rbin = searchsortedfirst(rbins, r) - 1
                mean_v_r[rbin] += v_r
                std_v_r[rbin] += v_r*v_r
                n_pairs[rbin] += 1
                end
        end
    end
    mean_v_r[n_pairs .> 0] = mean_v_r[n_pairs .> 0]./n_pairs[n_pairs .> 0]
    std_v_r[n_pairs .> 0] = sqrt(std_v_r[n_pairs .> 0]./n_pairs[n_pairs .> 0] - mean_v_r.^2)
    return mean_v_r, std_v_r
end

function get_cross_pairwise_velocity_distribution(
        left_positions, left_velocities,
        right_positions, right_velocities,
        rbins,
        boxsize
        )
    r_max = maximum(rbins)
    idxs = get_objects_in_rmax(left_positions, right_positions, r_max, boxsize)
    return compute_cross_pairwise_velocities(
                idxs,
                left_positions, 
                left_velocities, 
                right_positions, 
                right_velocities,
                boxsize, 
                rbins
            )
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
