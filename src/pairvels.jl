using CellListMap
using StaticArrays
using LinearAlgebra

export test 
export compute_pairwise_velocity_moments
export compute_pairwise_velocity_los_pdf

global n_moments  = 9

function compute_pairwise_mean!(x,y,i,j,d2,hist,velocities, rbins,sides)
    d = x - y
    r = sqrt.(d2)
    ibin = searchsortedfirst(rbins, r) - 1
    dv = velocities[i] - velocities[j]
    v_r = LinearAlgebra.dot(dv,d)/r
    cos_theta = d[3]/r
    sin_theta = sqrt(d[1]*d[1]+d[2]*d[2])/r
    cos_phi = d[1]/sqrt(d[1]*d[1]+d[2]*d[2])
    sin_phi = d[2]/sqrt(d[1]*d[1]+d[2]*d[2])
    if(sqrt(d[1]*d[1]+d[2]*d[2]) < 1e-10)
        cos_phi = 1.0
        sin_phi = 0.0
    end
    v_t = dv[1] * cos_theta * cos_phi + dv[2] * cos_theta * sin_phi - dv[3] * sin_theta
    if ibin > 0
        hist[1][ibin] += 1
        hist[2][ibin] += v_r # v_r
        hist[3][ibin] += v_r * v_r # std_r
        hist[4][ibin] += v_r * v_r * v_r # skew_r
        hist[5][ibin] += v_r * v_r * v_r * v_r # kur_r
        hist[6][ibin] += v_t * v_t # std_t
        hist[7][ibin] += v_r * v_t * v_t # skew_rt
        hist[8][ibin] += v_t * v_t * v_t * v_t # kur_t
        hist[9][ibin] += v_r * v_r * v_t * v_t # kur_rt
    end
    return hist
end

function compute_pairwise_mean!(x,y,i,j,d2,hist,velocities_left, velocities_right, rbins,sides)
    d = x - y
    r = sqrt.(d2)
    ibin = searchsortedfirst(rbins, r) - 1
    dv = velocities_left[i] - velocities_right[j]
    v_r = LinearAlgebra.dot(dv,d)/r
    cos_theta = d[3]/r
    sin_theta = sqrt(d[1]*d[1]+d[2]*d[2])/r
    cos_phi = d[1]/sqrt(d[1]*d[1]+d[2]*d[2])
    sin_phi = d[2]/sqrt(d[1]*d[1]+d[2]*d[2])
    if(sqrt(d[1]*d[1]+d[2]*d[2]) < 1e-10)
        cos_phi = 1.0
        sin_phi = 0.0
    end
    v_t = dv[1] * cos_theta * cos_phi + dv[2] * cos_theta * sin_phi - dv[3] * sin_theta
    if ibin > 0
        hist[1][ibin] += 1
        hist[2][ibin] += v_r
        hist[3][ibin] += v_r * v_r
        hist[4][ibin] += v_r * v_r * v_r
        hist[5][ibin] += v_r * v_r * v_r * v_r
        hist[6][ibin] += v_t * v_t
        hist[7][ibin] += v_r * v_t * v_t
        hist[8][ibin] += v_t * v_t * v_t * v_t
        hist[9][ibin] += v_r * v_r * v_t * v_t
    end
    return hist
end


function convert_histogram_into_moments!(hist)
    n_pairs = hist[1]
    mask = (n_pairs .> 0)
    hist[2][mask] = hist[2][mask] ./ n_pairs[mask]
    hist[3][mask] = hist[3][mask] ./ n_pairs[mask] - hist[2][mask].^2
    hist[6][mask] = hist[6][mask] ./ n_pairs[mask]
    hist[4][mask] = (
                hist[4][mask] ./ n_pairs[mask] 
                - 3. * hist[2][mask] .* hist[3][mask] 
                - hist[2][mask].^3.
    )
    hist[7][mask] = (
            hist[7][mask] ./ n_pairs[mask] 
            - hist[2][mask] .* hist[6][mask] 
    )
    hist[5][mask] = (
            hist[5][mask] ./ n_pairs[mask] 
            - 4. * hist[2][mask] .* hist[4][mask] 
            - 6. * hist[2][mask].^2. .* hist[3][mask] 
            - hist[2][mask] .^4.
    )
    hist[8][mask] = hist[8][mask] ./ n_pairs[mask]
    hist[9][mask] = (
        hist[9][mask] ./ n_pairs[mask] 
        - 2. * hist[7][mask] .* hist[2][mask] 
        - hist[2][mask].^2 .* hist[6][mask]
    )
    hist[3][:] = sqrt.(hist[3])
    hist[6][:] = sqrt.(hist[6])
    hist[4][:] = hist[4] ./ hist[3].^3
    hist[5][:] = hist[5] ./ hist[3].^4
    hist[8][:] = hist[8] ./ hist[6].^4
    hist[7][:] = hist[7] ./ hist[3].^2 ./ hist[6]
    hist[9][:] = hist[9] ./ hist[3].^2 ./ hist[6] .^2
end


function reduce_hist(hist,hist_threaded)
  hist = hist_threaded[1]
  for i in 2:Threads.nthreads()
    for moment in 1:n_moments
        hist[moment] .+= hist_threaded[i][moment]
    end
  end
  return hist
end

function reduce_hist_los(hist,hist_threaded)
  hist = hist_threaded[1]
  for i in 2:Threads.nthreads()
        hist .+= hist_threaded[i]
  end
  return hist
end


function get_pairwise_velocity_moments(
  positions, velocities,
  rbins,
  boxsize, 
  cl, box
)
    hist = (
            zeros(Int,length(rbins)-1), 
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
    )
    hist = map_pairwise!(
      (x,y,i,j,d2,hist) -> 
      compute_pairwise_mean!(x,y,i,j,d2,hist,velocities, rbins, boxsize),
      hist, box, cl,
      reduce=reduce_hist,
      parallel=true,
      show_progress=false,
    )
    convert_histogram_into_moments!(hist)
    return hist
end

function get_pairwise_velocity_moments(
  positions_left, velocities_left,
  positions_right, velocities_right,
  rbins,
  boxsize, 
  cl, box
)
    hist = (
            zeros(Int,length(rbins)-1), 
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
            zeros(Float64,length(rbins)-1),
    )
    hist = map_pairwise!(
      (x,y,i,j,d2,hist) -> 
      compute_pairwise_mean!(x,y,i,j,d2,hist,velocities_left, velocities_right,rbins, boxsize),
      hist, box, cl,
      reduce=reduce_hist
    )
    convert_histogram_into_moments!(hist)
    return hist
end


function compute_pairwise_velocity_moments(
    positions, velocities, rbins, boxsize
)
    println("Using n = ", Threads.nthreads() , " threads")
    Lbox = [boxsize,boxsize,boxsize]
    n = size(positions)[2]
    positions = reshape(reinterpret(SVector{3,Float64},positions),n)
    velocities = reshape(reinterpret(SVector{3,Float64},velocities),n)
    r_max = maximum(rbins)
    box = Box(Lbox, r_max, lcell=1)
    cl = CellList(positions,box)
    return get_pairwise_velocity_moments(
      positions,
      velocities,
      rbins,
      Lbox,
      cl,
      box,
    )

end

function compute_pairwise_velocity_moments(
    positions_left, velocities_left, positions_right, velocities_right, rbins, boxsize
)
    println("Using n = ", Threads.nthreads() , " threads")
    Lbox = [boxsize,boxsize,boxsize]
    positions_left = reshape(reinterpret(SVector{3,Float64},positions_left),size(positions_left)[2])
    velocities_left = reshape(reinterpret(SVector{3,Float64},velocities_left),size(velocities_left)[2])
    positions_right = reshape(reinterpret(SVector{3,Float64},positions_right),size(positions_right)[2])
    velocities_right = reshape(reinterpret(SVector{3,Float64},velocities_right),size(velocities_right)[2])
    r_max = maximum(rbins)
    box = Box(Lbox, r_max, lcell=1)
    cl = CellList(positions_left, positions_right, box)
    return get_pairwise_velocity_moments(
      positions_left,
      velocities_left,
      positions_right,
      velocities_right,
      rbins,
      Lbox,
      cl, box
    )

end


function compute_pairwise_los_pdf!(x,y,i,j,d2,hist,velocities,r_perp_bins, r_parallel_bins, vlos_bins,sides)
    d = x - y
    r_perp = sqrt(d[1]^2 + d[2]^2)
    r_parallel = d[3]
    vlos = (velocities[i][3] - velocities[j][3])*sign(r_parallel)
    if (r_perp > minimum(r_perp_bins)) & (r_perp < maximum(r_perp_bins)) & (r_parallel < maximum(r_parallel_bins)) & (r_parallel > minimum(r_parallel_bins)) & (vlos < maximum(vlos_bins)) & (vlos > minimum(vlos_bins))
        ibin_perp = searchsortedfirst(r_perp_bins, r_perp) - 1
        ibin_parallel = searchsortedfirst(r_parallel_bins, r_parallel) - 1
        ibin_vlos = searchsortedfirst(vlos_bins, vlos) - 1
        hist[ibin_perp, ibin_parallel, ibin_vlos] += 1
    end
    return hist
end

function compute_pairwise_los_pdf!(x,y,i,j,d2,hist,
        velocities_left,velocities_right,
        r_perp_bins, r_parallel_bins, vlos_bins,sides)
    d = x - y
    r_perp = sqrt(d[1]^2 + d[2]^2)
    r_parallel = d[3]
    vlos = (velocities_left[i][3] - velocities_right[j][3])*sign(r_parallel)
    if (r_perp < maximum(r_perp_bins)) & (r_parallel < maximum(r_parallel_bins)) & (r_parallel > minimum(r_parallel_bins)) & (vlos < maximum(vlos_bins)) & (vlos > minimum(vlos_bins))
        ibin_perp = searchsortedfirst(r_perp_bins, r_perp) - 1
        ibin_parallel = searchsortedfirst(r_parallel_bins, r_parallel) - 1
        ibin_vlos = searchsortedfirst(vlos_bins, vlos) - 1
        hist[ibin_perp, ibin_parallel, ibin_vlos] += 1
    end
    return hist
end


function get_pairwise_velocity_los_pdf(
  positions, velocities,
  r_perp_bins,
  r_parallel_bins,
  vlos_bins,
  boxsize, 
  cl, box
)
    hist = zeros(Int,(
            length(r_perp_bins)-1, 
            length(r_parallel_bins)-1,
            length(vlos_bins)-1
    )
    )
    return map_pairwise!(
      (x,y,i,j,d2,hist) -> 
      compute_pairwise_los_pdf!(x,y,i,j,d2,hist,velocities, 
        r_perp_bins, r_parallel_bins, vlos_bins, boxsize),
      hist, box, cl,
      reduce=reduce_hist_los, 
      parallel=true,
      show_progress=false,
    )
end

function get_pairwise_velocity_los_pdf(
  positions_left, velocities_left,
  positions_right, velocities_right,
  r_perp_bins,
  r_parallel_bins,
  vlos_bins,
  boxsize, 
  cl, box
)
    hist = zeros(Int,(
            length(r_perp_bins)-1, 
            length(r_parallel_bins)-1,
            length(vlos_bins)-1
    )
    )
    return map_pairwise!(
      (x,y,i,j,d2,hist) -> 
      compute_pairwise_los_pdf!(x,y,i,j,d2,hist,velocities_left, velocities_right,
        r_perp_bins, r_parallel_bins, vlos_bins, boxsize),
      hist, box, cl,
      reduce=reduce_hist_los,
      parallel=true,
      show_progress=false,
    )
end


function compute_pairwise_velocity_los_pdf(
    positions, velocities, r_perp_bins, r_parallel_bins, vlos_bins, boxsize
)
    println("Using n = ", Threads.nthreads() , " threads")
    Lbox = [boxsize,boxsize,boxsize]
    r_max = sqrt(maximum(r_perp_bins)^2 + maximum(r_parallel_bins)^2)
    positions = reshape(reinterpret(SVector{3,Float64},positions),size(positions)[2])
    velocities = reshape(reinterpret(SVector{3,Float64},velocities),size(velocities)[2])
    box = Box(Lbox, r_max, lcell=1)
    cl = CellList(positions,box)
    return get_pairwise_velocity_los_pdf(
      positions,
      velocities,
      r_perp_bins,
      r_parallel_bins,
      vlos_bins,
      Lbox,
      cl, box,
    )

end

function compute_pairwise_velocity_los_pdf(
    positions_left, velocities_left,
    positions_right, velocities_right,
    r_perp_bins, r_parallel_bins, vlos_bins, boxsize
)
    println("Using n = ", Threads.nthreads() , " threads")
    Lbox = [boxsize,boxsize,boxsize]
    r_max = sqrt(maximum(r_perp_bins)^2 + maximum(r_parallel_bins)^2)
    positions_left = reshape(reinterpret(SVector{3,Float64},positions_left),size(positions_left)[2])
    velocities_left = reshape(reinterpret(SVector{3,Float64},velocities_left),size(velocities_left)[2])
    positions_right = reshape(reinterpret(SVector{3,Float64},positions_right),size(positions_right)[2])
    velocities_right = reshape(reinterpret(SVector{3,Float64},velocities_right),size(velocities_right)[2])

    box = Box(Lbox, r_max, lcell=1)
    cl = CellList(positions_left,positions_right,box)
    return get_pairwise_velocity_los_pdf(
      positions_left,
      velocities_left,
      positions_right,
      velocities_right,
      r_perp_bins,
      r_parallel_bins,
      vlos_bins,
      Lbox,
      cl, box,
    )

end

