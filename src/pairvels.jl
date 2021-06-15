using CellListMap
using StaticArrays
using LinearAlgebra

export test 
export compute_pairwise_velocity_moments

global n_moments  = 9
function get_periodic_difference( x, y, period)
    delta = abs.(x - y)
    return ifelse.(delta .> 0.5 * period, delta - period, delta ) .* sign.(x-y)
end

function compute_pairwise_mean!(x,y,i,j,d2,hist,velocities, rbins,sides)
    d = get_periodic_difference(x,y,sides)
    r = sqrt.(d2)#LinearAlgebra.norm(d)
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
    hist[1][ibin] += 1
    hist[2][ibin] += v_r
    hist[3][ibin] += v_r * v_r
    hist[4][ibin] += v_r * v_r * v_r
    hist[5][ibin] += v_r * v_r * v_r * v_r
    hist[6][ibin] += v_t * v_t
    hist[7][ibin] += v_r * v_t * v_t
    hist[8][ibin] += v_t * v_t * v_t * v_t
    hist[9][ibin] += v_r * v_r * v_t * v_t
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

function get_pairwise_velocity_moments(
  positions, velocities,
  rbins,
  boxsize, 
  lc, box, n
)
    positions = reshape(reinterpret(SVector{3,Float64},positions),n)
    velocities = reshape(reinterpret(SVector{3,Float64},velocities),n)
    initlists!(positions,box,lc)
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
      hist, positions, box, lc,
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
    r_max = maximum(rbins)
    lc = LinkedLists(n)
    box = Box(Lbox, r_max)
    return get_pairwise_velocity_moments(
      positions,
      velocities,
      rbins,
      Lbox,
      lc, box, n
    )

end

function test()
    println("Using n = ", Threads.nthreads() , " threads")
    n_halos = 100_000
    boxsize = 250.
    Lbox = [boxsize,boxsize,boxsize]
    positions = boxsize .* rand(Float64, 3, n_halos)
    velocities = rand(Float64, 3, n_halos)
    rbins = [0.,2.,4.,6.,8.,10.]
      
    n = size(positions)[2]
    r_max = maximum(rbins)
    lc = LinkedLists(n)
    box = Box(Lbox, r_max)

    get_pairwise_velocity_radial_mean(
      positions,
      velocities,
      rbins,
      Lbox,
      lc, box, n
    )

end
