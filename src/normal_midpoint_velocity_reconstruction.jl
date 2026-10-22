abstract type EdgeMidpointReconstruction{N_MAX, TI, TF} <: LinearVoronoiOperator end
name_input(::EdgeMidpointReconstruction) = "edge"
name_output(::EdgeMidpointReconstruction) = "edge"
n_input(a::EdgeMidpointReconstruction) = n_output(a)

struct EdgeMidpointReconstructionLSq2{N_MAXP1, TI, TF} <: EdgeMidpointReconstruction{N_MAXP1, TI, TF}
    indices::SmVecArray{N_MAXP1, TI, 1}
    weights::SmVecArray{N_MAXP1, TF, 1}
end

function create_edgesOnEdge_e(::Val{NE}, e::Integer, cellsOnEdge, edgesOnCell, nEdgesOnCell) where {NE}

    TI = eltype(eltype(edgesOnCell))
    inds = SmallVector{NE, TI}()

    c1,c2 = cellsOnEdge[e]

    inds_e_c1 = edgesOnCell[c1]
    this_e_i = findfirst(isequal(e), inds_e_c1)::Int
    inds_e_c1_shifted = circshift(inds_e_c1, -this_e_i)
    nEdges_c1 = nEdgesOnCell[c1]

    for i in 1:(nEdges_c1-1)
        next_e = inds_e_c1_shifted[i]
        inds = push(inds, next_e)
    end

    inds_e_c2 = edgesOnCell[c2]
    this_e_i = findfirst(isequal(e), inds_e_c2)::Int
    inds_e_c2_shifted = circshift(inds_e_c2, -this_e_i)
    nEdges_c2 = nEdgesOnCell[c2]

    for i in 1:(nEdges_c2-1)
        next_e = inds_e_c2_shifted[i]
        inds = push(inds, next_e)
    end

    return inds
end

function compute_weightsOnEdge_normal_recon_lsq2_periodic!(edgesOnEdge::AbstractVector{<:SmallVector{NE, TI}}, weightsOnEdge::AbstractVector{<:SmallVector{NE, TF}}, edgePos, edgeMidpoints, edgeNormals, dvEdge, cellsOnEdge, edgesOnCell, nEdgesOnCell, xp::Number, yp::Number) where {NE, TI, TF}

    wdata = weightsOnEdge.data

    @parallel for e in eachindex(cellsOnEdge)
    @inbounds begin

        inds = pushfirst(create_edgesOnEdge_e(Val{NE}(), e, cellsOnEdge, edgesOnCell, nEdgesOnCell), e)

        edgesOnEdge[e] = inds

        bpos = edgePos[e]

        elements = map(@inline(i -> closest(bpos, @inbounds(edgePos[i]), xp, yp)), inds)

        elements_normal = map(@inline(i -> @inbounds(edgeNormals[i])), inds)

        # These are the weights to reconstruct ∇u⃗ at bpos
        @inline w∇v = compute_grad_weights_vec_lsq2(bpos, elements, elements_normal, 𝐢, 𝐣, dvEdge[e])

        Δx = closest(bpos, edgeMidpoints[e], xp, yp) - bpos
        nₑ = edgeNormals[e]

        w = map(∇u⃗ -> ((Δx ⋅ ∇u⃗) ⋅ nₑ), w∇v)
        w = Base.setindex(w, 1 + w[1], 1) 

        wdata[e] = fixedvector(w)

    end
    end
    return edgesOnEdge, weightsOnEdge
end

function compute_weightsOnEdge_normal_recon_lsq2_spherical!(edgesOnEdge::AbstractVector{<:SmallVector{NE, TI}}, weightsOnEdge::AbstractVector{<:SmallVector{NE, TF}}, edgePos, edgeMidpoints, edgeNormals, dvEdge, cellsOnEdge, edgesOnCell, nEdgesOnCell, R::Number) where {NE, TI, TF}

    wdata = weightsOnEdge.data

    @parallel for e in eachindex(cellsOnEdge)
    @inbounds begin

        inds = pushfirst(create_edgesOnEdge_e(Val{NE}(), e, cellsOnEdge, edgesOnCell, nEdgesOnCell), e)

        edgesOnEdge[e] = inds

        bpos_n = edgePos[e] / R

        elements = map(@inline(i -> edgePos[i] / R) , inds)

        elements_proj = project_points_to_tangent_plane(1, bpos_n, elements)

        elements_normal_proj = map(inds) do i
            @inline
            _n = @inbounds(edgeNormals[i])
            return normalize(_n - (_n ⋅ bpos_n)*bpos_n)
        end

        nₑ = edgeNormals[e]
        tₑ = normalize(bpos_n × nₑ)

        # These are the weights to reconstruct ∇u⃗ at bpos
        @inline w∇v = compute_grad_weights_vec_lsq2(bpos_n, elements_proj, elements_normal_proj, nₑ, tₑ, dvEdge[e] / R)

        b_mid_projected = project_points_to_tangent_plane(1, bpos_n, (edgeMidpoints[e] / R,))[1]

        Δx = b_mid_projected - bpos_n

        w = map(∇u⃗ -> ((Δx ⋅ ∇u⃗) ⋅ nₑ), w∇v)
        w = Base.setindex(w, 1 + w[1], 1) 

        wdata[e] = fixedvector(w)

    end
    end
    return edgesOnEdge, weightsOnEdge
end

function compute_weightsOnEdge_normal_recon_lsq2!(indices, w, cells::Cells{false}, edges::Edges{false})
    return compute_weightsOnEdge_normal_recon_lsq2_periodic!(
        indices, w,
        edges.position,
        edges.midpoint,
        edges.normal,
        edges.lengthDual,
        edges.cells,
        cells.edges,
        cells.nEdges,
        cells.x_period,
        cells.y_period
    )
end

function compute_weightsOnEdge_normal_recon_lsq2!(indices, w, cells::Cells{true}, edges::Edges{true})
    return compute_weightsOnEdge_normal_recon_lsq2_spherical!(
        indices, w,
        edges.position,
        edges.midpoint,
        edges.normal,
        edges.lengthDual,
        edges.cells,
        cells.edges,
        cells.nEdges,
        cells.sphere_radius,
    )
end

compute_weightsOnEdge_normal_recon_lsq2!(indices, w, mesh::AbstractVoronoiMesh) =
    compute_weightsOnEdge_normal_recon_lsq2!(indices, w, mesh.cells, mesh.edges)

function compute_weightsOnEdge_normal_recon_lsq2(mesh)

    nEdgesOnEdge = maximum(x->(x[1] + x[2] - 1), ((a,inds) -> @inbounds((a[inds[1]], a[inds[2]]))).((mesh.cells.nEdges,), mesh.edges.cells))

    indices =SmVecArray{nEdgesOnEdge, integer_type(mesh)}(mesh.edges.n)

    weights =SmallVectorArray(Vector{FixedVector{nEdgesOnEdge, float_type(mesh)}}(undef, mesh.edges.n), indices.length)

    return compute_weightsOnEdge_normal_recon_lsq2!(indices, weights, mesh)
end

EdgeMidpointReconstructionLSq2(m::AbstractVoronoiMesh) =
    EdgeMidpointReconstructionLSq2(compute_weightsOnEdge_normal_recon_lsq2(m)...)

method_name(::EdgeMidpointReconstructionLSq2) = "LSq2"

