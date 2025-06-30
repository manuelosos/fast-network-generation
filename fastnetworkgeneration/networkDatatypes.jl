module NetworkDataTypes


export SimpleNetwork, AdjacencyMatrix, NeighborList, infer_space_optimal_network_data_type, insert_edge!, delete_edge!


abstract type SimpleNetwork end


struct AdjacencyMatrix <: SimpleNetwork
    adjacency_matrix::BitMatrix
end


function AdjacencyMatrix(n_nodes::Integer)
    return AdjacencyMatrix(falses(n_nodes, n_nodes))
end


function edge_exists(adj_mat::AdjacencyMatrix, indices::Tuple{Integer, Integer})
    return convert(Bool, adj_mat.adjacency_matrix[indices[1], indices[2]])
end


function insert_edge!(adj_mat::AdjacencyMatrix, indices::Tuple{Integer, Integer})
    adj_mat.adjacency_matrix[indices[1], indices[2]] = 1
    adj_mat.adjacency_matrix[indices[2], indices[1]] = 1
end


function delete_edge!(adj_mat::AdjacencyMatrix, indices::Tuple{Integer, Integer})
    adj_mat.adjacency_matrix[indices[1], indices[2]] = 0
    adj_mat.adjacency_matrix[indices[2], indices[1]] = 0
end


struct NeighborList <: SimpleNetwork
    neighbor_list::AbstractVector{Vector{Integer}}
end


function NeighborList(n_nodes::Integer)
    node_index_dtype = infer_unsigned_integer_dtype(n_nodes)
    return NeighborList([Vector{node_index_dtype}() for _ in 1:n_nodes])
end


"""
    edge_exists(nlist::NeighborList, indices::Tuple{Integer, Integer})

    Checks if an edge is present in the network.
    Method for Neighborlist assumes that if edge is listed as neighbor
"""
function edge_exists(nlist::NeighborList, indices::Tuple{Integer, Integer})

    if length(nlist.neighbor_list[indices[1]]) < length(nlist.neighbor_list[indices[2]])
        if indices[2] in nlist.neighbor_list[indices[1]]
            return true
        end
    elseif indices[1] in nlist.neighbor_list[indices[2]]
        return true
    end
    return false

end


"""
insert_edge!(nlist::Neighborlist, indices::Tuple{Integer, Integer}, check_if_exists::Bool=true)
"""
function insert_edge!(
    nlist::NeighborList,
    indices::Tuple{Integer, Integer};
    check_if_exists::Bool=true
    )
    
    if check_if_exists && edge_exists(nlist, indices)
        return
    end

    push!(nlist.neighbor_list[indices[1]], indices[2])
    push!(nlist.neighbor_list[indices[2]], indices[1])
end


function delete_edge!(nlist::NeighborList, indices::Tuple{Integer, Integer})
    deleteat!(nlist.neighbor_list[indices[1]], indexin( indices[2], nlist.neighbor_list[indices[1]]))
    deleteat!(nlist.neighbor_list[indices[2]], indexin( indices[1], nlist.neighbor_list[indices[2]]))
end


"""
    infer_space_optimal_network_data_type(n_nodes, n_expected_edges, use_bitmatrix=true)

Infers a space optimal network representation type for a simple undirected graph with no edge weights 
based on the number of nodes and the expected number of edges.

Returns the matching datatype.

If use_bitmatrix is true, the adjacency matrix will be assumed to be a bitmatrix instead of an UInt8matrix.
Note that internally the adjacency matrix will always be a Bitmatrix. 
This option is only relevant if the network will be saved on disk since some dataformats do not support BitArrays.
"""
function infer_space_optimal_network_data_type(
    n_nodes::Integer,
    n_expected_edges::Integer; 
    use_bitmatrix::Bool=true
)
    max_n_edges = n_nodes^2
    
    node_index_dtype = infer_unsigned_integer_dtype(n_nodes)
    
    expected_size_adjacency_matrix = max_n_edges
    if use_bitmatrix
        expected_size_adjacency_matrix /= 8 
    end

    expected_size_neighbor_list = n_expected_edges * sizeof(node_index_dtype) * 2

    if expected_size_adjacency_matrix < expected_size_neighbor_list
        network_dtype = AdjacencyMatrix
    else
        network_dtype = AdjacencyMatrix
    end

    return network_dtype
end


function infer_unsigned_integer_dtype(n)

    if n > typemax(UInt64)
        n_dtype = UInt128
    elseif n > typemax(UInt32)    
        n_dtype = UInt64
    elseif n > typemax(UInt16)
        n_dtype = UInt32
    elseif n > typemax(UInt8)
        n_dtype = UInt16
    elseif n > 2
        n_dtype = UInt8
    elseif n >=0
        n_dtype = Bool
    else
        throw(ArgumentError("Number n must be non-negative"))
    end
    return n_dtype
end

end
