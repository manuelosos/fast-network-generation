module NetworkDataTypes


export SimpleNetwork, AdjacencyMatrix, NeighborList
export number_of_nodes
export insert_edge!, delete_edge!
export is_isolated, connect_isolates!
export infer_space_optimal_network_data_type, to_hdf5


abstract type SimpleNetwork end


include("networkfunctions.jl")

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


function number_of_nodes(adj_mat::AdjacencyMatrix)
    return size(adj_mat.adjacency_matrix)[1]
end


function is_isolated(adj_mat::AdjacencyMatrix, node_index::Integer)
    if sum(adj_mat.adjacency_matrix[node_index,:]) == 0
        return true
    end
    return false
end


function to_hdf5(adj_mat::AdjacencyMatrix)
    return Dict(
        "adjacency_matrix" => convert(Array{Bool}, adj_mat.adjacency_matrix)
    )

end


struct NeighborList <: SimpleNetwork
    neighbor_list::AbstractVector{Vector{Integer}}
end


function NeighborList(n_nodes::Integer)
    node_index_dtype = infer_unsigned_integer_dtype(n_nodes)
    return NeighborList([Vector{node_index_dtype}() for _ in 1:n_nodes])
end


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


function number_of_nodes(nlist::NeighborList)
    return size(nlist.neighbor_list)[1]
end


function is_isolated(nlist::NeighborList, node_index::Integer)
    if size(nlist.neighbor_list[node_index])[1] == 0
        return true
    end
    return false
end


"""
    to_hdf5(nlist::NeighborList)

Converts the neighborlist to an adjacency matrix in CSR format which is suitable for hdf5 saving.
"""
function to_hdf5(nlist::NeighborList)
    
    n_elements = sum([size(x)[1] for x in nlist.neighbor_list])

    col_ptr = Vector{Int64}(undef, n_elements)
    row_ptr = Vector{Int64}(undef, size(nlist.neighbor_list)[1]+1)

    col_ptr_index = 1

    n_nodes = size(nlist.neighbor_list)[1]

    for row_index = 1 : n_nodes
        
        row_ptr[row_index] = col_ptr_index
        neighbors = nlist.neighbor_list[row_index]
        n_neighbors = size(neighbors)[1]
        
        col_ptr[col_ptr_index: col_ptr_index + n_neighbors-1] = copy(neighbors)
        col_ptr_index += n_neighbors
    end
    row_ptr[end] = col_ptr_index

    # Converte to zero based index
    col_ptr .-= 1
    row_ptr .-= 1

    return Dict("adjacency_matrix_col_ptr" => col_ptr, "adjacency_matrix_row_ptr" => row_ptr)
    
end


"""
    infer_space_optimal_network_data_type(n_nodes, n_expected_edges, use_bitmatrix=true)

Infers a space optimal network representation type for a simple undirected graph with no edge weights 
based on the number of nodes and the expected number of edges.

Returns the matching datatype.

If `use_bitmatrix=true`, the adjacency matrix will be assumed to be a bitmatrix instead of an UInt8matrix.
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
