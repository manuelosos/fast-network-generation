
"""
    generate_uniform_random_graph_geometric(n_nodes, edge_probability, flipped=False)

Sequentially generates a G(n,p) random graph with the geometric algorithm as described in DOI: 10.1103/PhysRevE.71.036113

Set `compute_inverse=true` to execute the algorithm for the inverse of the graph.

"""
function sample_uniform_random_graph_geometric!(
    network::SimpleNetwork,
    n_nodes::Integer,
    edge_probability::Real;
    compute_inverse=false
)

    if compute_inverse 
        edge_probability = 1-edge_probability
    end

    if edge_probability == 0
        throw(error("Unimplemented"))
    end
        

    row_index:: Integer = 1
    col_index:: Integer = -1

    while row_index < n_nodes
        r = rand()
        col_index = col_index + 1 + floor(log(1 - r) / log(1 - edge_probability))
        while col_index >= row_index && row_index < n_nodes
            col_index -= row_index
            row_index += 1
        end
        if row_index < n_nodes
            if compute_inverse
                delete_edge!(network, (row_index+1, col_index+1))
            else
                insert_edge!(network, (row_index+1, col_index+1))
            end

        end
    end
end



function generate_uniform_random_graph_geometric(
    n_nodes::Integer,
    edge_probability::Real;
    network_dtype::typeof(SimpleNetwork) = infer_space_optimal_network_data_type(n_nodes, round(UInt64, edge_probability*n_nodes*(n_nodes-1)/2))
)

    compute_inverse = false

    if network_dtype == AdjacencyMatrix 

        if edge_probability <= 0.5
            network = AdjacencyMatrix(falses((n_nodes, n_nodes)))
        else
            network = AdjacencyMatrix(trues((n_nodes, n_nodes)))
            network.adjacency_matrix[diagind(network.adjacency_matrix)] .= false
            compute_inverse = true
        end

    elseif network_dtype == NeighborList
        network = NeighborList(n_nodes)
    end

    sample_uniform_random_graph_geometric!(network, n_nodes, edge_probability, compute_inverse=compute_inverse)

    return network
end
