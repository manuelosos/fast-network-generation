

module UniformRandomNetworks

include("networkDatatypes.jl")
using .NetworkDataTypes



"""
    generate_uniform_random_graph_geometric(n_nodes, edge_probability, flipped=False)

Sequentially generates a G(n,p) random graph with the geometric algorithm as described in DOI: 10.1103/PhysRevE.71.036113

Set flipped=true to execute the algorithm for the inverse of the graph.

"""
function generate_uniform_random_graph_geometric(
    network::SimpleNetwork,
    n_nodes::Int,
    edge_probability::Float64;
    compute_inverse=false
)
    row_index:: Int = 1
    col_index:: Int = -1

    if compute_inverse
        adj_matrix = trues(n_nodes, n_nodes)
    else
        adj_matrix = falses(n_nodes, n_nodes)
    end

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
    return network
end

end
