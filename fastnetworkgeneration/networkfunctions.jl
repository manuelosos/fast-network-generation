
function connect_isolates!(network::SimpleNetwork)
    n_nodes = number_of_nodes(network)
    for node_index = 1:n_nodes
        if is_isolated(network, node_index)
            insert_edge!(network, (node_index, rand(1: n_nodes)))
        end
    end
end


