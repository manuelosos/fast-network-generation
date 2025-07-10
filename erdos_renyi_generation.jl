
module Testsuite

using HDF5

include("fastnetworkgeneration/uniformRandomNetworks.jl")

using .UniformRandomNetworks

export main, generate_connected_erdos_renyi_network

struct ErdosRenyiNetwork
    network::SimpleNetwork
    n_nodes::Integer
    edge_probability::Real
    edge_probability_name::String
end


function nameof(network::ErdosRenyiNetwork)
    return "ER_n$(network.n_nodes)_$(network.edge_probability_name)"
end


function save_erdos_renyi_network(
    er_network::ErdosRenyiNetwork,
    path
)
    if !isdir(path)
        throw(error("Path does not exist or is not a directory"))
    end

    network_name = nameof(er_network)
    path = joinpath(path, network_name * ".hdf5")

    network_data = to_hdf5(er_network.network)

    er_attributes = Dict(
        "edge_probability" => er_network.edge_probability,
        "edge_probability_name" => er_network.edge_probability_name,
        "network_model" => "erdos renyi",
        "n_nodes" => er_network.n_nodes,
        "network_name" => network_name
    )
    
    
    h5open(path, "w") do fid
        network_data_group = create_group(fid, "network_data")
        for key in keys(network_data)
            network_data_group[key] = network_data[key]
        end

        for key in keys(er_attributes)
            attributes(network_data_group)[key] = er_attributes[key]
        end

    end

end


function generate_connected_erdos_renyi_network(n_nodes, edge_probability, edge_probability_name, path)

    res = generate_uniform_random_graph_geometric(n_nodes, edge_probability, network_dtype=NeighborList)
    connect_isolates!(res)

    er_res = ErdosRenyiNetwork(res, n_nodes, edge_probability, edge_probability_name)

    save_erdos_renyi_network(er_res, path)



end



function main()


    res = generate_uniform_random_graph_geometric(n_nodes, edge_probability, network_dtype=NeighborList)
    connect_isolates!(res)
    
    er_test = ErdosRenyiNetwork(res, n_nodes, edge_probability, "test-p")

    save_erdos_renyi_network(er_test, "/home/manuel/Documents/code/data/test_data/")


end

end

