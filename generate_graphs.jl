using ArgParse
using JSON
using Distributed

@everywhere using HDF5
@everywhere include("fast_gnp.jl")



 @everywhere function save_graph_adjacency_matrix(
    adjacency_matrix,
    path,
    metadata
)
    if !endswith(path, ".hdf5")
        path = path * ".hdf5"
    end
    h5open(path, "w") do fid
        network_group = create_group(fid, "network")
        network_group["adjacency_matrix"] = convert(Array{Bool}, adjacency_matrix)

        for key in keys(metadata)
            attributes(network_group)[key] = metadata[key]
        end
    end

end




function generate_graphs(
    n_nodes_list,
    edge_probabilities,
    edge_probability_names,
    save_path
)
	
	
    args = [(n, p, edge_density_abrv) for n in n_nodes_list for (p, edge_density_abrv) in zip(edge_probabilities, edge_probability_names) if p >= log(n)/n] 

    @sync @distributed for (n, p, edge_density_abrv) in args
        @info("$(n)n $(p)p")

        graph_name = "ER_n$(n)_$(edge_density_abrv)"

        adj_matrix = generate_uniform_random_graph(n, p)
        connect_isolates!(adj_matrix)

        save_graph_adjacency_matrix(
            adj_matrix, 
            joinpath(save_path, graph_name), 
            Dict(
                "network_name" => graph_name,
                "network_model" => "erdos-renyi",
                "n_nodes" => n,
                "edge_probability" => p,
                )
            )
    end
    

end



if abspath(PROGRAM_FILE) == @__FILE__


    
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "--test"
            help = "Run with test values and save results in current working directory."
            action = :store_true
    end

    parsed_args = parse_args(arg_settings)

    test = parsed_args["test"]

    path_data = JSON.parsefile("../code/paths.json")
    data_path = path_data["data_path"]

    save_path_results = test ? "" : path_data["save_path_networks"]
    n_nodes_list = test ? [10, 100, 1000] : [10, 100, 1000, 10000, 100000, 1000000]


    edge_probs = log.(n_nodes_list) ./ n_nodes_list        
    edge_prob_names = ["p-crit-$(n)" for n in n_nodes_list]


    generate_graphs(
        n_nodes_list,
        edge_probs,
        edge_prob_names,
        save_path_results
    ) 
end
