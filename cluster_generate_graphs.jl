using ArgParse
using JSON
using Distributed

include("erdos_renyi_generation.jl")

@everywhere using .Testsuite





function generate_networks(
    n_nodes_list,
    edge_probabilities,
    edge_probability_names,
    save_path
)
	
	
    args = [(n, p, edge_density_abrv) for n in n_nodes_list for (p, edge_density_abrv) in zip(edge_probabilities, edge_probability_names) if p >= log(n)/n] 

    for (n, p, edge_density_abrv) in args
        @info("$(n)n $(p)p")

        generate_connected_erdos_renyi_network(n, p, edge_density_abrv, save_path)
    
    end
    

end



if abspath(PROGRAM_FILE) == @__FILE__

    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "save_path"
            help = "Save path for the network"
            arg_type=String
        "--test"
            help = "Run with test values and save results in current working directory."
            action = :store_true
    end
    println("started")

    parsed_args = parse_args(arg_settings)

    test = parsed_args["test"]
    save_path = parsed_args["save_path"]

    save_path_results = test ? "" : save_path
    n_nodes_list = test ? [10, 100, 1000] : [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]


    edge_probs = log.(n_nodes_list) ./ n_nodes_list        
    edge_prob_names = ["p-crit-$(n)" for n in n_nodes_list]


    generate_networks(
        n_nodes_list,
        edge_probs,
        edge_prob_names,
        save_path_results
    ) 
end
