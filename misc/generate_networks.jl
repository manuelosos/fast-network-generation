import Graphs
using ArgParse
using JSON
using Printf
using Distributed
using Random
@everywhere using Graphs, SparseArrays, NPZ, MatrixMarket


# === Konfiguration ===
@everywhere function abrv(n_nodes::Int, p::Float64; edge_density_abrv=nothing)

    if isnothing(edge_density_abrv )
        return "ER_p$(Int(round(p * 100000)))_N$(n_nodes)"
    else
        return "ER_p-$(edge_density_abrv)_N$(n_nodes)"
    end
end



@everywhere function save_sparse_matrix_npz(path::String, matrix::SparseMatrixCSC)
   data = Dict(
        "data" => matrix.nzval,
        "indices" => matrix.rowval,
        "indptr" => matrix.colptr,
        "shape" => collect(size(matrix))
        ) 

    npzwrite(string(path,".npz"), data) 
end


@everywhere function save_network(path::String, graph::SimpleGraph, metadata::Dict)
    @info "Saving network to $path"
    mkpath(dirname(path))

    adj_matrix = Graphs.LinAlg.adjacency_matrix(graph)
   

    # save_sparse_matrix_npz(path, adj_matrix)

    mmwrite(path, adj_matrix)

end


# === Hauptfunktion zur Netzwerkerzeugung ===
@everywhere function generate_network(save_path::String, n_nodes::Int, edge_density::Float64, force_no_isolates::Bool; edge_density_abrv=nothing, verbose=false)
    if edge_density >= 1
        error("edge_density must be < 1")
    end
    if n_nodes <= 0
        error("n_nodes must be > 0")
    end
    if edge_density < log(n_nodes)/n_nodes
        error("Edge density must be larger than $(log(n_nodes)/n_nodes)")
    end

    start_time = time()

    graph = erdos_renyi(n_nodes, edge_density)

    if force_no_isolates
        # einfache Methode um Isolierte zu verbinden
        isolated = findall(v -> degree(graph, v) == 0, vertices(graph))
    
        while !isempty(isolated)
            for v in isolated
                u = rand(1:n_nodes)
                while u == v || has_edge(graph, u, v)
                    u = rand(1:n_nodes)
                end
                add_edge!(graph, v, u)
            end
            isolated = findall(v -> degree(graph, v) == 0, vertices(graph))
        end
    end

    save_network(save_path, graph, Dict("edge_density" => edge_density))

    elapsed_time = time() - start_time
    if verbose
        @info("Elapsed time: $(elapsed_time) seconds")
    end
end


function save_graph(
    graph,
    metadata
)

    

end



function main()

    # Lade Pfade
    path_data = JSON.parsefile("../code/paths.json")
    global data_path = get(path_data, "data_path", "")
    global save_path_results = get(path_data, "save_path_networks", "")

    # CLI Parsing
    s = ArgParseSettings()
    @add_arg_table s begin
        "n_nodes"
            help = "The size of the network"
            arg_type = Int
        "edge_density"
            help = "Edge density (p) in the G(n, p) model"
            arg_type = Float64
        "--force_no_isolates", "-i"
            help = "Set to force no isolates"
            action = :store_true
        "--ensemble", "-e"
            help = "Create several networks"
            action = :store_true
        "--test"
            help = "Run with small test values"
            action = :store_true
    end

    parsed_args = parse_args(s)

    n_nodes = parsed_args["n_nodes"]
    edge_density = parsed_args["edge_density"]
    force_no_isolates = parsed_args["force_no_isolates"]
    ensemble = parsed_args["ensemble"]
    test = parsed_args["test"]



    if ensemble
        list_n_nodes = test ? [10, 11, 12] : [10, 100, 1000, 10000, 100000, 1000000]
        list_edge_densities = log.(list_n_nodes) ./ list_n_nodes        
        list_abrvs_edge_densities = ["crit$(n)" for n in list_n_nodes]


        args = [(n, p, edge_density_abrv) for n in list_n_nodes for (p, edge_density_abrv) in zip(list_edge_densities, list_abrvs_edge_densities) if p >= log(n)/n] 

        @sync @distributed for (n, p, edge_density_abrv) in args
            @info("$(n)n $(p)p")

            network_name = abrv(n, p, edge_density_abrv=edge_density_abrv)
            save_path =  joinpath(save_path_results, network_name)
            generate_network(save_path, n, p, true)
        end


    else
        generate_network(n_nodes, edge_density, force_no_isolates)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end