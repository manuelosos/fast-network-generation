

module UniformRandomNetworks

import LinearAlgebra.diagind as diagind


include("networkDatatypes.jl")

using .NetworkDataTypes

export generate_uniform_random_graph_geometric
export generate_uniform_random_graph_PZER
export connect_isolates!
export NeighborList, AdjacencyMatrix, SimpleNetwork
export to_hdf5

export compute_edge_skips_gpu, compute_edge_coordinates_gpu, linear_to_uppertriagonal_index

include("sampling_algorithms/geometric.jl")
include("sampling_algorithms/PZER.jl")


end


using .UniformRandomNetworks
function test_new_gpu_alg()

    
    chunk_size = 64*1024
    edge_probability = 0.5
    n_nodes = 10000

    println("Starting precompile runs")
    compute_edge_skips_gpu(1024, edge_probability, 1)
    compute_edge_coordinates_gpu(1024, n_nodes, edge_probability, 1)
    println("finished precompile")

    println("Starting runs")


    execution_time = 0

    n_trials = 100

    for i =1:n_trials
        start_time = time()
        compute_edge_skips_gpu(chunk_size, edge_probability, 1)
        execution_time += time() - start_time 
    end



    #start_time_edge_coordinates = time()
    #compute_edge_coordinates_gpu(chunk_size, n_nodes, edge_probability, 1)
    #elapsed_time_edge_coordinates = time() - start_time_edge_coordinates




    println("edge")
    println("Chunksize: $(chunk_size)")
    println("Estimated nodesize $(sqrt(2*chunk_size))")
    println("Average execution time after $(n_trials) runs: $(execution_time/n_trials)")
    #println("Edge skips: ", elapsed_time_edge_skips)
    #println("Edge_coordinates: ", elapsed_time_edge_coordinates)
    
end


function test_new_alg()

    chunk_size = 64*1024
    edge_probability = 0.5
    n_nodes = 1000000

    print("Starting precompile")
    generate_uniform_random_graph_PZER(10, 0.5, gpu_chunksize=1024)
    
    
    generate_uniform_random_graph_PZER(n_nodes, edge_probability; gpu_chunksize=chunk_size)

end

test_new_gpu_alg()