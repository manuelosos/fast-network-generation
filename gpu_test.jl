include("fastnetworkgeneration/uniformRandomNetworks.jl")

using .UniformRandomNetworks

function main()

    n_nodes = 1000
    edge_probability = 0.5
    chunksize = 1024*64*2

    generate_uniform_random_graph_PZER(5, edge_probability; gpu_chunksize=1024, verbose=false)
    #ENV["JULIA_DEBUG"] = Main
    total_execution_time_start = time()
    generate_uniform_random_graph_PZER(n_nodes, edge_probability; gpu_chunksize=chunksize)
    total_execution_time = time()-total_execution_time_start

    println("total execution time: $(total_execution_time/1_000_000)")
    println("") 

end


main()