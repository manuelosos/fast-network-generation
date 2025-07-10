

module UniformRandomNetworks

import LinearAlgebra.diagind as diagind


include("networkDatatypes.jl")

using .NetworkDataTypes

export generate_uniform_random_graph_geometric
export sample_uniform_random_graph_PZER
export connect_isolates!
export NeighborList, AdjacencyMatrix, SimpleNetwork
export to_hdf5

include("sampling_algorithms/geometric.jl")
include("sampling_algorithms/PZER.jl")


end

using .UniformRandomNetworks

function main()
    n_nodes = 5000
    
    p = 0.5
        
    chunksize = 1024*64

    
    @info "Starting precompile"
    sample_uniform_random_graph_PZER(5, p; gpu_chunksize=chunksize, verbose=false)



    sample_uniform_random_graph_PZER(n_nodes, p, gpu_chunksize=chunksize; verbose=true)

    return

end

main()
