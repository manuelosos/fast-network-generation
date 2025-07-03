

module UniformRandomNetworks

import LinearAlgebra.diagind as diagind

include("networkDatatypes.jl")

using .NetworkDataTypes

export generate_uniform_random_graph_geometric, compute_uniform_random_graph_PZER
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
        
    chunksize = 1024*4

    
    compute_uniform_random_graph_PZER(5, p, chunksize; verbose=false)



    compute_uniform_random_graph_PZER(n_nodes, p, chunksize; verbose=true)

    return

end

main()