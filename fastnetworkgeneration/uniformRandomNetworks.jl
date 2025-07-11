

module UniformRandomNetworks

import LinearAlgebra.diagind as diagind


include("networkDatatypes.jl")

using .NetworkDataTypes

export generate_uniform_random_graph_geometric
export generate_uniform_random_graph_PZER
export connect_isolates!
export NeighborList, AdjacencyMatrix, SimpleNetwork
export to_hdf5

include("sampling_algorithms/geometric.jl")
include("sampling_algorithms/PZER.jl")


end


