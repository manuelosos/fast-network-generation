include("NetworkDataTypes.jl")
using .NetworkDataTypes


function main()
    adjmat = AdjacencyMatrix(5)
    insert_edge!(adjmat, (1, 2))
    println(adjmat)

    neighborlist = NeighborList(10)
    insert_edge!(neighborlist,(1,2))
    insert_edge!(neighborlist, (1,2), check_if_exists=true)
    println(neighborlist)
    delete_edge!(neighborlist, (1,2))

    delete_edge!(neighborlist, (1,2))
    println(neighborlist)
end

main()