using Test
include("../fastnetworkgeneration/networkDatatypes.jl")
using .NetworkDataTypes



@testset "NetworkDataTypes" begin

    @testset "Adjacency matrix" begin
        
        n_nodes = 10
        test_matrix = AdjacencyMatrix(n_nodes)
        result_matrix = falses((n_nodes, n_nodes))
        @test test_matrix.adjacency_matrix == result_matrix 
        
        result_matrix[3, 4] = 1
        result_matrix[4, 3] = 1

        insert_edge!(test_matrix, (3,4))

        @test test_matrix.adjacency_matrix == result_matrix

         
    end

end