using Test
using LinearAlgebra

include("../fastnetworkgeneration/networkGeneration.jl")
using .UniformRandomNetworks
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


@testset "Graph Generation" begin

    @testset "Geometric Algorithm" begin

        @testset "Type Correctness" begin
        n_nodes = 10
        edge_probability = 0.5

        test_network = generate_uniform_random_graph_geometric(n_nodes, edge_probability)

        @test typeof(test_network) == AdjacencyMatrix

        test_network = generate_uniform_random_graph_geometric(n_nodes, edge_probability, network_dtype=NeighborList)

        @test typeof(test_network) == NeighborList
        end


        @testset "Value Correctness" begin
        
        n_nodes = 10
        edge_probability = 0.5

        test_network = generate_uniform_random_graph_geometric(n_nodes, edge_probability)
        result = begin
            result = trues((n_nodes, n_nodes))
            result[diagind(result)] .= 0
            result
        end


        @test test_network.adjacency_matrix == result skip=true
        


        end 
    end
end