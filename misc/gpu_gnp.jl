using Plots
using CUDA
using BenchmarkTools
using Statistics
using Base.Threads


function compute_single_edge_skip(edge_probability::Float64)
    return max(1, ceil(log(1-edge_probability, rand(Float64))))
end


function compute_edge_skips_gpu(
    n_skips::Int,
    edge_probability::Real,
    start_index::Int
)
    skip_values = CuArray{Float64}(undef, n_skips)
    rng_gen(_) = compute_single_edge_skip(edge_probability)
    skip_values = CUDA.map(rng_gen, skip_values)   
    skip_values = CUDA.accumulate(+, skip_values)

    skip_values .+= start_index 

    return skip_values
end


function gpu_compute_loop(
    buffer::Channel,
    n_nodes,
    edge_probabiltiy,
    chunk_size
)

    current_adj_mat_index = 0
    while true
        
        edge_list_d = compute_edge_skips_gpu(chunk_size, edge_probabiltiy, current_adj_mat_index)
        edge_list_h = Array(edge_list_d)
        current_adj_mat_index = convert(Int, edge_list_h[end])

        put!(buffer, edge_list_h)
        
        if edge_list_h[end] > n_nodes^2/2
            return
            break
        end
    end
end


function cpu_compute_loop!(
    adj_mat::BitMatrix,
    buffer::Channel,
    n_nodes::Int;
    flipped::Bool=false,
)


    row_index::Int = 1
    row_index_bound = 1
    col_index_bound = 0

    for edge_list in buffer
        for edge_index in edge_list
            while row_index_bound < edge_index 
                row_index +=1
                col_index_bound = row_index_bound 
                row_index_bound = (row_index^2+row_index)/2
            end
           
            col_index::Int = edge_index - col_index_bound
            
            if row_index < n_nodes && col_index <= n_nodes
                adj_mat[row_index+1, col_index] = !flipped
                adj_mat[col_index, row_index+1] = !flipped
            end 
        end
    end
end


function compute_uniform_random_graph_PZER(n_nodes, edge_probability)
    

    chunksize = 1024
    buffer = Channel{Vector{Float64}}(5)

    adj_mat = falses(n_nodes, n_nodes)

    task1 = @spawn gpu_compute_loop(buffer, n_nodes, edge_probability, chunksize)
    task2 = @spawn cpu_compute_loop!(adj_mat, buffer, n_nodes)

    wait(task1)
    wait(task2)

    return adj_mat
end

#display_device_attributes()

function main()
    n_nodes = 10000
    
    p = 0.5

    println("starting precompile run")
    compute_uniform_random_graph_PZER(n_nodes, p)

    println("precompile complete")



    display(CUDA.@profile compute_uniform_random_graph_PZER(n_nodes, p))
    @time compute_uniform_random_graph_PZER(n_nodes, p) 
    res = zeros(Int, n_nodes, n_nodes)
    trials = 1
    

    for i in 1:trials
        res .+= compute_uniform_random_graph_PZER(n_nodes, p)   
    end

    emp_mean_edges = sum(res)/trials


    anal_mean_edges = p*n_nodes*(n_nodes-1)

    edge_prob_mat = res./trials

    println("edge_prob_mat")
    println("max:   ", maximum(edge_prob_mat))
    println("mean: ", mean(edge_prob_mat))
    println("min:   ", minimum(edge_prob_mat))

    #display(res./trials)
    println("anal mean edges ", anal_mean_edges)
    println("emp mean edges ", emp_mean_edges)



end

main()  




