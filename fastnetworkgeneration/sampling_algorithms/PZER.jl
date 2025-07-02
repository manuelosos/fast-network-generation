using CUDA
#using BenchmarkTools
using Statistics
#using Base.Threads


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
    n_nodes::Integer,
    edge_probabiltiy::Real,
    chunk_size::Integer
)
    @info "Starting GPU loop"
    current_adj_mat_index = 0
    execution_times = 0
    execution_counts = 0
    while true

        start_time = time()
        edge_list_d = compute_edge_skips_gpu(chunk_size, edge_probabiltiy, current_adj_mat_index)
        
        edge_list_h = Array(edge_list_d)

        current_adj_mat_index = convert(Int, edge_list_h[end])


        put!(buffer, edge_list_h)
        
        @info "GPU execution time for edge list $((time()-start_time)/1_000_000)"
        execution_times  += time()-start_time
        execution_counts += 1
        if edge_list_h[end] > n_nodes^2/2
            @debug "Finishing GPU loop"
            close(buffer)
            return execution_counts, execution_times
            break
        end
    end
    return execution_counts, execution_times
end


function cpu_compute_loop!(
    adj_mat::BitMatrix,
    buffer::Channel,
    n_nodes::Int;
    compute_inverse::Bool=false,
)
    @info "Starting CPU loop"

    row_index::Int = 1
    row_index_bound = 1
    col_index_bound = 0

    execution_times = 0
    execution_counts = 0


    for edge_list in buffer
        start_time = time()
        for edge_index in edge_list
            while row_index_bound < edge_index 
                row_index +=1
                col_index_bound = row_index_bound 
                row_index_bound = (row_index^2 + row_index)/2
            end
           
            col_index::Int = edge_index - col_index_bound

            if row_index < n_nodes && col_index <= n_nodes
                @inbounds adj_mat[row_index+1, col_index] = !compute_inverse
                @inbounds adj_mat[col_index, row_index+1] = !compute_inverse
            else
               break 
            end
        end

        execution_counts += 1
        execution_times += time() - start_time
        @info "CPU execution time for edge list: $((time()-start_time)/1_000_000)"


        if row_index >= n_nodes
            break
        end
    end
    @info "Finishing CPU loop"
    return execution_counts, execution_times
end


function compute_uniform_random_graph_PZER(n_nodes, edge_probability)
    

    chunksize = 1024*4
    buffer = Channel{Vector{Float64}}(5)

    adj_mat = falses(n_nodes, n_nodes)

    local gpu_task, cpu_task
    begin 
        gpu_task = Base.Threads.@spawn gpu_compute_loop(buffer, n_nodes, edge_probability, chunksize)
        cpu_task = Base.Threads.@spawn cpu_compute_loop!(adj_mat, buffer, n_nodes)
    end

    gpu_counts, gpu_exec_time = fetch(gpu_task)
    cpu_counts, cpu_exec_time = fetch(cpu_task)

    @info "GPU and CPU loop finished"
    println("GPU executions: $(gpu_count), average GPU execution time $(gpu_exec_time/gpu_counts/1_000_000)")
    println("CPU executions: $(cpu_count), average CPU execution time $(cpu_exec_time/cpu_counts/1_000_000)")

    return adj_mat
end

#display_device_attributes()

function main()
    n_nodes = 50000
    
    p = 0.5

    println("starting precompile run")
    compute_uniform_random_graph_PZER(5, p)

    println("precompile complete")


    compute_uniform_random_graph_PZER(n_nodes, p)

    return
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






