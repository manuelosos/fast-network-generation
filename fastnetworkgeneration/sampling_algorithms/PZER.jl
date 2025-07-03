using CUDA
using Statistics


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
        execution_times  += time() - start_time
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


function compute_uniform_random_graph_PZER(n_nodes, edge_probability, chunksize; verbose=true)
    

    buffer = Channel{Vector{Float64}}(5)

    adj_mat = falses(n_nodes, n_nodes)
    start_time = time()
    local gpu_task, cpu_task
    begin 
        gpu_task = Base.Threads.@spawn gpu_compute_loop(buffer, n_nodes, edge_probability, chunksize)
        cpu_task = Base.Threads.@spawn cpu_compute_loop!(adj_mat, buffer, n_nodes)
    end
    total_execution_time = time() - start_time
    gpu_counts, gpu_exec_time = fetch(gpu_task)
    cpu_counts, cpu_exec_time = fetch(cpu_task)

    @info "GPU and CPU loop finished"

    average_gpu_exec_time = gpu_exec_time/gpu_counts
    average_cpu_exec_time = cpu_exec_time/cpu_counts
    
    if verbose
        println("Total execution time: $(total_execution_time/1_000_000)")
        println("N nodes: $(n_nodes)")
        println("Edge Probability $(edge_probability)")
        println("Chunksize: $(chunksize)")
        println("Total GPU execution time: $(gpu_exec_time/1_000_000)")
        println("Total CPU execution time: $(cpu_exec_time/1_000_000)")
        println("GPU executions: $(gpu_counts), average GPU execution time $(average_gpu_exec_time/1_000_000)")
        println("CPU executions: $(cpu_counts), average CPU execution time $(average_cpu_exec_time/1_000_000)")
        println("Factor: GPU/CPU $(average_gpu_exec_time/average_cpu_exec_time)")
    end

    return adj_mat
end









