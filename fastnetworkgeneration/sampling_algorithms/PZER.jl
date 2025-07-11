using CUDA
using Statistics


function compute_single_edge_skip(edge_probability::Float64)::Int
    return max(1, ceil(log(1-edge_probability, rand(Float64))))
end


function linear_to_uppertriagonal_index(n::Int, k::Int, max_edge_index::Int)
    k = min(k, max_edge_index-1) # Avoids Domainerrors caused by negative values under the squareroor
    i::Int = n - 2 - floor(Int, sqrt(-8*k + 4*n*(n-1)-7)/2 - 0.5)
    j::Int = k + i + 1 - (n*(n-1) ÷ 2) +(n-i)*((n-i) - 1) ÷ 2  
    return (i, j)
end


function compute_edge_coordinates_gpu(
    n_skips::Int,
    n_nodes::Int,
    edge_probability::Real,
    start_index::Int
)
    skip_values = CuArray{Float64}(undef, n_skips)
    rng_gen(_) = compute_single_edge_skip(edge_probability)
    skip_values = CUDA.map(rng_gen, skip_values)   
    skip_values = CUDA.accumulate(+, skip_values)

    skip_values .+= start_index 

    lin2utri(k) = linear_to_uppertriagonal_index(n_nodes, k, convert(Int,n_nodes*(n_nodes-1)/2))    
    edge_coordinates = CUDA.map(lin2utri, skip_values)    

    
    return edge_coordinates 
end


function gpu_edge_coordinates_loop(
    buffer::Channel,
    n_nodes::Integer,
    edge_probability::Real,
    chunk_size::Integer
)

    @debug "Starting GPU loop"
    last_edge_index = 0
    execution_times = 0
    execution_counts = 0

    max_edges = n_nodes*(n_nodes-1)/2

    while true

        start_time = time()
        edge_coordinates_d = compute_edge_coordinates_gpu(chunk_size, n_nodes, edge_probability, last_edge_index)        
        edge_coordinates_h = Array(edge_coordinates_d)
        @debug "GPU creation time for edge coordinate list $((time()-start_time)/1_000_000)"

        memory_start_time = time()
        put!(buffer, edge_coordinates_h)
        @debug "GPU execution time for memory swapping $((time()-memory_start_time)/1_000_000)"
        

        execution_times  += time() - start_time
        execution_counts += 1


        last_edge_index = convert(Int, edge_coordinates_h[end][1] * n_nodes + edge_coordinates_h[end][2])
        if last_edge_index > max_edges 
            close(buffer)
            return execution_counts, execution_times
            break
        end

    end

    return execution_counts, execution_times

end


function cpu_edge_coordinates_loop!(
    network::SimpleNetwork,
    buffer::Channel,
    n_nodes::Int;
)
    @debug "Starting CPU loop"

    max_n_edges = n_nodes * (n_nodes-1) / 2
    
    execution_times = 0
    execution_counts = 0

    for edge_list in buffer
        start_time = time()
        for edge_coordinate in edge_list

            # break condition
            if edge_coordinate[2] > n_nodes
                break
            end

            insert_edge!(network, (edge_coordinate[1], edge_coordinate[2]))
        end

        execution_counts += 1
        execution_times += time() - start_time

        @debug "CPU decoding time for edge list: $((time()-start_time)/1_000_000)"

        if row_index >= n_nodes
            break
        end

    end
    @debug "Finishing CPU loop"
    return execution_counts, execution_times
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
    @debug "Starting GPU loop"
    current_adj_mat_index = 0
    execution_times = 0
    execution_counts = 0

    max_edges = n_nodes*(n_nodes-1)/2

    while true

        start_time = time()
        edge_list_d = compute_edge_skips_gpu(chunk_size, edge_probabiltiy, current_adj_mat_index)
        
        edge_list_h = Array(edge_list_d)

        current_adj_mat_index = convert(Int, edge_list_h[end])

        @debug "GPU creation time for edge list $((time()-start_time)/1_000_000)"
        
        memory_start_time = time()
        put!(buffer, edge_list_h)
        @debug "GPU execution time for memory swapping $((time()-memory_start_time)/1_000_000)"
        
        execution_times  += time() - start_time
        execution_counts += 1
        if edge_list_h[end] > max_edges 
            close(buffer)
            return execution_counts, execution_times
            break
        end
    end

    return execution_counts, execution_times
end


function cpu_compute_loop!(
    network::SimpleNetwork,
    buffer::Channel,
    n_nodes::Int;
    compute_inverse::Bool=false,
)
    @debug "Starting CPU loop"

    row_index::Int = 1
    row_index_bound = 1
    col_index_bound = 0

    max_n_edges = n_nodes * (n_nodes-1) / 2
    
    execution_times = 0
    execution_counts = 0


    for edge_list in buffer
        start_time = time()
        for edge_index in edge_list

            # break condition
            if edge_index > max_n_edges
                break
            end

            while row_index_bound < edge_index 
                row_index += 1
                col_index_bound = row_index_bound 
                row_index_bound = (row_index^2 + row_index)/2
            end
           
            col_index::Int = edge_index - col_index_bound
            
            insert_edge!(network, (row_index+1, col_index))
            
        end

        execution_counts += 1
        execution_times += time() - start_time
        @debug "CPU decoding time for edge list: $((time()-start_time)/1_000_000)"

        if row_index >= n_nodes
            break
        end
    end
    @debug "Finishing CPU loop"
    return execution_counts, execution_times
end


function sample_uniform_random_graph_PZER!(
    network::SimpleNetwork,
    n_nodes::Integer, 
    edge_probability::Real;
    gpu_chunksize::Integer=1024, 
    buffersize::Integer=true,
    verbose=true
    )
    
    #buffer = Channel{Vector{Integer}}(buffersize)
    buffer = Channel{Vector{Tuple{Integer, Integer}}}(buffersize)

    start_time = time()
    local gpu_task, cpu_task
    begin 
        #gpu_task = Base.Threads.@spawn gpu_compute_loop(buffer, n_nodes, edge_probability, gpu_chunksize)
        #cpu_task = Base.Threads.@spawn cpu_compute_loop!(network, buffer, n_nodes)
        gpu_task = Base.Threads.@spawn gpu_edge_coordinates_loop(buffer, n_nodes, edge_probability, gpu_chunksize)
        cpu_task = Base.Threads.@spawn cpu_edge_coordinates_loop!(network, buffer, n_nodes)
    end
    total_execution_time = time() - start_time
    gpu_counts, gpu_exec_time = fetch(gpu_task)
    cpu_counts, cpu_exec_time = fetch(cpu_task)

    if verbose
        @debug "GPU and CPU loop finished"
    end

    average_gpu_exec_time = gpu_exec_time/gpu_counts
    average_cpu_exec_time = cpu_exec_time/cpu_counts
    
    if verbose
        println("Total execution time: $(total_execution_time/1_000_000) seconds")
        println("Network dtype: $(typeof(network))")
        println("N nodes: $(n_nodes)")
        println("Edge Probability $(edge_probability)")
        println("Chunksize: $(gpu_chunksize)")
        println("Total GPU execution time: $(gpu_exec_time/1_000_000) seconds")
        println("Total CPU execution time: $(cpu_exec_time/1_000_000) seconds")
        println("GPU executions: $(gpu_counts), average GPU execution time $(average_gpu_exec_time/1_000_000)")
        println("CPU executions: $(cpu_counts), average CPU execution time $(average_cpu_exec_time/1_000_000)")
        println("Factor: GPU/CPU $(average_gpu_exec_time/average_cpu_exec_time)")
        println("")
    end

end


function generate_uniform_random_graph_PZER(
    n_nodes::Integer, 
    edge_probability::Real;
    network_dtype::typeof(SimpleNetwork)=infer_space_optimal_network_data_type(n_nodes, round(UInt64, edge_probability*n_nodes*(n_nodes-1)/2)),
    gpu_chunksize::Integer=1024, 
    buffersize::Integer=true,
    verbose=true
    )
    
    if network_dtype == AdjacencyMatrix
        network = AdjacencyMatrix(falses((n_nodes, n_nodes)))
    elseif network_dtype == NeighborList
        network = NeighborList(n_nodes)
    else
        throw(Error("Unknown network dtype"))
    end
    
    sample_uniform_random_graph_PZER!(
        network,
        n_nodes, 
        edge_probability;  
        gpu_chunksize=gpu_chunksize,
        verbose=verbose
        )

        return network
end




