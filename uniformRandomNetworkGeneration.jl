using Distributions


module UniformRandomGraphSampling

export generate_uniform_random_graph_geometric, generate_uniform_random_graph
"""
Generates a G(n,p) random graph with the PreZER algorithm desribed in https://doi.org/10.1145/1951365.1951406.
Has some kind of bias for certain edges. Number of edges is slightly higher due to that bias.
"""
function generate_uniform_random_graph_PreZER(
    n_nodes::Int,
    edge_probability::Float64;
    flipped::Bool=false,
    cumu_dist_lookup_table_tol::Float64=0.99
)

    if flipped
        adj_matrix = trues(n_nodes, n_nodes)
    else
        adj_matrix = falses(n_nodes, n_nodes)
    end


    cumu_dist = [edge_probability]
    k = 2
    while cumu_dist[end] < cumu_dist_lookup_table_tol
        append!(cumu_dist, 1-(1-edge_probability)^(k))    
        k += 1
    end

    m = length(cumu_dist)
    row_index::Int = 1
    col_index::Int = 0

    #debugging variables

    edge_skip = 0 
    while row_index <= n_nodes
        α = rand()

        # Check for breakpoints in comulative dist funtion
        j = 1
        while j <= m
            if cumu_dist[j] > α
                edge_skip = j
                break
            end
            j += 1
        end
        # If number exceeds breakpoints, compute manually
        if j == m+1
            edge_skip = ceil(log(1-edge_probability, 1-α))-1
        end

        col_index += edge_skip
        
        # adjust column and row index to fit in square matrix again
        while col_index >= row_index && row_index <= n_nodes
            col_index -= row_index-1
            row_index += 1
        end
        

        if row_index <= n_nodes
            adj_matrix[row_index, col_index] = !flipped 
            adj_matrix[col_index, row_index] = !flipped 
        end


    end
    return adj_matrix
end


"""
    generate_uniform_random_graph_geometric(n_nodes, edge_probability, flipped=False)

Sequentially generates a G(n,p) random graph with the geometric algorithm as described in DOI: 10.1103/PhysRevE.71.036113

Set flipped=true to execute the algorithm for the inverse of the graph.

"""
function generate_uniform_random_graph_geometric(
    n_nodes::Int,
    edge_probability::Float64;
    flipped=false
)
    row_index:: Int = 1
    col_index:: Int = -1

    if flipped
        adj_matrix = trues(n_nodes, n_nodes)
    else
        adj_matrix = falses(n_nodes, n_nodes)
    end

    while row_index < n_nodes
        r = rand()
        col_index = col_index + 1 + floor(log(1 - r) / log(1 - edge_probability))
        while col_index >= row_index && row_index < n_nodes
            col_index -= row_index
            row_index += 1
        end
        if row_index < n_nodes
            adj_matrix[row_index+1, col_index+1] = !flipped 
            adj_matrix[col_index+1, row_index+1] = !flipped 

        end
    end
    return adj_matrix
end


function generate_uniform_random_graph(
    n_nodes::Int,
    edge_probability::Float64
)
    if edge_probability > 0.5
        return generate_uniform_random_graph_PreZER(n_nodes, 1-edge_probability, flipped=true)
    else
        return generate_uniform_random_graph_PreZER(n_nodes, edge_probability)
    end
end




function main2()


    p = 0.2
    n_nodes = 10
    trials = 2800000


end



function main()
    p = 0.2
    n_nodes = 10
    trials = 2800000
    n_t=28
    chunksize = trials ÷ n_t 
    results = zeros(Int, 28, n_nodes, n_nodes)
    generate_uniform_random_graph_PreZER(n_nodes, p)
    Threads.@threads for i in 1:n_t
        for j in 1:chunksize
            tmp = generate_uniform_random_graph_PreZER(n_nodes, p)        
            results[i,:,:] += tmp
        end

    end

    emp_mean_edges = sum(results)/trials
    emp_mean_mat = dropdims(sum(results, dims=1)/trials,dims=1)

    anal_mean_edges = p*n_nodes*(n_nodes-1)

    println("anal mean edges ", anal_mean_edges)
    println("emp mean edges ", emp_mean_edges)
    #display(emp_mean_mat-fill(p, (n_nodes, n_nodes)))
    display(emp_mean_mat)
    println(argmax(emp_mean_mat))
    println(maximum(emp_mean_mat))


end

end