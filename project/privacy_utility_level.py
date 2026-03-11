import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from partitioning import anonymize_with_partitioning
from label_list import anonymize_with_label_lists
from utils import (
    randomly_delete_half_interactions,
    run_query,
    load_movielens_as_graph
)
from query import (
    query_pair_ratings,
    query_degree_distribution_test,
    query_structural_similarity
)

"""
    Utility analysis on the movies dataset
"""
def privacy_vs_utility_analysis(nodes, interaction_graph, movies_df, output_dir="privacy_utility_analysis"):
    # Create output directory if not exist
    os.makedirs(output_dir, exist_ok=True)
    
    #  Randomly delete half of the interactions of the original dataset,
    # otherwise with such a big number of interactions (> 100000) and
    # relatively small number of users (610) the classes will have few components because it would
    #be challenging to satisfy the safety condition and this would alter the analysis
    print("Reducing dataset size")
    reduced_graph = randomly_delete_half_interactions(interaction_graph)
    
    # Update nodes to reflect new interaction counts
    updated_nodes = []
    for node in nodes:
        user_id = node['id']
        if user_id in reduced_graph:
            updated_node = node.copy()
            updated_node['num_ratings'] = len(reduced_graph[user_id])
            updated_nodes.append(updated_node)

    # Parameter combinations to test
    m_values = [5, 10, 15, 20]
    k_ratios = [0.7, 0.9]  # k as ratio of m
    sort_orders = [['num_ratings'], ['age']]
    
    # Query configurations
    queries_config = [
        {
            'name': 'Genre_Documentary_Min40',
            'function': query_pair_ratings,
            'params': {'min_ratings': 40, 'target_genre': 'Documentary', 'movies_df': movies_df, 'nodes': updated_nodes},
            'description': 'High-activity users rating Documentary movies'
        },
        {
            'name': 'Degree_Distribution_25',
            'function': query_degree_distribution_test,
            'params': {'target_degree': 25},
            'description': 'Users with exactly 25 ratings'
        },
        {
            'name': 'Structural_Similarity_5',
            'function': query_structural_similarity,
            'params': {'min_shared_movies': 5},
            'description': 'User pairs sharing 5+ movies'
        }
    ]
    
    results = {}

    for m in m_values:
        for k_ratio in k_ratios:
            k = max(1, int(m * k_ratio))
            for sort_order in sort_orders:
                param_key = f"m{m}_k{k}_sort{''.join(sort_order)}"

                # Run anonymization methods

                # Label List Full Pattern
                ll_full_mapping, ll_full_classes, ll_full_graph = anonymize_with_label_lists(
                    updated_nodes, reduced_graph, m=m, k=m, pattern_type='full', sort_attributes=sort_order)

                # Label List Prefix Pattern
                ll_prefix_mapping, ll_prefix_classes, ll_prefix_graph = anonymize_with_label_lists(
                    updated_nodes, reduced_graph, m=m, k=k, pattern_type='prefix', sort_attributes=sort_order)

                # Partitioning
                partition_graph = anonymize_with_partitioning(
                    updated_nodes, reduced_graph, m=m, sort_attributes=sort_order)

                # Test all queries for this parameter combination
                param_results = {}
                
                for query_config in queries_config:
                    query_name = query_config['name']
                    query_func = query_config['function']
                    query_params = query_config['params'].copy()
                    
                    print(f"  Testing query: {query_name}")
                    
                    # Original query result
                    original_result, _ = run_query(query_func, reduced_graph, anon_type="original", **query_params)

                    # Test anonymized versions
                    methods = [
                        ("Label_List_Full", "label_list", ll_full_mapping, "full", ll_full_classes),
                        ("Label_List_Prefix", "label_list", ll_prefix_mapping, "prefix", ll_prefix_classes),
                        ("Partitioning", "partition", partition_graph, None, None)
                    ]
                    
                    query_results = {'original': original_result}
                    
                    for method_name, anon_type, mapping_or_graph, pattern_type, classes_or_attrs in methods:
                        if anon_type == "label_list":
                            result, error = run_query(query_func, reduced_graph,
                                                    anon_type=anon_type, anon_mapping=mapping_or_graph,
                                                    pattern_type=pattern_type, classes=classes_or_attrs, **query_params)
                        else:  # partition
                            result, error = run_query(query_func, reduced_graph,
                                                    anon_type=anon_type, anon_graph=mapping_or_graph,
                                                    classes=classes_or_attrs, **query_params)

                        query_results[method_name] = {'result': result, 'error': error}
                        #print(f"    {method_name}: {result:.2f} (Error: {error*100:.2f}%)")

                    param_results[query_name] = query_results
                    print(param_results)
                
                results[param_key] = param_results
    
    # summary plots
    print(f" GENERATING PLOTS")

    methods = ["Label_List_Full", "Label_List_Prefix", "Partitioning"]
    method_colors = ['blue', 'green', 'red']
    
    # separate plots for each sort order
    for sort_order in sort_orders:
        sort_name = ''.join(sort_order)
        
        # subplots for each query
        fig, axes = plt.subplots(1, len(queries_config), figsize=(5*len(queries_config), 6))
        if len(queries_config) == 1:
            axes = [axes]
        
        fig.suptitle(f'Error Rates (Sort Order: {sort_name})', fontsize=16)

        for query_idx, query_config in enumerate(queries_config):
            ax = axes[query_idx]
            query_name = query_config['name']
            
            # Plot each method
            for method_idx, method in enumerate(methods):
                m_vals = []
                errors = []
                
                # Collect data for this specific sort order
                for param_key, param_results in results.items():
                    # Check if this param_key matches the current sort order
                    if f"sort{sort_name}" in param_key and query_name in param_results:
                        m_val = int(param_key.split('_')[0][1:])  # Extract m from "m10_k7_sort..."
                        if method in param_results[query_name] and 'error' in param_results[query_name][method]:
                            m_vals.append(m_val)
                            errors.append(param_results[query_name][method]['error'] * 100)
                
                # Group by m and average errors across k values
                m_groups = {}
                for m_val, error in zip(m_vals, errors):
                    if m_val not in m_groups:
                        m_groups[m_val] = []
                    m_groups[m_val].append(error)
                
                if m_groups:  # Only plot if we have data
                    m_sorted = sorted(m_groups.keys())
                    avg_errors = [np.mean(m_groups[m]) for m in m_sorted]
                    
                    # Plot with error bars
                    ax.errorbar(m_sorted, avg_errors,
                              color=method_colors[method_idx],
                              label=method, linewidth=2, markersize=8)
            
            ax.set_xlabel('m (max class size)')
            ax.set_ylabel('Error Rate (%)')
            ax.set_title(f'{query_config["description"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set reasonable y-axis limits
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plot_filename = f'error_rates_sort_{sort_name}.png'
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot: {plot_filename}")
    
    # Compare sort orders for each query plot
    fig, axes = plt.subplots(1, len(queries_config), figsize=(5*len(queries_config), 6))
    if len(queries_config) == 1:
        axes = [axes]
    
    fig.suptitle('Sort Order Comparison', fontsize=16)
    
    for query_idx, query_config in enumerate(queries_config):
        ax = axes[query_idx]
        query_name = query_config['name']
        
        # For each sort order, calculate average error across all methods and m values
        sort_errors = {}
        for sort_order in sort_orders:
            sort_name = ''.join(sort_order)
            all_errors = []
            
            for param_key, param_results in results.items():
                if f"sort{sort_name}" in param_key and query_name in param_results:
                    for method in methods:
                        if method in param_results[query_name] and 'error' in param_results[query_name][method]:
                            all_errors.append(param_results[query_name][method]['error'] * 100)
            
            if all_errors:
                sort_errors[sort_name] = {
                    'mean': np.mean(all_errors),
                    'std': np.std(all_errors)
                }
        
        if sort_errors:
            sort_names = list(sort_errors.keys())
            means = [sort_errors[s]['mean'] for s in sort_names]
            
            ax.bar(sort_names, means, alpha=0.7, capsize=5)
            ax.set_ylabel('Average Error Rate (%)')
            ax.set_title(f'{query_config["description"]}')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    sort_comparison_filename = f'sort_order_comparison.png'
    plt.savefig(os.path.join(output_dir, sort_comparison_filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {sort_comparison_filename}")

    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Plots saved to: {output_dir}")
    

if __name__ == "__main__":
    # Load MovieLens dataset
    print("LOADING MOVIELENS DATASET")
    ratings_path = 'ml-latest-small/ratings.csv'
    movies_path = 'ml-latest-small/movies.csv'

    user_nodes, interaction_graph = load_movielens_as_graph(ratings_path, min_ratings_per_user=5)
    movies_df = pd.read_csv(movies_path)

    # Run utility analysis
    print("Starting utility analysis...")

    privacy_vs_utility_analysis(user_nodes, interaction_graph, movies_df)
    print("analysis completed!")


