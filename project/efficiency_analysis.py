import time
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from partitioning import anonymize_with_partitioning 
from label_list import anonymize_with_label_lists
from utils import load_movielens_as_graph

"""
    Measure the execution time of a function
    Args:
        func: Function to measure
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
    Returns:
        (result, execution_time_seconds)
"""
def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


"""
    Analyze the execution time of the different anonymization techniques
"""
def efficiency_analysis(nodes, interaction_graph, output_dir="efficiency_analysis"):
    # Create output directory, if not exist
    os.makedirs(output_dir, exist_ok=True)

    print("ANONYMIZATION EFFICIENCY ANALYSIS")
    
    # Parameters to test
    m_values = [5, 10, 15, 20, 25, 30]
    sort_orders = [['age']]
    
    # Store results
    efficiency_results = {
        'label_list_full': {},
        'label_list_prefix': {},
        'partitioning': {}
    }
    
    # Test each combination
    for m in m_values:
        for sort_order in sort_orders:
            sort_name = '_'.join(sort_order)
            
            print(f"Testing m={m}, sort={sort_name}")

            # Label List anonymization (full pattern)

            (ll_mapping, ll_classes, ll_graph), ll_time = measure_execution_time(
                anonymize_with_label_lists,
                nodes, interaction_graph,
                m=m, k=m, pattern_type='full', sort_attributes=sort_order
            )

            key = f"m{m}_sort{sort_name}"
            efficiency_results['label_list_full'][key] = {
                'execution_time': ll_time
                #'num_classes': len(ll_classes),
                #'m_value': m,
                #'sort_order': sort_name
            }

            print(f"  Label List Full: {ll_time:.4f}s")

            # Label List anonymization (prefix pattern)

            print(f"Label List Prefix (m={m})")

            (ll_prefix_mapping, ll_prefix_classes, ll_prefix_graph), ll_prefix_time = measure_execution_time(
                anonymize_with_label_lists,
                nodes, interaction_graph,
                m=m, k=int(m*0.7), pattern_type='prefix', sort_attributes=sort_order
            )

            efficiency_results['label_list_prefix'][key] = {
                'execution_time': ll_prefix_time
                #'num_classes': len(ll_prefix_classes),
                #'m_value': m,
                #'sort_order': sort_name
            }

            print(f"  Label List Prefix: {ll_prefix_time:.4f}s")

            # Partitioning anonymization
            print(f"Partitioning (m={m})")

            part_graph, part_time = measure_execution_time(
                anonymize_with_partitioning,
                nodes, interaction_graph,
                m=m, sort_attributes=sort_order
            )

            efficiency_results['partitioning'][key] = {
                'execution_time': part_time,
                #'num_interactions': len(part_graph),
                #'m_value': m,
                #'sort_order': sort_name
            }

            print(f"  Partitioning: {part_time:.4f}s")
    
    # Generate efficiency plots

    # Create execution time comparison plots
    methods = ['label_list_full', 'label_list_prefix', 'partitioning']
    method_colors = {
        'label_list_full': 'blue', 
        'label_list_prefix': 'green',
        'partitioning': 'red'
    }
    method_names = {
        'label_list_full': 'Label List (Full)', 
        'label_list_prefix': 'Label List (Prefix)',
        'partitioning': 'Partitioning'
    }

    
    # Create summary bar chart for average execution times
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Average Execution Time by Method', fontsize=16)
    
    method_avg_times = []
    method_labels = []
    
    for method in methods:
        all_times = []
        for key, data in efficiency_results[method].items():
            all_times.append(data['execution_time'])
        
        if all_times:
            avg_time = np.mean(all_times)
            method_avg_times.append(avg_time)
            method_labels.append(method_names[method])
    
    bars = ax.bar(method_labels, method_avg_times, 
                 color=[method_colors[method] for method in methods if method_names[method] in method_labels])
    
    # Add value labels on bars
    for bar, value in zip(bars, method_avg_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(method_avg_times)*0.01,
               f'{value:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Average Execution Time (seconds)')
    ax.set_title('Overall Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    summary_filename = f'efficiency_summary.png'
    plt.savefig(os.path.join(output_dir, summary_filename), dpi=300, bbox_inches='tight')
    plt.show()

    return efficiency_results


if __name__ == "__main__":
    # Load MovieLens dataset
    ratings_path = 'ml-latest-small/ratings.csv'

    user_nodes, interaction_graph = load_movielens_as_graph(ratings_path)

    results = efficiency_analysis(user_nodes, interaction_graph)
        
