import os
import pandas as pd
import json
from partitioning import (anonymize_with_partitioning)
from label_list import anonymize_with_label_lists
from utils import (
    load_movielens_as_graph, 
    run_query
)
from query import (
    query_pair_ratings,
    query_degree_distribution_test,
    query_structural_similarity
)
from privacy_utility_level import privacy_vs_utility_analysis
from statistical_analysis import analyze_genre_age_anonymization
from efficiency_analysis import efficiency_analysis
from algorithm_correctness import run_validation


# Global variables to store data and results
data_store = {
    'nodes': None,
    'interaction_graph': None,
    'movies_df': None,
    'll_full_mapping': None,
    'll_full_classes': None,
    'll_full_graph': None,
    'll_prefix_mapping': None,
    'll_prefix_classes': None,
    'll_prefix_graph': None,
    'partition_graph': None,
    'm_param': None,
    'k_param': None,
    'sort_order': ['num_ratings']
}

"""Wait for user input to continue."""
def wait_for_continue():
    input("\nPress Enter to continue...")
    print()  # Add blank line for spacing

"""CLI loop."""
def run_cli():
    user_input = 0
    while user_input != 99:
        print_menu()
        user_input = int(input("Enter command: "))
        execute_command(user_input)
        
        # Wait for user input before showing menu again (except for exit)
        if user_input != 99:
            wait_for_continue()

"""the CLI menu."""
def print_menu():
    print("DATA LOADING:")
    print(" 1  -> Load MovieLens dataset")
    print("")
    print("ANONYMIZATION:")
    print(" 2  -> Set anonymization parameters (m, k, sort order)")
    print(" 3  -> Run Label List anonymization (Full pattern)")
    print(" 4  -> Run Label List anonymization (Prefix pattern)")
    print(" 5  -> Run Partitioning anonymization")
    print(" 6  -> Run all anonymization methods")
    print(" 7  -> View anonymized graphs")
    print("")
    print("QUERIES:")
    print(" 8  -> Query: High-activity users rating specific genre")
    print(" 9  -> Query: Users with specific number of ratings (degree test)")
    print(" 10 -> Query: User pairs with shared movies (structural similarity)")
    print("")
    print("ANALYSIS:")
    print(" 11 -> Run privacy_vs_utility Query Analysis")
    print(" 12 -> Run statistical Analysis")
    print(" 13 -> Run efficiency Analysis")
    print(" 14 -> Run Algorithm Correctness Validation")
    print("")
    print("UTILITIES:")
    print(" 15 -> Export anonymized data")
    print("")
    print(" 99 -> Exit")
    print("="*60)

"""Execute the selected command"""
def execute_command(command):
    if requires_data(command) and not has_data():
        print("[ERROR] Please load a dataset first (command 1)")
        return
        
    if requires_anonymization(command) and not has_anonymization(command):
        print("[ERROR] Please run anonymization first (commands 4-7)")
        return

    if command == 1:
        load_movielens()
    elif command == 2:
        set_parameters()
    elif command == 3:
        run_label_list_full()
    elif command == 4:
        run_label_list_prefix()
    elif command == 5:
        run_partitioning()
    elif command == 6:
        run_all_anonymization()
    elif command == 7:
        show_anonymized_graph()
    elif command == 8:
        query_genre_ratings()
    elif command == 9:
        query_degree_test()
    elif command == 10:
        query_structural_similarity_test()
    elif command == 11:
        run_privacy_vs_utility_analysis()
    elif command == 12:
        run_statistical_analysis()
    elif command == 13:
        run_efficiency_analysis()
    elif command == 14:
        run_correctness_validation()
    elif command == 15:
        export_data()
    elif command == 99:
        print("Exit")
    else:
        print("[ERROR] Invalid command")

"""Check if command requires necessary data."""
def requires_data(command):
    return command in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

"""Check if command requires anonymization to be run"""
def requires_anonymization(command):
    return command in [7, 8, 9, 10, 15]

"""Check if necessary data are loaded."""
def has_data():
    return data_store['nodes'] is not None and data_store['interaction_graph'] is not None

"""Check if any anonymization has been run or if it will be running that do themselves anonymization (11-14)"""
def has_anonymization(command):
    return (data_store['ll_full_mapping'] is not None or
            data_store['ll_prefix_mapping'] is not None or 
            data_store['partition_graph'] is not None or
            command in [11, 12, 13, 14])

"""Load MovieLens dataset"""
def load_movielens():
    ratings_path = input("Enter path to ratings.csv (default ml-latest-small/ratings.csv): ").strip()
    if not ratings_path:
        ratings_path = 'ml-latest-small/ratings.csv'
    
    movies_path = input("Enter path to movies.csv (default ml-latest-small/movies.csv): ").strip()
    if not movies_path:
        movies_path = 'ml-latest-small/movies.csv'
    
    min_ratings = int(input(" Minimum ratings per user (default 5): ") or "5")

    data_store['nodes'], data_store['interaction_graph'] = load_movielens_as_graph(ratings_path, min_ratings)
    if os.path.exists(movies_path):
        data_store['movies_df'] = pd.read_csv(movies_path)
        print(f"Loaded movies metadata: {len(data_store['movies_df'])} movies")
    print(f" Dataset loaded: {len(data_store['nodes'])} users, {len(data_store['interaction_graph'])} user-movie connections")

# PARAMETER SETTING
"""Set anonymization parameters."""
def set_parameters():
    print("\n=== SET ANONYMIZATION PARAMETERS ===")
    data_store['m_param'] = int(input(f"Enter m (max class size, current: {data_store['m_param']}): ") or str(data_store['m_param'] or 30))
    data_store['k_param'] = int(input(f"Enter k (prefix size, current: {data_store['k_param']}): ") or str(data_store['k_param'] or 20))
    
    if data_store['k_param'] > data_store['m_param']:
        print("k should be <= m, setting k = m")
        data_store['k_param'] = data_store['m_param']
    
    print("Sort order (current:", data_store['sort_order'], ")")
    print("Available attributes: age, num_ratings, group")
    new_sort = input("Enter new sort order (comma-separated, or press Enter to keep current): ").strip()
    if new_sort:
        data_store['sort_order'] = [attr.strip() for attr in new_sort.split(',')]
    
    print(f"Parameters set: m={data_store['m_param']}, k={data_store['k_param']}, sort_order={data_store['sort_order']}")

# ANONYMIZATION COMMANDS
"""Run Label List anonymization with full pattern."""
def run_label_list_full():
    if data_store['m_param'] is None:
        set_parameters()
    
    print(f"\n Running Label List (Full) with m={data_store['m_param']}...")
    data_store['ll_full_mapping'], data_store['ll_full_classes'], data_store['ll_full_graph'] = anonymize_with_label_lists(
        data_store['nodes'], data_store['interaction_graph'], data_store['m_param'], data_store['m_param'], data_store['sort_order'], 'full'
    )
    print(f"Label List (Full) completed: {len(data_store['ll_full_classes'])} classes")

"""Run Label List anonymization with prefix pattern."""
def run_label_list_prefix():
    if data_store['m_param'] is None or data_store['k_param'] is None:
        set_parameters()
    
    print(f"\nRunning Label List (Prefix) with m={data_store['m_param']}, k={data_store['k_param']}...")
    data_store['ll_prefix_mapping'], data_store['ll_prefix_classes'], data_store['ll_prefix_graph'] = anonymize_with_label_lists(
        data_store['nodes'], data_store['interaction_graph'], data_store['m_param'], data_store['k_param'], data_store['sort_order'], 'prefix'
    )
    print(f"Label List (Prefix) completed: {len(data_store['ll_prefix_classes'])} classes")

"""Run Partitioning anonymization."""
def run_partitioning():
    if data_store['m_param'] is None:
        set_parameters()
    
    print(f"\nRunning Partitioning with m={data_store['m_param']}...")
    data_store['partition_graph'] = anonymize_with_partitioning(
        data_store['nodes'], data_store['interaction_graph'], data_store['m_param'], data_store['sort_order']
    )
    print(f"Partitioning completed: {len(data_store['partition_graph'])} interactions")

"""Run all anonymization methods."""
def run_all_anonymization():
    if data_store['m_param'] is None:
        set_parameters()
    
    print("\nRunning all anonymization methods...")
    run_label_list_full()
    run_label_list_prefix()
    run_partitioning()
    print("All anonymization methods completed")

"""Show the obtained anonymized graphs."""
def show_anonymized_graph():
    print("\n=== ANONYMIZED GRAPHS ===")
    
    if data_store['ll_full_graph']:
        print("\n--- Label List (Full) Anonymized Graph ---")
        print("Format: label_list -> {interactions}")
        for label_list, interactions in list(data_store['ll_full_graph'].items())[:10]:  # Show first 10
            print(f"{label_list} -> {interactions}")
    
    if data_store['ll_prefix_graph']:
        print("\n--- Label List (Prefix) Anonymized Graph ---")
        print("Format: label_list -> {interactions}")
        for label_list, interactions in list(data_store['ll_prefix_graph'].items())[:10]:  # Show first 10
            print(f"{label_list} -> {interactions}")
    
    if data_store['partition_graph']:
        print("\n--- Partitioning Anonymized Graph ---")
        print("Format: interaction -> {class_label: count}")
        for interaction, class_counts in list(data_store['partition_graph'].items())[:10]:  # Show first 10
            print(f"{interaction} -> {dict(class_counts)}")
    
    if not any([data_store['ll_full_graph'], data_store['ll_prefix_graph'], data_store['partition_graph']]):
        print("No anonymization graphs available. Run anonymization first (commands 4-7).")


# QUERY COMMANDS
"""Run genre-based ratings query."""
def query_genre_ratings():
    if data_store['movies_df'] is None or data_store['movies_df'].empty:
        print("[ERROR] Movies metadata not loaded")
        return
    
    min_ratings = int(input("[INPUT] Minimum ratings threshold (default 50): ") or "50")
    target_genre = input("[INPUT] Target genre (default 'Sci-Fi'): ") or "Sci-Fi"
    
    print(f"\n=== QUERY: Users with >{min_ratings} ratings who rated {target_genre} movies ===")
    
    query_params = {
        "nodes": data_store['nodes'],
        "movies_df": data_store['movies_df'],
        "min_ratings": min_ratings,
        "target_genre": target_genre
    }
    
    # Original
    original_result, _ = run_query(query_pair_ratings, data_store['interaction_graph'], 
                                 anon_type="original", **query_params)
    print(f"Original Graph: {original_result}")
    
    # Test anonymized versions
    if data_store['ll_full_mapping']:
        result, error = run_query(query_pair_ratings, data_store['interaction_graph'],
                                anon_type="label_list", anon_mapping=data_store['ll_full_mapping'],
                                pattern_type="full", classes=data_store['ll_full_classes'], **query_params)
        print(f"Label List (Full): {result:.1f} (Error: {error*100:.2f}%)")
    
    if data_store['ll_prefix_mapping']:
        result, error = run_query(query_pair_ratings, data_store['interaction_graph'],
                                anon_type="label_list", anon_mapping=data_store['ll_prefix_mapping'],
                                pattern_type="prefix", classes=data_store['ll_prefix_classes'], **query_params)
        print(f"Label List (Prefix): {result:.1f} (Error: {error*100:.2f}%)")
    
    # Partitioning
    if data_store['partition_graph']:
        result, error = run_query(query_pair_ratings, data_store['interaction_graph'],
                                anon_type="partition", anon_graph=data_store['partition_graph'],
                                **query_params)
        print(f"Partitioning: {result:.1f} (Error: {error*100:.2f}%)")

"""Run degree distribution test query."""
def query_degree_test():
    target_degree = input("[INPUT] Target number of ratings (default 45): ")
    target_degree = int(target_degree) if target_degree.isdigit() else 45
    
    print(f"\n=== QUERY: Users with exactly {target_degree} ratings ===")
    
    query_params = {
        "target_degree": target_degree
    }
    
    # Original
    original_result, _ = run_query(query_degree_distribution_test, data_store['interaction_graph'],
                                 anon_type="original", **query_params)
    print(f"Original Graph: {original_result}")
    
    # Label List Full
    if data_store['ll_full_mapping']:
        result, error = run_query(query_degree_distribution_test, data_store['interaction_graph'],
                                anon_type="label_list", anon_mapping=data_store['ll_full_mapping'],
                                pattern_type="full", classes=data_store['ll_full_classes'], **query_params)
        print(f"Label List (Full): {result:.1f} (Error: {error*100:.2f}%)")
    
    # Label List Prefix
    if data_store['ll_prefix_mapping']:
        result, error = run_query(query_degree_distribution_test, data_store['interaction_graph'],
                                anon_type="label_list", anon_mapping=data_store['ll_prefix_mapping'],
                                pattern_type="prefix", classes=data_store['ll_prefix_classes'], **query_params)
        print(f"Label List (Prefix): {result:.1f} (Error: {error*100:.2f}%)")
    
    # Partitioning
    if data_store['partition_graph']:
        result, error = run_query(query_degree_distribution_test, data_store['interaction_graph'],
                                anon_type="partition", anon_graph=data_store['partition_graph'],
                                **query_params)
        print(f"Partitioning: {result:.1f} (Error: {error*100:.2f}%)")

"""Run structural similarity query."""
def query_structural_similarity_test():
    min_shared_movies = input("[INPUT] Minimum shared movies (default 5): ")
    min_shared_movies = int(min_shared_movies) if min_shared_movies.isdigit() else 5
    
    print(f"\n=== QUERY: User pairs sharing at least {min_shared_movies} movies ===")
    
    query_params = {
        "min_shared_movies": min_shared_movies
    }
    
    # Original
    original_result, _ = run_query(query_structural_similarity, data_store['interaction_graph'],
                                 anon_type="original", **query_params)
    print(f"Original Graph: {original_result}")
    
    # Label List Full
    if data_store['ll_full_mapping']:
        result, error = run_query(query_structural_similarity, data_store['interaction_graph'],
                                anon_type="label_list", anon_mapping=data_store['ll_full_mapping'],
                                pattern_type="full", classes=data_store['ll_full_classes'], **query_params)
        print(f"Label List (Full): {result:.1f} (Error: {error*100:.2f}%)")
    
    # Label List Prefix
    if data_store['ll_prefix_mapping']:
        result, error = run_query(query_structural_similarity, data_store['interaction_graph'],
                                anon_type="label_list", anon_mapping=data_store['ll_prefix_mapping'],
                                pattern_type="prefix", classes=data_store['ll_prefix_classes'], **query_params)
        print(f"Label List (Prefix): {result:.1f} (Error: {error*100:.2f}%)")
    
    # Partitioning
    if data_store['partition_graph']:
        result, error = run_query(query_structural_similarity, data_store['interaction_graph'],
                                anon_type="partition", anon_graph=data_store['partition_graph'],
                                **query_params)
        print(f"Partitioning: {result:.1f} (Error: {error*100:.2f}%)")


# UTILITY COMMAND
"""Export anonymized graphs."""
def export_data():
    output_dir = input("[INPUT] Output directory (default ./output): ") or "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = []
    
    # Export Label List (Full) anonymized graph
    if data_store['ll_full_graph']:
        with open(f"{output_dir}/label_list_full_graph.json", 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable = {}
            for label_list, interactions in data_store['ll_full_graph'].items():
                serializable[str(label_list)] = list(interactions)
            json.dump(serializable, f, indent=2)
        exported_files.append("label_list_full_graph.json")
    
    # Export Label List (Prefix) anonymized graph
    if data_store['ll_prefix_graph']:
        with open(f"{output_dir}/label_list_prefix_graph.json", 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable = {}
            for label_list, interactions in data_store['ll_prefix_graph'].items():
                serializable[str(label_list)] = list(interactions)
            json.dump(serializable, f, indent=2)
        exported_files.append("label_list_prefix_graph.json")
    
    # Export Partitioning anonymized graph
    if data_store['partition_graph']:
        with open(f"{output_dir}/partition_graph.json", 'w') as f:
            # Convert tuple keys to strings for JSON serialization
            serializable = {}
            for interaction, class_counts in data_store['partition_graph'].items():
                serializable[interaction] = {str(k): v for k, v in class_counts.items()}
            json.dump(serializable, f, indent=2)
        exported_files.append("partition_graph.json")
    
    print(f"Exported {len(exported_files)} anonymized graphs to {output_dir}/")
    for f in exported_files:
        print(f"  - {f}")


# ANALYSIS COMMANDS

def run_privacy_vs_utility_analysis():
    print("\nRunning privacy_vs_utility analysis on reduced dataset...")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

    results = privacy_vs_utility_analysis(data_store['nodes'], data_store['interaction_graph'],
                                          data_store['movies_df'])

    print("Query analysis completed!")
    print("Results saved to reduced_utility_analysis/ directory")


def run_statistical_analysis():
    
    print("\nRunning genre-age analysis...")
    print("This analysis shows how average age per genre changes with anonymization")

    results = analyze_genre_age_anonymization(
        data_store['nodes'],
        data_store['interaction_graph'],
        data_store['movies_df']
    )

    print("Genre-age analysis completed!")
    print("Results saved to genre_age_analysis/ directory")


def run_efficiency_analysis():
    print("\nRunning efficiency analysis...")
    print("This analysis measures execution time of different anonymization methods")

    results = efficiency_analysis(
        data_store['nodes'],
        data_store['interaction_graph']
    )
    # Show quick summary
    if results:
        print("\nQuick Summary:")
        for method, method_results in results.items():
            if method_results:
                avg_time = sum(data['execution_time'] for data in method_results.values()) / len(method_results)
                print(f"  {method}: Average {avg_time:.4f}s per run")

def run_correctness_validation():
    
    print("\nRunning algorithm correctness validation...")
    print("This validates that all anonymization algorithms work correctly")

    results = run_validation()

    # Count passed tests
    total_tests = 0
    passed_tests = 0

    for algorithm, result_dict in results.items():
        total_tests += 1
        if result_dict.get('overall', False):
            passed_tests += 1

    print(f"{passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("All tests passed")
    else:
        print(" Some tests failed")


if __name__ == "__main__":
    run_cli()