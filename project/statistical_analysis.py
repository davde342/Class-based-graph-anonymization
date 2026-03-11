import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from partitioning import anonymize_with_partitioning
from label_list import anonymize_with_label_lists
from utils import (
    load_movielens_as_graph,
    reconstruct_from_partition,
    reconstruct_from_label_list,
    randomly_delete_half_interactions
)

"""
    Compute genre-age statistics using reconstructed graphs from anonymization methods.
    Args:
        nodes: List of user nodes with attributes
        interaction_graph: Original interaction graph
        movies_df: DataFrame with movie information
        anon_type: "original", "label_list", or "partition"
        num_samples: Number of reconstruction samples for anonymized methods
        **kwargs: Additional parameters for reconstruction

    Returns:
        dict: Genre -> {'avg_age': float, 'user_count': int, 'total_ratings': int}
"""


def compute_genre_age_statistics_from_reconstruction(nodes, interaction_graph, movies_df, anon_type="original",
                                                     num_samples=5, **kwargs):
    if anon_type == "original":
        # Use original data directly
        return compute_genre_age_statistics(nodes, interaction_graph, movies_df)

    # For anonymized methods, sample multiple reconstructions and average
    all_genre_stats = []

    for sample_idx in range(num_samples):
        if anon_type == "label_list":
            anon_mapping = kwargs.get('anon_mapping')
            classes = kwargs.get('classes')
            pattern_type = kwargs.get('pattern_type', 'full')
            reconstructed_graph = reconstruct_from_label_list(anon_mapping, classes, interaction_graph, pattern_type)
            sample_stats = compute_genre_age_statistics(nodes, reconstructed_graph, movies_df)

        elif anon_type == "partition":
            partition_graph = kwargs.get('partition_graph')
            # Reconstruct interaction graph from partition
            reconstructed_graph = reconstruct_from_partition(partition_graph)
            # Use original nodes
            sample_stats = compute_genre_age_statistics(nodes, reconstructed_graph, movies_df)

        all_genre_stats.append(sample_stats)

    # Get all genres that appear in any sample
    all_genres = set()
    for stats in all_genre_stats:
        all_genres.update(stats.keys())

    # Average statistics for each genre
    averaged_stats = {}
    for genre in all_genres:
        ages = []
        user_counts = []
        total_ratings = []

        for stats in all_genre_stats:
            if genre in stats:
                ages.append(stats[genre]['avg_age'])
                user_counts.append(stats[genre]['user_count'])
                total_ratings.append(stats[genre]['total_ratings'])

        if ages:
            averaged_stats[genre] = {
                'avg_age': np.mean(ages),
                'user_count': int(np.mean(user_counts)),
                'total_ratings': int(np.mean(total_ratings))
            }

    return averaged_stats


"""
    Compute average age of users who reviewed each genre.

    Args:
        nodes: List of user nodes with attributes
        interaction_graph: Dict mapping user_id -> set of movie_ids
        movies_df: DataFrame with movie information

    Returns:
        dict: Genre -> {'avg_age': float, 'user_count': int, 'total_ratings': int}
"""


def compute_genre_age_statistics(nodes, interaction_graph, movies_df):
    user_lookup = {node['id']: node for node in nodes}

    # Create movie to genres mapping
    movie_genres = {}
    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        genres = [g.strip() for g in row['genres'].split('|')]
        # Store both integer and string versions to handle both formats
        # movie_genres[movie_id] = [g for g in genres if g != '(no genres listed)']
        movie_genres[f"m{movie_id}"] = [g for g in genres if g != '(no genres listed)']

    # For each genre, collect user ages who rated movies of that genre
    genre_stats = {}
    movies_found = 0

    for user_id, rated_movies in interaction_graph.items():
        user_age = user_lookup[user_id].get('age', 0)
        # Find all genres this user has rated
        user_genres = set()
        user_movies_found = 0
        for movie_id in rated_movies:
            if movie_id in movie_genres:
                user_movies_found += 1
                movies_found += 1
                user_genres.update(movie_genres[movie_id])

        # Add this user's age to each genre they've rated
        for genre in user_genres:
            if genre not in genre_stats:
                genre_stats[genre] = {'ages': [], 'user_count': 0, 'total_ratings': 0}

            genre_stats[genre]['ages'].append(user_age)
            # Count how many movies of this genre this user rated
            genre_ratings = sum(1 for movie_id in rated_movies
                                if movie_id in movie_genres and genre in movie_genres[movie_id])
            genre_stats[genre]['total_ratings'] += genre_ratings

    # Compute averages
    result = {}
    for genre, data in genre_stats.items():
        result[genre] = {
            'avg_age': np.mean(data['ages']),
            'user_count': len(data['ages']),  # Count of age entries (users who rated this genre)
            'total_ratings': data['total_ratings']
        }

    return result


"""
   Function to analyze how statistical distribution of average
   age per genre changes with different anonymization methods and m values.

   Args:
       nodes: List of user nodes with attributes
       interaction_graph: Original interaction graph (reduced dataset)
       movies_df: Movies dataframe
       output_dir: Directory to save results
"""


def analyze_genre_age_anonymization(nodes, interaction_graph, movies_df, output_dir="statistical_analysis"):
    # Create output directory, if not exist
    os.makedirs(output_dir, exist_ok=True)
    print("Genre-Age analysis")
    reduced_graph = randomly_delete_half_interactions(interaction_graph)

    # Update nodes to reflect new interaction counts
    updated_nodes = []
    for node in nodes:
        user_id = node['id']
        if user_id in reduced_graph:
            updated_node = node.copy()
            updated_node['num_ratings'] = len(reduced_graph[user_id])
            updated_nodes.append(updated_node)

    # Parameters to test - multiple m values
    m_values = [5, 10, 15, 20, 25, 30]
    sort_orders = [['age']]

    # Compute original genre-age statistics
    print("Computing original statistical distribution...")
    original_stats = compute_genre_age_statistics_from_reconstruction(
        updated_nodes, reduced_graph, movies_df, anon_type="original"
    )

    analysis_results = {
        'original': original_stats,
        'anonymized': {}
    }

    # Run anonymization for each m value
    print("Running anonymization for multiple m values...")
    
    for m_val in m_values:
        print(f"Processing m={m_val}...")
        sort_name = '_'.join(sort_orders[0])
        
        # Label List anonymization (full pattern)
        ll_mapping, ll_classes, ll_graph = anonymize_with_label_lists(
            updated_nodes, reduced_graph, m=m_val, k=m_val, pattern_type='full', sort_attributes=sort_orders[0])

        ll_genre_stats = compute_genre_age_statistics_from_reconstruction(
            updated_nodes, reduced_graph, movies_df,
            anon_type="label_list",
            anon_mapping=ll_mapping,
            classes=ll_classes,
            pattern_type='full',
            num_samples=5
        )

        # Label List anonymization (prefix pattern)
        k_val = max(1, int(m_val * 0.7))  # k is 70% of m
        ll_prefix_mapping, ll_prefix_classes, ll_prefix_graph = anonymize_with_label_lists(
            updated_nodes, reduced_graph, m=m_val, k=k_val, pattern_type='prefix',
            sort_attributes=sort_orders[0])

        ll_prefix_genre_stats = compute_genre_age_statistics_from_reconstruction(
            updated_nodes, reduced_graph, movies_df,
            anon_type="label_list",
            anon_mapping=ll_prefix_mapping,
            classes=ll_prefix_classes,
            pattern_type='prefix',
            num_samples=5
        )

        # Partitioning anonymization
        part_graph = anonymize_with_partitioning(
            updated_nodes, reduced_graph, m=m_val, sort_attributes=sort_orders[0])

        # Use reconstruction-based approach for Partitioning
        part_genre_stats = compute_genre_age_statistics_from_reconstruction(
            updated_nodes, reduced_graph, movies_df,
            anon_type="partition",
            partition_graph=part_graph,
            num_samples=5
        )

        # Store results
        key = f"m{m_val}_sort{sort_name}"
        analysis_results['anonymized'][key] = {
            'label_list_full': ll_genre_stats,
            'label_list_prefix': ll_prefix_genre_stats,
            'partitioning': part_genre_stats
        }

    # Generate analysis plots
    print("\n=== GENERATING STATISTICAL ANALYSIS PLOTS ===")

    # Calculate differences between original and anonymized for each genre
    genre_differences = {}

    for genre, original_data in original_stats.items():
        original_age = original_data['avg_age']
        max_diff = 0

        # Find maximum difference across all methods and parameters
        for key, data in analysis_results['anonymized'].items():
            for method, method_data in data.items():
                if genre in method_data:
                    anon_age = method_data[genre]['avg_age']
                    diff = abs(anon_age - original_age)
                    max_diff = max(max_diff, diff)

        if max_diff > 0:  # Only include genres that have differences
            genre_differences[genre] = max_diff

    # Select top 10 genres with most difference
    top_diff_genres = sorted(genre_differences.items(),
                             key=lambda x: x[1], reverse=True)[:10]
    top_genre_names = [genre for genre, _ in top_diff_genres]

    # Method configuration
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

    sort_name = '_'.join(sort_orders[0])

    # PLOT 1: Bar chart comparison for a specific m value (m=15)
    print("Creating Plot 1: Bar chart comparison (m=15)")
    m_val = 15
    key = f"m{m_val}_sort{sort_name}"

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    fig.suptitle(f'Average Age by Genre (Top 10 Most Changed) - m={m_val}', fontsize=16)

    # Prepare data for bar chart
    x_positions = np.arange(len(top_genre_names))
    bar_width = 0.2

    # Collect ages for each method
    original_ages = []
    method_ages = {method: [] for method in methods}

    for genre in top_genre_names:
        # Original age
        original_age = original_stats.get(genre, {}).get('avg_age', 0)
        original_ages.append(original_age)

        # Anonymized ages for each method
        for method in methods:
            anon_data = analysis_results['anonymized'][key][method].get(genre, {})
            anon_age = anon_data.get('avg_age', 0)
            method_ages[method].append(anon_age)

    # Create bars
    bars_original = ax.bar(x_positions - 1.5 * bar_width, original_ages, bar_width,
                           label='Original', color='gray', alpha=0.7)

    for i, method in enumerate(methods):
        bars = ax.bar(x_positions - 0.5 * bar_width + i * bar_width, method_ages[method],
                      bar_width, label=method_names[method], color=method_colors[method], alpha=0.8)

    # Customize plot
    ax.set_xlabel('Genre')
    ax.set_ylabel('Average Age (years)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(top_genre_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_filename = f'genre_age_bar_m{m_val}_{sort_name}.png'
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {plot_filename}")

    # PLOT 2: Line plot showing Film-Noir genre with original vs anonymized comparison
    print("Creating Plot 2: Film-Noir average age vs m values")
    
    # Focus on Film-Noir genre specifically
    target_genre = 'Film-Noir'
    
    # Check if Film-Noir exists in the data
    if target_genre not in original_stats:
        print(f"Warning: {target_genre} not found in original stats. Available genres:")
        print(list(original_stats.keys())[:10])  # Show first 10 available genres
        # Use the first available genre as fallback
        target_genre = list(original_stats.keys())[0]
        print(f"Using {target_genre} instead.")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{target_genre} Genre: Original vs Anonymized Average Age', fontsize=16)
    
    # Get original age for Film-Noir
    original_age_constant = original_stats.get(target_genre, {}).get('avg_age', 0)
    original_ages = [original_age_constant] * len(m_values)
    
    # Plot for each anonymization method
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        # Collect anonymized ages for this method across different m values
        anonymized_ages = []
        for m_val in m_values:
            key = f"m{m_val}_sort{sort_name}"
            anon_age = analysis_results['anonymized'][key][method].get(target_genre, {}).get('avg_age', 0)
            anonymized_ages.append(anon_age)
        
        # Plot both lines
        ax.plot(m_values, original_ages, label='Original',
                color='gray', linewidth=3, linestyle='-')
        ax.plot(m_values, anonymized_ages, label='Anonymized',
                color=method_colors[method], linewidth=3)
        
        # Customize subplot
        ax.set_title(method_names[method], fontsize=14, fontweight='bold')
        ax.set_xlabel('M Value', fontsize=12)
        ax.set_ylabel('Average Age (years)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to better show differences
        all_ages = original_ages + anonymized_ages
        y_min, y_max = min(all_ages), max(all_ages)
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    plot_filename = f'film_noir_age_comparison_{sort_name}.png'
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {plot_filename}")

    return analysis_results


if __name__ == "__main__":
    # Load MovieLens dataset
    ratings_path = 'ml-latest-small/ratings.csv'
    movies_path = 'ml-latest-small/movies.csv'

    user_nodes, interaction_graph = load_movielens_as_graph(ratings_path, min_ratings_per_user=5)
    movies_df = pd.read_csv(movies_path)

    # Run genre-age analysis
    print("Starting genre-age analysis...")
    results = analyze_genre_age_anonymization(user_nodes, interaction_graph, movies_df)
    print("Genre-age analysis complete!")
