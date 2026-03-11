import operator
import pandas as pd
import random
from collections import defaultdict

"""
    Check if node_v_id can join class_c without sharing any interaction with its members
"""

def safety_condition(node_v_id, class_c, graph_v):
    interactions_of_v = graph_v.get(node_v_id, set())
    for member_id in class_c:
        interactions_of_member = graph_v.get(member_id, set())
        #shared_interactions = interactions_of_v.intersection(interactions_of_member)
        if not interactions_of_v.isdisjoint(interactions_of_member):    #I use this condition isntead of the intersection for efficiency
            return False
    return True

"""
Partition of nodes into classes of size ≤ m that respect safety_condition,
from the paper:

SORT(V);
for v ∈ V do
flag ← true;
for class c do
if SAFETYCONDITION(c,v) and SIZE(c) < m then
INSERT(c,v);
flag ← false; break;
if flag then INSERT(CREATENEWCLASS(),v,E);
"""
def divide_nodes(nodes, graph_v, m, sort_attributes):
    nodes_sorted = sorted(nodes, key=operator.itemgetter(*sort_attributes))
    classes = []
    for node in nodes_sorted:
        node_id = node["id"]
        flag = False
        for c in classes:
            if len(c) < m and safety_condition(node_id, c, graph_v):
                c.append(node_id)
                flag = True
                break
        if not flag:
            classes.append([node_id])
    return classes


"""
    Add realistic ages using a realistic distribution, I add them to have 
    a new sorting dimension and for make specific queries and statistical analysis
"""
def add_realistic_ages_random(nodes, seed=42):
    random.seed(seed)

    for node in nodes:
        # Movie watchers are typically 13-80, with most being 18-50
        # Using a weighted distribution
        age_ranges = [
            (13, 17, 0.05),  # Teens: 5%
            (18, 25, 0.25),  # Young adults: 25%
            (26, 35, 0.30),  # Adults: 30%
            (36, 50, 0.25),  # Middle-aged: 25%
            (51, 65, 0.12),  # Older adults: 12%
            (66, 80, 0.03),  # Seniors: 3%
        ]
        
        # Choose age range based on weights
        rand_val = random.random()
        cumulative = 0
        for min_age, max_age, weight in age_ranges:
            cumulative += weight
            if rand_val <= cumulative:
                node['age'] = random.randint(min_age, max_age)
                break
    
    return nodes


"""
    Loads the MovieLens ratings dataset from file and converts it into
    the bipartite interaction grpah required by the anonymization algorithm.

    Args:
        ratings_filepath: Path to the ratings.csv file.
        min_ratings_per_user: Filter users with fewer than this number of ratings.

    Returns:
         A tuple containing:
              1. The list of user nodes (with attributes).
              2. The graph_v dictionary (userId -> {set of movieIds} (interactions)).
"""
def load_movielens_as_graph(ratings_filepath, min_ratings_per_user=5):
    df = pd.read_csv(ratings_filepath)

    # For movies, we'll treat them as strings
    df['movieId'] = 'm' + df['movieId'].astype(str)
    
    # Create the graph structure
    # We want to filter out users who haven't rated many movies
    user_rating_counts = df.groupby('userId').size()
    active_users = user_rating_counts[user_rating_counts >= min_ratings_per_user].index
    
    df_filtered = df[df['userId'].isin(active_users)]
    
    graph_v = defaultdict(set)
    for _, row in df_filtered.iterrows():
        graph_v[row['userId']].add(row['movieId'])

    # Create the nodes list
    nodes = []
    for user_id in graph_v.keys():
        num_ratings = len(graph_v[user_id])
        nodes.append({
            'id': user_id,
            'num_ratings': num_ratings,
        })

    nodes = add_realistic_ages_random(nodes)
    return nodes, dict(graph_v)

"""
    Auxiliary function to check if a sampled user from
     a class is contained in a prefix pattern label list
"""


def check_correct_sampled_element_prefix_pattern(aux_list, anon_mapping, nodes_in_class):
    for i, node in enumerate(nodes_in_class):
        if aux_list[i] not in anon_mapping[node]:
            return False
    return True


"""
    Sample a graph from a label list anonymization, to perform queries on it.
    From the paper: 'A probabilistic approach is for
    the analyst to randomly sample a graph that is consistent with the
    anonymized data, and perform the analysis on this graph. That
    is, for each class, they choose an assignment of nodes to entities
    consistent with the possible labels (in the style of the methods de-
    scribed in Section 3.2.2). The query can be evaluated over the re-
    sulting graph'

    Args:
        anon_mapping: Dictionary mapping original_node_id -> assigned_label_list
        classes: List of node classes
        pattern_type: 'full' or 'prefix'
        original_graph: The original interaction graph to reconstruct from

    Returns:
        dict: A newly constructed graph_v (user_id -> {set of interactions}).
"""


def reconstruct_from_label_list(anon_mapping, classes, original_graph, pattern_type='full'):
    # First create the sampled mapping as before
    sampled_mapping = {}
    for nodes_in_class in classes:
        aux_list = list(nodes_in_class)
        # Randomly assign a true identity to each node in the class
        random.shuffle(aux_list)
        if pattern_type == 'prefix':
            while not check_correct_sampled_element_prefix_pattern(aux_list, anon_mapping, nodes_in_class):
                random.shuffle(aux_list)
        for i, node_id in enumerate(nodes_in_class):
            sampled_mapping[node_id] = aux_list[i]

    # Now construct the reconstructed graph using the sampled mapping
    reconstructed_graph = defaultdict(set)
    for original_user, sampled_user in sampled_mapping.items():
        if original_user in original_graph:
            reconstructed_graph[sampled_user].update(original_graph[original_user])

    return dict(reconstructed_graph)

"""
    Sample a graph from a label list anonymization, to perform queries on it.
    It randomly assigns interactions to members of a class to match the counts.

    Returns:
        dict: A newly constructed graph_v (user_id -> {set of interactions}).
"""
def reconstruct_from_partition(partition_graph):
    reconstructed_graph = defaultdict(set)
    for interaction, class_counts in partition_graph.items():
        for class_label, count in class_counts.items():
            # Get list of user IDs in this class
            members = list(class_label)
            # Randomly choose 'count' members for this interaction
            chosen_members = random.sample(members, k=count)
            for member_id in chosen_members:
                reconstructed_graph[member_id].add(interaction)

    return dict(reconstructed_graph)


"""
    A generic function to run a query against an anonymized graph
    It samples 'num_samples' graph from the anonymized ones as explain in the reconstruct function,
    runs the query_function on each, and averages the result.
"""
def run_query(query_function, original_graph, anon_type, anon_graph=None, pattern_type=None,classes=None, anon_mapping=None, num_samples=10, **kwargs):
    #anon_type = kwargs.get("anon_type")

    # Run once on the ground truth
    ground_truth_result = query_function(graph=original_graph, **kwargs)

    if anon_type == "original":
        return ground_truth_result, 0.0  # No error for the original

    results = []
    for _ in range(num_samples):
        if anon_type == "label_list":
            # Reconstruct a possible graph and run the query
            sampled_graph = reconstruct_from_label_list(anon_mapping, classes, original_graph, pattern_type)
            sample_result = query_function(graph=sampled_graph, **kwargs)
            #print(sample_result)
            results.append(sample_result)
        elif anon_type == "partition":
            # Reconstruct a possible graph and run the query
            sampled_graph = reconstruct_from_partition(anon_graph)
            sample_result = query_function(graph=sampled_graph, **kwargs)  # Query this new graph as if it were the original
            results.append(sample_result)

    estimated_result = sum(results) / len(results)
    relative_error = abs(estimated_result - ground_truth_result) / ground_truth_result if ground_truth_result > 0 else 0

    return estimated_result, relative_error

"""
    Randomly delete half of the interactions from the dataset
    otherwise with such a big number of interactions (> 100000) and relatively small number of users (610)
    the classes will have few components because it would
    be challenging to satisfy the safety condition and this would alter the privacy-utility analysis
"""
def randomly_delete_half_interactions(interaction_graph, seed=42):
    random.seed(seed)

    # Create a new graph with reduced interactions
    reduced_graph = {}

    for user_id, interactions in interaction_graph.items():
        interactions_list = list(sorted(interactions))  # Sort for deterministic ordering
        # Randomly select half of the interactions to keep
        num_to_keep = max(1, len(interactions_list) // 2)  # Keep at least 1 interaction
        kept_interactions = set(random.sample(interactions_list, num_to_keep))
        reduced_graph[user_id] = kept_interactions

    return reduced_graph