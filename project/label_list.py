import random
from collections import defaultdict
from utils import divide_nodes


"""
    Generates label lists using the "full pattern" (k=m)
    From the paper: Given a class of m entities Cj, a collection of m la-
    bel lists is formed based on an integer “pattern” p = {p0, p1 . . . pk−1},
    which is a subset of {0 . . . m − 1} of size exactly k. The label lists
    generated from p and 0 ≤ i < m for entities labeled u0 . . . um−1 are:
    list(p, i) = {ui+p0 mod m, ui+p1 mod m, . . . ui+pk−1 mod m}. 
"""

def generate_full_pattern_lists(class_nodes):
    full_list = sorted(class_nodes) #Sorting for determinism
    return [full_list] * len(class_nodes)


"""
Generates label lists using the "prefix pattern"
"""

def generate_prefix_pattern_lists(class_nodes, k):
    m = len(class_nodes)
    generated_lists = []
    for i in range(m):
        current_list_set = set()
        for j in range(k):
            node_index = (i + j) % m
            current_list_set.add(class_nodes[node_index])
        generated_lists.append(sorted(current_list_set))
        
    return generated_lists

"""
    function for label list anonymization process
    Returns anonymized node mapping, node classes, and anonymized graph structure.
    
    Returns:
        tuple: (anonymized_node_mapping, node_classes, anonymized_graph)
               - anonymized_node_mapping: original_node_id -> assigned_label_list
               - node_classes: list of node classes obtained from 'divide_nodes'
               - anonymized_graph: label_list -> {interactions} (like original graph structure)
"""

def anonymize_with_label_lists(nodes, graph_v, m, k, sort_attributes, pattern_type='full'):
    node_classes = divide_nodes(nodes, graph_v, m, sort_attributes)
    print("Step 1: Nodes partitioned into classes:")
    print(node_classes)
    
    anonymized_node_mapping = {}
    anonymized_graph = defaultdict(set)

    for class_c in node_classes:
        m_actual = len(class_c)

        if pattern_type == 'full':
            lists_for_class = generate_full_pattern_lists(class_c)
            # For full pattern, all get the same list, so order doesn't matter
            assignment_mapping = {i: i for i in range(len(class_c))}
        else: # 'prefix'
            k_actual = min(k, m_actual)
            class_c.sort() 
            lists_for_class = generate_prefix_pattern_lists(class_c, k_actual)
            
        # Randomic label_list assignment: the bipartite matching as described in the paper
            # A valid bipartite matching: each node can only be assigned to lists containing it
            valid_assignments = {}
            for node_idx, node_id in enumerate(class_c):
                valid_assignments[node_idx] = []
                # Find all lists that contain this node
                for list_idx, label_list in enumerate(lists_for_class):
                    if node_id in label_list:
                        valid_assignments[node_idx].append(list_idx)
            
            # Find a random valid matching
            used_lists = set()
            assignment_mapping = {}
            
            # Randomize the order in which assigning nodes
            node_order = list(range(len(class_c)))
            random.shuffle(node_order)
            
            for node_idx in node_order:
                # Get available lists for this node (valid and not yet used)
                available_lists = [list_idx for list_idx in valid_assignments[node_idx] 
                                 if list_idx not in used_lists]
                
                if available_lists:
                    # Randomly pick one of the available lists
                    chosen_list_idx = random.choice(available_lists)
                    assignment_mapping[node_idx] = chosen_list_idx
                    used_lists.add(chosen_list_idx)
                else:
                    #if no valid unused list, use any valid list
                    assignment_mapping[node_idx] = valid_assignments[node_idx][0]

        
        for i, node_id in enumerate(class_c):
            assigned_list_index = assignment_mapping[i]
            assigned_list = tuple(lists_for_class[assigned_list_index])  # Convert to tuple for hashing
            anonymized_node_mapping[node_id] = assigned_list
    
    # Build the anonymized graph structure: label_list -> {interactions}
    print("Step 2: Building anonymized graph structure...")
    for node_id, interactions in graph_v.items():
        label_list = anonymized_node_mapping[node_id]
        # Add all interactions of this node to the label list's interaction set
        anonymized_graph[label_list].update(interactions)

    # Convert defaultdict to regular dict for a better output
    return anonymized_node_mapping, node_classes, dict(anonymized_graph)


if __name__ == "__main__":
    # The Original, Private Data ---
    original_nodes = [
        {'id': 'u1', 'age': 29, 'sex': 'F', 'loc': 'NY'}, {'id': 'u2', 'age': 20, 'sex': 'M', 'loc': 'JP'},
        {'id': 'u3', 'age': 24, 'sex': 'F', 'loc': 'UK'}, {'id': 'u4', 'age': 31, 'sex': 'M', 'loc': 'NJ'},
        {'id': 'u5', 'age': 18, 'sex': 'M', 'loc': 'NJ'}, {'id': 'u6', 'age': 21, 'sex': 'F', 'loc': 'CA'},
        {'id': 'u7', 'age': 44, 'sex': 'M', 'loc': 'DE'},
    ]
    original_graph_v = {
        'u1': {'email1', 'friend1', 'game1'}, 'u2': {'email1', 'email2'}, 'u3': {'friend1', 'blog2'},
        'u4': {'game1', 'blog1'}, 'u5': {'email2'}, 'u6': {'blog1'}, 'u7': {'blog2'},
    }


    m_param, k_param = 3, 2
    sort_order = ['loc', 'age'] 

    #Anonymized Data (Full Pattern)
    anon_mapping_full, node_classes_full, anon_graph_full = anonymize_with_label_lists(
        original_nodes, original_graph_v, m_param, k_param, sort_order, pattern_type='full'
    )

    print("         ANONYMIZED DATA (FULL PATTERN, m=3)")
    print("Anonymized Graph (label_list -> {interactions}):")
    print(anon_graph_full)

    #Anonymized Data (Prefix Pattern)
    anon_mapping_prefix, node_classes_prefix, anon_graph_prefix = anonymize_with_label_lists(
        original_nodes, original_graph_v, m_param, k_param, sort_order, pattern_type='prefix'
    )

    print(f"        ANONYMIZED DATA (PREFIX PATTERN, m=3, k=2)")
    print("Anonymized Graph (label_list -> {interactions}):")
    print(anon_graph_prefix)

