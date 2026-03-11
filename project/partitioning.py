from collections import defaultdict
from utils import divide_nodes

"""
The partitioning anonymization function

Args:
    nodes (list): The list of node attributes as dictionaries.
    original_graph (dict): The original graph structure.
    m (int): The maximum class size.
    sort_attributes (list): Node attributes to sort by for grouping.

Returns:
    A tuple containing:
      The anonymized partition graph (interaction -> {class_id -> count}).

"""

def anonymize_with_partitioning(nodes, original_graph, m, sort_attributes):
    # 1. Partition nodes into classes using the same core algorithm
    node_classes = divide_nodes(nodes, original_graph, m, sort_attributes)
    print("Nodes partitioned into classes:")
    print(node_classes)

    # Initialize the two data structures that will be published
    partition_graph = defaultdict(dict)

    # 2. Iterate through each class to build the summarized graph and attributes
    for class_c in node_classes:
        # Create a canonical, hashable identifier for the class
        class_label = tuple(sorted(class_c))
        for node_id in class_c:
            interactions = original_graph.get(node_id, set())
            for interaction_id in interactions:
                # Get the current count of links from this class to this interaction
                current_count = partition_graph[interaction_id].get(class_label, 0)
                partition_graph[interaction_id][class_label] = current_count + 1

    return dict(partition_graph)




if __name__ == "__main__":
    original_nodes = [
        {'id': 'u1', 'age': 29, 'sex': 'F', 'loc': 'NY'}, {'id': 'u2', 'age': 20, 'sex': 'M', 'loc': 'JP'},
        {'id': 'u3', 'age': 24, 'sex': 'F', 'loc': 'UK'}, {'id': 'u4', 'age': 31, 'sex': 'M', 'loc': 'NJ'},
        {'id': 'u5', 'age': 18, 'sex': 'M', 'loc': 'NJ'}, {'id': 'u6', 'age': 21, 'sex': 'F', 'loc': 'CA'},
        {'id': 'u7', 'age': 44, 'sex': 'M', 'loc': 'DE'},
    ]
    original_graph = {
        'u1': {'email1', 'friend1', 'game1'}, 'u2': {'email1', 'email2'}, 'u3': {'friend1', 'blog2'},
        'u4': {'game1', 'blog1'}, 'u5': {'email2'}, 'u6': {'blog1'}, 'u7': {'blog2'},
    }

    print("              ORIGINAL (PRIVATE) DATA")
    print("Original User Attributes:")
    print(original_nodes)
    print("\nOriginal Graph Structure (user -> interactions):")
    print(original_graph)
    
    #Anonymization Parameters
    m_param = 3
    sort_order = ['loc', 'age']  

    #The Anonymized Data (Partitioning)
    partition_graph = anonymize_with_partitioning(
        original_nodes, original_graph, m_param, sort_order
    )

    print("         ANONYMIZED DATA")
    print(partition_graph)