from partitioning import anonymize_with_partitioning
from label_list import anonymize_with_label_lists

"""
    Create a test dataset for algorithm validation
"""
def create_test_dataset():
    nodes = [
        {'id': 'u1', 'age': 25, 'location': 'NY', 'num_ratings': 10},
        {'id': 'u2', 'age': 30, 'location': 'NY', 'num_ratings': 15},
        {'id': 'u3', 'age': 35, 'location': 'CA', 'num_ratings': 8},
        {'id': 'u4', 'age': 28, 'location': 'CA', 'num_ratings': 12},
        {'id': 'u5', 'age': 22, 'location': 'TX', 'num_ratings': 20},
        {'id': 'u6', 'age': 40, 'location': 'TX', 'num_ratings': 5},
        {'id': 'u7', 'age': 33, 'location': 'FL', 'num_ratings': 18},
        {'id': 'u8', 'age': 27, 'location': 'FL', 'num_ratings': 14},
        {'id': 'u9', 'age': 31, 'location': 'WA', 'num_ratings': 9},
        {'id': 'u10', 'age': 26, 'location': 'WA', 'num_ratings': 11}
    ]
    
    graph_v = {
        'u1': {'movie1', 'movie2', 'movie3'},
        'u2': {'movie4', 'movie5'},
        'u3': {'movie1', 'movie6'},
        'u4': {'movie7', 'movie8'},
        'u5': {'movie2', 'movie9'},
        'u6': {'movie10'},
        'u7': {'movie3', 'movie11'},
        'u8': {'movie12'},
        'u9': {'movie4', 'movie13'},
        'u10': {'movie5', 'movie14'}
    }
    
    return nodes, graph_v


"""
    Check if the safety condition is satisfied for all classes.

    Args:
        graph_v: Original interaction graph
        node_classes: List of node classes from partitioning

    Returns:
        tuple: (is_valid, violations) - validation result and list of violations
"""
def validate_safety_condition(graph_v, node_classes):
    print("SAFETY CONDITION CHECK")
    violations = []
    
    for class_idx, class_nodes in enumerate(node_classes):
        # Check all pairs within the class
        for i, node1 in enumerate(class_nodes):
            for j, node2 in enumerate(class_nodes):
                if i >= j:
                    continue
                    
                interactions1 = graph_v.get(node1, set())
                interactions2 = graph_v.get(node2, set())
                
                # Check if they share any interactions (violation of safety condition)
                shared_interactions = interactions1.intersection(interactions2)
                if shared_interactions:
                    violation = {
                        'class_index': class_idx,
                        'node1': node1,
                        'node2': node2,
                        'shared_interactions': shared_interactions
                    }
                    violations.append(violation)
    
    is_valid = len(violations) == 0
    
    if is_valid:
        print("Safety condition satisfied")
    else:
        print(f"Safety condition not satisfied")
        for v in violations:
            print(f"  Class {v['class_index']}: {v['node1']} and {v['node2']} share {v['shared_interactions']}")
    
    return is_valid, violations


"""
    Check if all classes have size <= m

    Args:
        node_classes: List of node classes
        m: Maximum allowed class size

    Returns:
        tuple: (is_valid, violations) - validation result and oversized classes
"""
def validate_m_anonymity(node_classes, m):
    print(f" ALIDATING m-ANONYMITY (m={m})")
    violations = []
    
    for class_idx, class_nodes in enumerate(node_classes):
        class_size = len(class_nodes)
        if class_size > m:
            violations.append({
                'class_index': class_idx,
                'class_size': class_size,
                'nodes': class_nodes
            })
    
    is_valid = len(violations) == 0
    
    if is_valid:
        print(f"Class size satisfied - All classes have size <= {m}")
    else:
        print(f"Class size not satisfied")
        for v in violations:
            print(f"  Class {v['class_index']}: size {v['class_size']}")
    
    return is_valid, violations


"""
    Check specific properties of label list anonymization.
"""
def validate_label_list_properties(node_mapping, node_classes, pattern_type, k=None):
    print(f"LABEL LIST PROPERTIES CHECK ({pattern_type.upper()})")
    violations = []
    
    # Check that each node gets a valid label list
    for node_id, label_list in node_mapping.items():
        # Find which class this node belongs to
        node_class = None
        for class_nodes in node_classes:
            if node_id in class_nodes:
                node_class = class_nodes
                break
        
        # Validate label list properties based on pattern type
        if pattern_type == 'full':
            # For full pattern, label list should contain all nodes in the class
            expected_label = sorted(node_class)
            if sorted(label_list) != expected_label:
                violations.append(f"Node {node_id}: expected full class {expected_label}, got {label_list}")
        
        elif pattern_type == 'prefix':
            # For prefix pattern, label list should be a subset of the class with size <= k
            if not set(label_list).issubset(set(node_class)):
                violations.append(f"Node {node_id}: label list {label_list} not subset of class {node_class}")
            
            if k and len(label_list) > k:
                violations.append(f"Node {node_id}: label list size {len(label_list)} > k={k}")
            
        # Validate that the node itself is in its label list
        if node_id not in label_list:
            violations.append(f"Node {node_id} not in its own label list {label_list}")
    
    is_valid = len(violations) == 0
    
    if is_valid:
        print(f"Label list properties satisfied")
    else:
        print(f"Label list properties not satisfied - {len(violations)} violations:")
    return is_valid, violations


"""
    Check specific properties of partitioning anonymization

    Args:
        partition_graph: Partitioning result (interaction -> {class -> count})
        original_graph: Original interaction graph

    Returns:
        tuple: (is_valid, violations)
"""
def validate_partitioning_properties(partition_graph, original_graph):
    print("CHECKING PARTITIONING PROPERTIES")
    violations = []
    
    # Validate that interaction counts are correct
    for interaction, class_counts in partition_graph.items():
        total_count = sum(class_counts.values())
        
        # Count how many original nodes had this interaction
        original_count = sum(1 for node_interactions in original_graph.values() 
                           if interaction in node_interactions)
        
        if total_count != original_count:
            violations.append(f"Interaction {interaction}: count mismatch {total_count} != {original_count}")
    
    # Validate that all original interactions are represented
    original_interactions = set()
    for interactions in original_graph.values():
        original_interactions.update(interactions)
    
    partition_interactions = set(partition_graph.keys())
    missing = original_interactions - partition_interactions
    extra = partition_interactions - original_interactions
    
    if missing:
        violations.extend([f"Missing interaction: {i}" for i in missing])
    if extra:
        violations.extend([f"Extra interaction: {i}" for i in extra])
    
    is_valid = len(violations) == 0
    
    if is_valid:
        print("Partitioning properties satisfied")
    else:
        print(f"Partitioning properties not satisfied")
    
    return is_valid, violations


"""
    Run different validation tests on all anonymization algorithms

    Returns:
        dict: validation results
"""
def run_validation():
    print("Algorithms correctness analysis")
    
    # Create test dataset
    nodes, graph_v = create_test_dataset()
    m_value = 3
    k_value = 2
    sort_attributes = ['age']
    
    validation_results = {}
    
    # Test 1: Label List Full Pattern
    print("Testing Label List Full Pattern")

    ll_mapping_full, ll_classes_full, ll_graph_full = anonymize_with_label_lists(
        nodes, graph_v, m=m_value, k=m_value, pattern_type='full', sort_attributes=sort_attributes)

    # Validate safety condition
    safety_valid, safety_violations = validate_safety_condition(graph_v, ll_classes_full)

    # Validate m-anonymity
    m_anon_valid, m_anon_violations = validate_m_anonymity(ll_classes_full, m_value)

    # Validate label list properties
    ll_props_valid, ll_violations = validate_label_list_properties(ll_mapping_full, ll_classes_full, 'full')

    validation_results['label_list_full'] = {
        'safety_condition': safety_valid,
        'm_anonymity': m_anon_valid,
        'label_list_properties': ll_props_valid,
        'overall': safety_valid and m_anon_valid  and ll_props_valid
        }

    # Test 2: Label List Prefix Pattern
    print("Testing Label List Prefix Pattern")
    

    ll_mapping_prefix, ll_classes_prefix, ll_graph_prefix = anonymize_with_label_lists(
        nodes, graph_v, m=m_value, k=k_value, pattern_type='prefix', sort_attributes=sort_attributes)

    # Validate safety condition
    safety_valid, safety_violations = validate_safety_condition(graph_v, ll_classes_prefix)

    # Validate m-anonymity
    m_anon_valid, m_anon_violations = validate_m_anonymity(ll_classes_prefix, m_value)

    # Validate label list properties
    ll_props_valid, ll_violations = validate_label_list_properties(ll_mapping_prefix, ll_classes_prefix, 'prefix', k_value)

    validation_results['label_list_prefix'] = {
        'safety_condition': safety_valid,
        'm_anonymity': m_anon_valid,
        'label_list_properties': ll_props_valid,
        'overall': safety_valid and m_anon_valid and ll_props_valid
    }

    
    # Test 3: Partitioning
    print("Testing Partitioning")

    partition_graph = anonymize_with_partitioning(
        nodes, graph_v, m=m_value, sort_attributes=sort_attributes)

    # Validate safety condition
    safety_valid, safety_violations = validate_safety_condition(graph_v, ll_classes_full)

    # Validate m-anonymity
    m_anon_valid, m_anon_violations = validate_m_anonymity(ll_classes_full, m_value)

    # Validate partitioning properties
    part_props_valid, part_violations = validate_partitioning_properties(partition_graph, graph_v)

    validation_results['partitioning'] = {
        'safety_condition': safety_valid,
        'm_anonymity': m_anon_valid,
        'partitioning_properties': part_props_valid,
        'overall': safety_valid and m_anon_valid and part_props_valid
    }

    return validation_results


if __name__ == "__main__":
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
