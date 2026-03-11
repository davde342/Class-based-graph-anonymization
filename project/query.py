
"""
    Counts how many users with > `min_ratings` rated a movie from a `target_genre`
"""
def query_pair_ratings(graph, **kwargs):
    min_ratings = kwargs.get("min_ratings", 50)
    target_genre = kwargs.get("target_genre", "Sci-Fi")
    movies_df = kwargs.get("movies_df")  # For the movie metadata
    nodes = kwargs.get("nodes")

    # Find all movies in the target genre
    genre_movies = set('m' + movies_df[movies_df['genres'].str.contains(target_genre)]['movieId'].astype(str))

    count = 0
    node_attr_lookup = {n['id']: n for n in nodes}

    for user_id_in_graph, interactions in graph.items():
        # Check if the user has the required property
        user_attrs = node_attr_lookup.get(user_id_in_graph)
        if user_attrs.get("num_ratings", 0) > min_ratings:
            # Check if this user rated any movie from the target genre
            if not interactions.isdisjoint(genre_movies):
                count += 1
    return count


"""
    Counts users with exactly 'target_degree' ratings.
"""
def query_degree_distribution_test(graph, **kwargs):
    target_degree = kwargs.get("target_degree", 45)
    
    count = 0
    for user_id_in_graph, interactions in graph.items():
        if len(interactions) >= target_degree:
            count += 1
    
    return count

"""
    Counts user pairs who share at least 'min_shared_movies' movies.
"""
def query_structural_similarity(graph, **kwargs):
    min_shared_movies = kwargs.get("min_shared_movies", 5)
    
    users = list(graph.keys())
    pair_count = 0
    # Check all pairs of users
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user1, user2 = users[i], users[j]
            movies1 = graph.get(user1, set())
            movies2 = graph.get(user2, set())
            
            # Count shared movies
            shared = len(movies1.intersection(movies2))
            if shared >= min_shared_movies:
                pair_count += 1
    
    return pair_count
