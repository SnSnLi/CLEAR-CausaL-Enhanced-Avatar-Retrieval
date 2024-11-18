required_functions = ['get_text_embedding', 'get_node_embedding', 'compute_similarity_score']
for func in required_functions:
    if func not in globals():
        print(f"Missing function: {func}")