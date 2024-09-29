
from flask import Flask, request, jsonify

import pandas as pd
import pickle
import json
from embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

from flask_cors import CORS

#model
EMBEDDING_MODEL = "text-embedding-3-small"

#dataset
input_json_path = "./ngos_list.json"

with open(input_json_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

df = pd.DataFrame(json_data)
embedding_cache_path = "recommendations_embeddings_cache.pkl"
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

def print_recommendations_from_strings(
    strings: list[str],
    index_of_source_string: int,
    k_nearest_neighbors: int = 1,
    model=EMBEDDING_MODEL,
    prompt: str = ""
) -> list[int]:
    article_descriptions.append(prompt)
    embeddings = [embedding_from_string(string, model=model) for string in strings]
    query_embedding = embeddings[index_of_source_string]
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    query_string = strings[index_of_source_string]
    k_counter = 0
    nearest_dict = {}
    
    for i in indices_of_nearest_neighbors:
        if query_string == strings[i]:
            continue
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1
        nearest_id = df.iloc[i]['ID']  # Assuming 'ID' is the column name in your DataFrame
        
        # Format the distance
        distance = f'{distances[i]:0.3f}'
        
        # Store ID and distance in the dictionary
        nearest_dict[nearest_id] = distance
    return nearest_dict





# prompt = 'Starsza schorowana pani potrzebująca pomocy socjalnej'
article_descriptions = df["combined"].tolist()
index_of_source_string = len(article_descriptions) - 1


# tony_blair_articles = print_recommendations_from_strings(
#     strings=article_descriptions,
#     index_of_source_string=index_of_source_string,
#     k_nearest_neighbors=5,
#     prompt = 'Starsza schorowana pani potrzebująca pomocy socjalnej'
# )


app = Flask(__name__)

CORS(app)

@app.route('/search', methods=['GET'])
def knn_search():
    try:

        k = request.args.get('k', type=int)
        prompt = request.args.get('prompt', type=str)

        if k is None or prompt is None:
            return jsonify({"error": "Missing required parameters: 'k' and 'prompt'."}), 400

        # Append the prompt to the article descriptions
        article_descriptions.append(prompt)
        index_of_source_string = len(article_descriptions) - 1  # The index of the prompt

        # Call your recommendation function
        nearest_neighbors = print_recommendations_from_strings(article_descriptions, index_of_source_string, int(k), prompt=prompt)

        response = {
            "k_nearest_neighbours": nearest_neighbors
        }
        return jsonify(response)
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred."}), 500
    

@app.route('/list-ngos', methods=['GET'])
def get_ngos():
    # Get the array of UUIDs from the query parameters
    uuids = request.args.getlist('uuid')
    
    # Filter ngos_data to find objects that match the given UUIDs and exclude specific fields
    matched_ngos = [
        {k: v for k, v in ngo.items() if k not in ['combined', 'n_tokens', 'embedding']}
        for ngo in json_data if ngo['ID'] in uuids
    ]
    
    # Return the matched objects as JSON
    return jsonify(matched_ngos)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
