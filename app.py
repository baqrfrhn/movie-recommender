import os
import logging
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@keras.utils.register_keras_serializable()
def choose_embedding(inputs):
    """
    Custom layer to choose between user ID embedding and user embedding.
    """
    userId_embed, userEmbed, userId = inputs
    condition = tf.equal(userId, -1)
    return tf.where(condition, userEmbed, userId_embed)

# Load model and data
try:
    model = tf.keras.models.load_model('files/ncf_model.keras')
    user_encoder = joblib.load('files/user_encoder.joblib')
    movie_encoder = joblib.load('files/movie_encoder.joblib')
    scaler = joblib.load('files/scaler.joblib')
    movies_df = pd.read_csv('dataset/movies.csv')
    logger.info("Model and data loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or data: {str(e)}")
    raise

def format_title(title: str) -> str:
    """
    Format movie titles by moving 'The' to the beginning if it's at the end.
    """
    pattern = r'^(.*), The(\s+\(\d{4}\))$'
    match = re.match(pattern, title)
    if match:
        return f"The {match.group(1)}{match.group(2)}"
    return title

def optimize_user_embedding(user_ratings: List[Dict[str, Any]], num_iterations: int = 500, learning_rate: float = 0.005) -> np.ndarray:
    """
    Optimize the user embedding based on their ratings.
    """
    # Initialize user embedding with the average of movie embeddings
    rated_movie_ids = [r['movieId'] for r in user_ratings]
    movie_indices = movie_encoder.transform(rated_movie_ids)
    movie_embeddings = model.get_layer('movie_embedding')(tf.constant(movie_indices, dtype=tf.int32))
    initial_user_embed = tf.reduce_mean(movie_embeddings, axis=0, keepdims=True)
    
    user_embed = tf.Variable(initial_user_embed, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate)

    rated_movies = tf.constant([[movie_encoder.transform([r['movieId']])[0]] for r in user_ratings], dtype=tf.int32)
    true_ratings = tf.constant([[r['rating']] for r in user_ratings], dtype=tf.float32)

    num_movies = tf.shape(rated_movies)[0]
    user_id = tf.fill([num_movies, 1], -1)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            user_embed_repeated = tf.repeat(user_embed, repeats=num_movies, axis=0)
            predictions = model([user_id, user_embed_repeated, rated_movies])
            loss = tf.reduce_mean(tf.square(predictions - true_ratings))
            # Add L2 regularization
            l2_loss = 0.01 * tf.reduce_sum(tf.square(user_embed))
            total_loss = loss + l2_loss

        gradients = tape.gradient(total_loss, [user_embed])
        optimizer.apply_gradients(zip(gradients, [user_embed]))
        return loss

    losses = []
    patience = 20
    best_loss = float('inf')
    best_embed = None
    epochs_no_improve = 0

    for i in range(num_iterations):
        loss = train_step()
        losses.append(loss.numpy())

        if loss < best_loss:
            best_loss = loss
            best_embed = user_embed.numpy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at iteration {i}")
            break

    # Use exponential moving average for smoothing
    alpha = 0.1
    ema_embed = best_embed
    for _ in range(10):  # Apply EMA a few times for stability
        ema_embed = alpha * user_embed.numpy() + (1 - alpha) * ema_embed

    return ema_embed

def get_diverse_recommendations(recommendations: List[Dict[str, Any]], n: int = 10, user_ratings: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Get diverse movie recommendations based on genres, decades, and franchises.
    """
    diverse_recommendations = []
    genre_counter = Counter()
    decade_counter = Counter()
    franchise_counter = Counter()
    
    # Extract franchises and genres from user ratings
    user_franchises = set()
    user_genres = set()
    if user_ratings:
        for movie in user_ratings:
            franchise = get_franchise(movie['title'])
            if franchise:
                user_franchises.add(franchise)
            user_genres.update(movie['genres'].split('|'))
    
    def get_decade(year):
        return (year // 10) * 10
    
    def get_franchise(title):
        franchise_patterns = [
            r'^(.*?)\s*\d+',  # Matches "Franchise Name 2"
            r'^(.*?)\s*:\s',  # Matches "Franchise Name: Subtitle"
            r'^The\s(.*?)\s'  # Matches "The Franchise Name"
        ]
        for pattern in franchise_patterns:
            match = re.match(pattern, title)
            if match:
                return match.group(1).strip().lower()
        return None
    
    def movie_diversity_score(movie):
        genres = movie['genres'].split('|')
        year = int(re.search(r'\((\d{4})\)', movie['title']).group(1))
        decade = get_decade(year)
        franchise = get_franchise(movie['title'])
        
        genre_score = sum(2 / (genre_counter[genre] + 1) for genre in genres)
        decade_score = 5 / (decade_counter[decade] + 1)
        franchise_score = 3 / (franchise_counter[franchise] + 1) if franchise else 3
        
        # Boost score for genres and franchises not in user ratings
        if user_ratings:
            genre_novelty = sum(3 for genre in genres if genre not in user_genres)
            franchise_novelty = 5 if franchise and franchise not in user_franchises else 0
        else:
            genre_novelty = franchise_novelty = 0
        
        # Recency score: favor more recent movies slightly
        current_year = datetime.now().year
        recency_score = min(3, (current_year - year) / 10)
        
        return genre_score + decade_score + franchise_score + genre_novelty + franchise_novelty + recency_score
    
    while len(diverse_recommendations) < n and recommendations:
        best_movie = max(recommendations, key=movie_diversity_score)
        diverse_recommendations.append(best_movie)
        
        genres = best_movie['genres'].split('|')
        year = int(re.search(r'\((\d{4})\)', best_movie['title']).group(1))
        decade = get_decade(year)
        franchise = get_franchise(best_movie['title'])
        
        for genre in genres:
            genre_counter[genre] += 1
        decade_counter[decade] += 1
        if franchise:
            franchise_counter[franchise] += 1
        
        recommendations.remove(best_movie)
    
    return diverse_recommendations

@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('.', 'index.html')

@app.route('/api/search', methods=['GET'])
def search_movies():
    """
    Search for movies based on a query string.
    """
    try:
        query = request.args.get('q', '').lower()
        logger.info(f"Received search query: {query}")
        results = movies_df[movies_df['title'].str.lower().str.contains(query)].to_dict('records')
        logger.info(f"Found {len(results)} results for query: {query}")
        return jsonify(results[:10])
    except Exception as e:
        logger.error(f"Error in search_movies: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Generate movie recommendations based on user ratings.
    """
    try:
        data = request.json
        user_ratings = data['ratings']
        diverse = data.get('diverse', False)
        logger.info(f"Received request for recommendations. User ratings: {len(user_ratings)}, Diverse: {diverse}")

        user_embed = optimize_user_embedding(user_ratings)
        
        all_movies = np.arange(len(movie_encoder.classes_)).reshape(-1, 1)
        num_movies = len(all_movies)
        user_id = np.full((num_movies, 1), -1)  # Always use -1 for user ID
        user_embed_repeated = np.tile(user_embed, (num_movies, 1))

        # Predict in batches to avoid memory issues
        batch_size = 1024
        predictions = []
        for i in range(0, num_movies, batch_size):
            batch_end = min(i + batch_size, num_movies)
            batch_predictions = model.predict([
                user_id[i:batch_end],
                user_embed_repeated[i:batch_end],
                all_movies[i:batch_end]
            ], verbose=0).flatten()
            predictions.extend(batch_predictions)

        predictions = np.array(predictions)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Create a set of movie IDs that the user has already rated
        rated_movie_ids = set(r['movieId'] for r in user_ratings)

        # Filter out already rated movies and create movie_predictions dictionary
        movie_predictions = {
            movie_encoder.inverse_transform([i])[0]: pred 
            for i, pred in enumerate(predictions)
            if movie_encoder.inverse_transform([i])[0] not in rated_movie_ids
        }

        sorted_predictions = sorted(movie_predictions.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for movie_id, predicted_rating in sorted_predictions[:50]:  # Get top 50 for diversity
            movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            recommendations.append({
                'movieId': int(movie_id),
                'title': format_title(movie['title']),
                'genres': movie['genres'],
                'predicted_rating': float(predicted_rating),
                'original_rating': float(predicted_rating)
            })

        if diverse:
            recommendations = get_diverse_recommendations(recommendations)
        else:
            recommendations = recommendations[:10]

        # Map ratings to 1-5 star scale for display
        for r in recommendations:
            r['star_rating'] = min(5, max(1, round(r['predicted_rating'])))

        logger.info(f"Returning {len(recommendations)} recommendations")
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)