import re
from typing import Tuple, List, Dict, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt
import os
import joblib

@keras.utils.register_keras_serializable()
def choose_embedding(inputs):
    userId_embed, userEmbed, userId = inputs
    condition = tf.equal(userId, -1)
    return tf.where(condition, userEmbed, userId_embed)

class MovieRecommendationSystem:
    def __init__(self) -> None:
        self.model: keras.Model
        self.user_encoder: LabelEncoder
        self.movie_encoder: LabelEncoder
        self.scaler: MinMaxScaler
        self.df: pd.DataFrame
        self.num_users: int
        self.num_movies: int
        self.embedding_dim: int = 64

    @staticmethod
    def format_title(title: str) -> str:
        pattern = r'^(.*), The(\s+\(\d{4}\))$'
        match = re.match(pattern, title)
        if match:
            return f"The {match.group(1)}{match.group(2)}"
        return title

    def load_data(self) -> None:
        ratings = pd.read_csv('dataset/ratings.csv').dropna()
        movies = pd.read_csv('dataset/movies.csv').dropna()
        
        movies['title'] = movies['title'].apply(self.format_title)
        
        self.df = pd.merge(ratings, movies).groupby('movieId').filter(lambda x: len(x) >= 100).drop(['timestamp', 'genres'], axis=1)

    def preprocess_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.df['userId'] = self.user_encoder.fit_transform(self.df['userId'])
        self.df['movieId'] = self.movie_encoder.fit_transform(self.df['movieId'])

        X = self.df[['userId', 'movieId']]
        y = self.df['rating']

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        self.num_users = self.df['userId'].nunique()
        self.num_movies = self.df['movieId'].nunique()

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_scaled, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)

        train_data = {
            'user': X_train['userId'].values.reshape(-1, 1),
            'movie': X_train['movieId'].values.reshape(-1, 1),
            'rating': y_train
        }
        val_data = {
            'user': X_val['userId'].values.reshape(-1, 1),
            'movie': X_val['movieId'].values.reshape(-1, 1),
            'rating': y_val
        }
        test_data = {
            'user': X_test['userId'].values.reshape(-1, 1),
            'movie': X_test['movieId'].values.reshape(-1, 1),
            'rating': y_test
        }

        return train_data, val_data, test_data
    
    def create_model(self) -> keras.Model:
        userId_input = keras.Input(shape=(1,), name='userId_input')
        userEmbed_input = keras.Input(shape=(self.embedding_dim,), name='userEmbed_input')
        movie_input = keras.Input(shape=(1,), name='movie_input')

        user_embedding = layers.Embedding(input_dim=self.num_users, output_dim=self.embedding_dim, name='user_embedding')(userId_input)
        user_embedding = layers.Flatten()(user_embedding)

        user_vector = layers.Lambda(choose_embedding, name='choose_embedding')([user_embedding, userEmbed_input, userId_input])
        user_vector = layers.Dense(self.embedding_dim, activation='relu')(user_vector)

        movie_embedding = layers.Embedding(input_dim=self.num_movies, output_dim=self.embedding_dim, name='movie_embedding')(movie_input)
        movie_vector = layers.Flatten()(movie_embedding)

        concatenated = layers.Concatenate()([user_vector, movie_vector])
        multiply = layers.Multiply()([user_vector, movie_vector])
        combined = layers.Concatenate()([concatenated, multiply])

        dense1 = layers.Dense(256, activation='relu')(combined)
        dropout1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(128, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.3)(dense2)
        dense3 = layers.Dense(64, activation='relu')(dropout2)
        dropout3 = layers.Dropout(0.3)(dense3)
        output = layers.Dense(1, activation='linear')(dropout3)

        model = keras.Model(inputs=[userId_input, userEmbed_input, movie_input], outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def train_model(self, train_data: Dict[str, np.ndarray], val_data: Dict[str, np.ndarray], epochs: int = 50, batch_size: int = 4096) -> keras.callbacks.History:
        X_user_train = train_data['user']
        X_movie_train = train_data['movie']
        y_train = train_data['rating']
        X_userEmbed_train = np.zeros((len(X_user_train), self.embedding_dim))  # Dummy embeddings for training

        X_user_val = val_data['user']
        X_movie_val = val_data['movie']
        y_val = val_data['rating']
        X_userEmbed_val = np.zeros((len(X_user_val), self.embedding_dim))  # Dummy embeddings for validation

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

        history = self.model.fit(
            [X_user_train, X_userEmbed_train, X_movie_train], y_train,
            validation_data=([X_user_val, X_userEmbed_val, X_movie_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr]
        )

        return history

    def evaluate_model(self, test_data: Dict[str, np.ndarray]) -> Tuple[float, float]:
        X_user_test = test_data['user']
        X_movie_test = test_data['movie']
        y_test = test_data['rating']
        
        dummy_embedding = np.zeros((len(X_user_test), self.embedding_dim))

        loss, mae = self.model.evaluate([X_user_test, dummy_embedding, X_movie_test], y_test)
        return loss, mae

    @staticmethod
    def plot_history(history: keras.callbacks.History) -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss (Scaled)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE (Scaled)')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self) -> None:
        model_save_path = 'ncf_model.keras'
        self.model.save(model_save_path, save_format='tf')
        print(f"Model saved to {model_save_path}")

        joblib.dump(self.user_encoder, 'user_encoder.joblib')
        joblib.dump(self.movie_encoder, 'movie_encoder.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        print("Encoders and scaler saved.")

    def load_saved_model(self) -> bool:
        if os.path.exists('files/ncf_model.keras'):
            custom_objects = {'choose_embedding': choose_embedding}
            self.model = keras.models.load_model('files/ncf_model.keras', custom_objects=custom_objects)
            self.user_encoder = joblib.load('files/user_encoder.joblib')
            self.movie_encoder = joblib.load('files/movie_encoder.joblib')
            self.scaler = joblib.load('files/scaler.joblib')
            print("Model, encoders, and scaler loaded successfully.")
            return True
        print("No saved model found.")
        return False

    def unscale_predictions(self, scaled_preds: np.ndarray) -> np.ndarray:
        unscaled = self.scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
        return unscaled

    def get_recommendations(self, user_input: Union[int, np.ndarray], top_n: int = 10) -> List[Dict[str, Union[str, float]]]:
        all_movies = np.arange(self.num_movies).reshape(-1, 1)
        
        if isinstance(user_input, int):
            user_id = np.full((len(all_movies), 1), self.user_encoder.transform([user_input])[0])
            user_embed = np.zeros((len(all_movies), self.embedding_dim))
        else:
            user_id = np.full((len(all_movies), 1), -1)  # Dummy user ID for new users
            user_embed = np.tile(user_input, (len(all_movies), 1))

        predictions = self.model.predict([user_id, user_embed, all_movies]).flatten()
        predictions = self.unscale_predictions(predictions)
        
        encoded_to_original = dict(zip(range(len(self.movie_encoder.classes_)), self.movie_encoder.classes_))
        valid_movie_ids = set(self.df['movieId'].unique())
        
        movie_indices = np.argsort(predictions)[::-1]
        
        recommendations = []
        for idx in movie_indices:
            original_movie_id = encoded_to_original[idx]
            if original_movie_id in valid_movie_ids:
                movie_rows = self.df[self.df['movieId'] == original_movie_id]
                title = movie_rows['title'].values[0]
                predicted_rating = predictions[idx]
                recommendations.append({
                    'movieId': original_movie_id,
                    'title': title,
                    'predicted_rating': predicted_rating
                })
                if len(recommendations) == top_n:
                    break
        
        return recommendations

def main() -> None:
    recommender = MovieRecommendationSystem()
    recommender.load_data()

    if os.path.exists('files/ncf_model.keras'):
        load_model = input("Do you want to load a saved model? (y/n): ").lower().strip()
        if load_model == 'y':
            if recommender.load_saved_model():
                recommender.num_users = recommender.df['userId'].nunique()
                recommender.num_movies = recommender.df['movieId'].nunique()
            else:
                print("Failed to load model. Proceeding with training a new model.")
                load_model = 'n'
        else:
            load_model = 'n'
    else:
        load_model = 'n'

    if load_model == 'n':
        train_data, val_data, test_data = recommender.preprocess_data()
        recommender.model = recommender.create_model()
        history = recommender.train_model(train_data, val_data, epochs=10)
        recommender.plot_history(history)

        test_loss, test_mae = recommender.evaluate_model(test_data)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")

        save_model = input("Do you want to save the model? (y/n): ").lower().strip()
        if save_model == 'y':
            recommender.save_model()

    user_id = int(input("Enter an existing user ID to get recommendations: "))
    recommendations = recommender.get_recommendations(user_id)
    print(f"\nTop {len(recommendations)} movie recommendations for user {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Predicted Rating: {rec['predicted_rating']:.2f})")

    new_user_embedding = np.random.rand(recommender.embedding_dim)
    recommendations = recommender.get_recommendations(new_user_embedding)
    print(f"\nTop {len(recommendations)} movie recommendations for a new user:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Predicted Rating: {rec['predicted_rating']:.2f})")

if __name__ == "__main__":
    main()