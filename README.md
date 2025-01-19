# Neural Collaborative Filtering Movie Recommender

A Flask-based web application that provides personalized movie recommendations using a Neural Collaborative Filtering (NCF) approach. The system can handle both existing users and new users by optimizing user embeddings based on their ratings.

## Overview

This project implements a movie recommendation system using deep learning techniques. It combines collaborative filtering with neural networks to provide accurate and personalized movie recommendations. The system can:

- Generate recommendations for existing users based on their rating history
- Create personalized recommendations for new users based on their initial ratings
- Provide diverse recommendations considering genres, decades, and movie franchises
- Search through the movie database
- Handle real-time rating updates and embedding optimization

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Flask
- NumPy
- Pandas
- scikit-learn
- joblib
- Flask-CORS

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `app.py` - Main Flask application implementing the recommendation API
- `script.py` - Training script for the Neural Collaborative Filtering model
- `dataset/` - Directory for movie and ratings data (not included)
- `files/` - Directory for model files (not included)
  - `ncf_model.keras` - Trained model
  - `user_encoder.joblib` - User ID encoder
  - `movie_encoder.joblib` - Movie ID encoder
  - `scaler.joblib` - Rating scaler

## Dataset

The system requires two CSV files in the `dataset` directory:
- `movies.csv`: Contains movie information
  - Columns: movieId, title, genres
- `ratings.csv`: Contains user ratings
  - Columns: userId, movieId, rating, timestamp

## Model Architecture

The recommendation system uses a Neural Collaborative Filtering approach with:
- Separate embedding layers for users and movies
- Dense layers with ReLU activation
- Dropout layers for regularization
- Custom embedding selection layer for handling new users
- Optimization using Adam optimizer

## API Endpoints

### GET /api/search
Search for movies by title.
- Query parameters:
  - `q`: Search query string
- Returns up to 10 matching movies

### POST /api/recommendations
Generate movie recommendations.
- Request body:
  ```json
  {
    "ratings": [
      {
        "movieId": 1,
        "rating": 4.5,
        "title": "Movie Title"
      }
    ],
    "diverse": true
  }
  ```
- Returns top 10 recommended movies with predicted ratings

## Running the Application

1. Prepare your dataset:
   - Place `movies.csv` and `ratings.csv` in the `dataset` directory

2. Train the model:
   ```bash
   python script.py
   ```
   This will create the necessary model files in the `files` directory.

3. Start the Flask server:
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`

## Features

- **Real-time Embedding Optimization**: Optimizes user embeddings based on provided ratings
- **Diverse Recommendations**: Optional diversity-aware recommendation algorithm considering:
  - Genre distribution
  - Release decade
  - Movie franchises
  - User rating history
- **Early Stopping**: Implements early stopping during user embedding optimization
- **Batch Processing**: Handles large-scale predictions efficiently using batch processing
- **Error Handling**: Comprehensive error handling and logging throughout the application

## Notes

- The model and dataset files are not included in the repository due to size constraints
- The system requires initial training data to generate the model and encoder files
- Performance may vary based on the size and quality of the training dataset
- The application includes CORS support for cross-origin requests

## Error Handling

The application includes comprehensive error handling and logging:
- All errors are caught and logged with appropriate context
- API endpoints return proper error responses with status codes
- Logging includes timestamps and error levels for debugging

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.