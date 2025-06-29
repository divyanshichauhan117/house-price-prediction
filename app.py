from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import json

app = Flask(__name__)

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features_to_use = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'house_age', 'renovated', 'lat', 'long'
        ]
        self.train_model()
    
    def create_sample_data(self):
        """Create synthetic dataset for training"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 5, n_samples),
            'sqft_living': np.random.randint(500, 5000, n_samples),
            'sqft_lot': np.random.randint(1000, 20000, n_samples),
            'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'view': np.random.randint(0, 5, n_samples),
            'condition': np.random.randint(1, 6, n_samples),
            'grade': np.random.randint(3, 13, n_samples),
            'sqft_above': np.random.randint(400, 4000, n_samples),
            'sqft_basement': np.random.randint(0, 2000, n_samples),
            'yr_built': np.random.randint(1900, 2021, n_samples),
            'yr_renovated': np.random.choice([0] + list(range(1950, 2021)), n_samples, p=[0.7] + [0.3/71]*71),
            'zipcode': np.random.choice([98001, 98002, 98003, 98004, 98005, 98006], n_samples),
            'lat': np.random.uniform(47.15, 47.75, n_samples),
            'long': np.random.uniform(-122.5, -121.3, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate realistic prices based on features
        price = (
            df['sqft_living'] * 150 +
            df['bedrooms'] * 10000 +
            df['bathrooms'] * 15000 +
            df['waterfront'] * 200000 +
            df['view'] * 20000 +
            df['grade'] * 25000 +
            (2021 - df['yr_built']) * (-500) +
            np.random.normal(0, 50000, n_samples)
        )
        
        df['price'] = np.maximum(price, 50000)
        return df
    
    def train_model(self):
        """Train the Random Forest model"""
        # Create sample data
        df = self.create_sample_data()
        
        # Feature engineering
        df['house_age'] = 2021 - df['yr_built']
        df['renovated'] = (df['yr_renovated'] > 0).astype(int)
        
        # Prepare features and target
        X = df[self.features_to_use]
        y = df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Train scaler (for consistency, though RF doesn't need it)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        
        print("Model trained successfully!")
    
    def predict_price(self, house_features):
        """Predict house price given features"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert to DataFrame
        features_df = pd.DataFrame([house_features], columns=self.features_to_use)
        
        # Make prediction
        prediction = self.model.predict(features_df)[0]
        
        return max(prediction, 50000)  # Minimum price $50k
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return {}
        
        importance_dict = dict(zip(self.features_to_use, self.model.feature_importances_))
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_features

# Initialize the predictor
predictor = HousePricePredictor()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from form
        data = request.get_json()
        
        # Extract features in the correct order
        features = [
            int(data['bedrooms']),
            int(data['bathrooms']),
            int(data['sqft_living']),
            int(data['sqft_lot']),
            float(data['floors']),
            int(data['waterfront']),
            int(data['view']),
            int(data['condition']),
            int(data['grade']),
            int(data['sqft_above']),
            int(data['sqft_basement']),
            int(data['house_age']),
            int(data['renovated']),
            float(data['lat']),
            float(data['long'])
        ]
        
        # Make prediction
        predicted_price = predictor.predict_price(features)
        
        # Calculate price per sqft
        price_per_sqft = predicted_price / int(data['sqft_living'])
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'formatted_price': f"${predicted_price:,.0f}",
            'price_per_sqft': round(price_per_sqft, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/feature_importance')
def feature_importance():
    """Return feature importance data"""
    try:
        importance = predictor.get_feature_importance()
        return jsonify({
            'success': True,
            'features': importance
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/sample_predictions')
def sample_predictions():
    """Generate sample predictions for demonstration"""
    try:
        sample_houses = [
            {
                'description': 'Modest Family Home',
                'features': [3, 2, 1800, 7000, 1.5, 0, 2, 4, 7, 1800, 0, 20, 0, 47.5, -122.2]
            },
            {
                'description': 'Luxury Waterfront Property',
                'features': [4, 3, 2500, 9000, 2, 1, 4, 5, 9, 2200, 300, 15, 1, 47.6, -122.1]
            },
            {
                'description': 'Starter Home',
                'features': [2, 1, 1200, 5000, 1, 0, 1, 3, 6, 1200, 0, 30, 0, 47.4, -122.3]
            }
        ]
        
        predictions = []
        for house in sample_houses:
            price = predictor.predict_price(house['features'])
            predictions.append({
                'description': house['description'],
                'predicted_price': round(price, 2),
                'formatted_price': f"${price:,.0f}"
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=5000)