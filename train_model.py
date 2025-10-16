import hopsworks
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
from datetime import datetime
import tensorflow as tf 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_pipeline():
    try:
        # Connect to Hopsworks
        logging.info("Connecting to Hopsworks...")
        HOPSWORKS_API_KEY = "DOXxlrr308Rq2xqN.QmlA3Cfoy8ljM9h8nOiYYpxHA3EoSPGhp9qPBcONsXHRL7XIpsGjbcc80R3OoCz5"
        HOPSWORKS_PROJECT = "AQI_Project_10"
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        
        # Fetch feature group with data validation
        logging.info("Fetching feature group...")
        fg = fs.get_feature_group(name="aqi_features", version=1)
        df = fg.read()
        
        # Validate data
        if df is None or df.empty:
            raise ValueError("No data retrieved from feature store")
            
        logging.info(f"Data shape: {df.shape}")
        logging.info("Sample of retrieved data:")
        logging.info(df.head())
        
        # Data preparation with validation
        drop_cols = ['timestamp', 'datetime', 'station_name', 'station_url', 'data_quality', 'time_of_day']
        feature_cols = [col for col in df.columns if col not in drop_cols and col != 'aqi']
        
        if not feature_cols:
            raise ValueError("No valid feature columns found")
            
        X = df[feature_cols]
        y = df['aqi']
        
        if len(X) < 2:
            raise ValueError(f"Insufficient data for training. Found only {len(X)} samples")
            
        logging.info(f"Features being used: {feature_cols}")
        logging.info(f"Number of samples: {len(X)}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Ensure minimum data for splitting
        min_samples = 10  # Set minimum required samples
        if len(X) < min_samples:
            raise ValueError(f"Need at least {min_samples} samples for training. Found {len(X)}")
            
        # Train XGBoost with validation
        logging.info("\nTraining XGBoost model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Train-test split resulted in empty sets")
            
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )
        xgb_model.fit(X_train, y_train, verbose=True)
        xgb_pred = xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_r2 = r2_score(y_test, xgb_pred)
        logging.info(f"XGBoost Results - RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.4f}")

        # Train LSTM with dynamic sequence length
        logging.info("\nTraining LSTM model...")
        
        # Dynamically adjust sequence length based on data size
        seq_len = min(12, len(X_scaled) // 3)  # Use smaller sequence length for small datasets
        logging.info(f"Using sequence length: {seq_len}")
        
        # Check if we have enough data for sequences
        if len(X_scaled) <= seq_len:
            logging.warning("Not enough data for LSTM sequence creation. Skipping LSTM training.")
            logging.info("\nFinal Results:")
            logging.info("-" * 50)
            logging.info(f"XGBoost - RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.4f}")
            logging.info("LSTM - Not trained (insufficient data)")
            
            # Save XGBoost model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/xgboost_model_{timestamp}.json"
            xgb_model.save_model(model_path)
            logging.info(f"Saved XGBoost model to {model_path}")
            return
        
        # Create sequences only if we have enough data
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_len):
            X_seq.append(X_scaled[i:i+seq_len])
            y_seq.append(y.iloc[i+seq_len])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Verify sequence creation
        if len(X_seq) == 0:
            logging.warning("No sequences could be created. Skipping LSTM training.")
            return
            
        logging.info(f"Created sequences - Shape: {X_seq.shape}")
        
        # Split with minimum validation
        val_size = min(0.2, 1/len(X_seq))  # Adjust validation size for small datasets
        X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
            X_seq, y_seq, 
            test_size=val_size,
            random_state=42,
            shuffle=False  # Keep time order for sequences
        )
        
        # Adjust LSTM architecture for small datasets
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(seq_len, X_scaled.shape[1])),
            Dropout(0.1),
            LSTM(16),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Reduce epochs and batch size for small datasets
        batch_size = min(8, len(X_seq_train) // 2)
        model.fit(
            X_seq_train, 
            y_seq_train, 
            epochs=20,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        lstm_pred = model.predict(X_seq_test)
        lstm_rmse = np.sqrt(mean_squared_error(y_seq_test, lstm_pred))
        lstm_r2 = r2_score(y_seq_test, lstm_pred)
        logging.info(f"LSTM Results - RMSE: {lstm_rmse:.2f}, R²: {lstm_r2:.4f}")

        # Model comparison and saving
        logging.info("\nModel Comparison:")
        logging.info("-" * 50)
        logging.info(f"XGBoost - RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.4f}")
        logging.info(f"LSTM    - RMSE: {lstm_rmse:.2f}, R²: {lstm_r2:.4f}")
        
        # Save models with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if xgb_r2 > lstm_r2:
            model_path = f"models/xgboost_model_{timestamp}.json"
            xgb_model.save_model(model_path)
            logging.info(f"Saved XGBoost model (better R²) to {model_path}")
        else:
            model_path = f"models/lstm_model_{timestamp}.h5"
            model.save(model_path)
            logging.info(f"Saved LSTM model (better R²) to {model_path}")

    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        logging.error(f"Data shape: {X_scaled.shape if 'X_scaled' in locals() else 'Not available'}")
        raise

if __name__ == "__main__":
    logging.info("Starting AQI prediction model training pipeline...")
    train_pipeline()
    logging.info("Training pipeline completed!")