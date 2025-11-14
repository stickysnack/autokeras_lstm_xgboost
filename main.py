import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AutoKerasModelComparison:
    def __init__(self, train_path, test_path):
        print("="*70)
        print("Loading Data...")
        print("="*70)
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"✓ Training data loaded: {self.train_df.shape}")
        print(f"  - Date range: {self.train_df['date_id'].min()} to {self.train_df['date_id'].max()}")
        print(f"  - Contains target 'forward_returns': {self.train_df['forward_returns'].notna().sum()} non-null values")
        
        print(f"\n✓ Test data loaded: {self.test_df.shape}")
        print(f"  - Date range: {self.test_df['date_id'].min()} to {self.test_df['date_id'].max()}")
        if 'is_scored' in self.test_df.columns:
            scored_count = self.test_df['is_scored'].sum()
            print(f"  - Rows to be scored: {scored_count}")
        
        self.scaler_features = StandardScaler()
        self.scaler_lagged = StandardScaler()
        self.lstm_model = None
        self.xgb_model = None
        self.best_lstm_params = None
        self.best_xgb_params = None
        self.feature_cols = None
        self.lagged_feature_cols = None
        
    def preprocess_data(self, lookback=10):
        """Preprocess the financial data correctly"""
        print("\n" + "="*70)
        print("Data Preprocessing...")
        print("="*70)
        
        # === TRAINING DATA ===
        target = 'forward_returns'
        
        # Remove rows where target is missing
        train_clean = self.train_df.dropna(subset=[target]).copy()
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns']
        self.feature_cols = [col for col in train_clean.columns if col not in exclude_cols]
        
        print(f"\n📊 Training Set Features:")
        print(f"  - Total features: {len(self.feature_cols)}")
        print(f"  - Feature categories: D (Dummy), E (Economic), I (Interest), M (Market), P (Price), S (Sentiment), V (Volatility)")
        
        # Fill missing values with column median
        for col in self.feature_cols:
            median_val = train_clean[col].median()
            train_clean[col].fillna(median_val, inplace=True)
        
        # Sort by date_id to ensure temporal order
        train_clean = train_clean.sort_values('date_id').reset_index(drop=True)
        
        # Split features and target
        X_train_full = train_clean[self.feature_cols].values
        y_train_full = train_clean[target].values
        
        # Time series split: use last 20% for validation
        split_idx = int(len(X_train_full) * 0.8)
        X_train, X_val = X_train_full[:split_idx], X_train_full[split_idx:]
        y_train, y_val = y_train_full[:split_idx], y_train_full[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_val_scaled = self.scaler_features.transform(X_val)
        
        print(f"\n📈 Training/Validation Split:")
        print(f"  - Training samples: {len(X_train)} (80%)")
        print(f"  - Validation samples: {len(X_val)} (20%)")
        
        # Create sequences for LSTM
        X_train_lstm, y_train_lstm = self.create_sequences(X_train_scaled, y_train, lookback)
        X_val_lstm, y_val_lstm = self.create_sequences(X_val_scaled, y_val, lookback)
        
        print(f"  - LSTM sequences (lookback={lookback}): {X_train_lstm.shape}")
        
        # === TEST DATA PREPARATION ===
        test_clean = self.test_df.copy()
        
        # Get test feature columns (same as training, excluding lagged columns)
        available_features = [col for col in self.feature_cols if col in test_clean.columns]
        missing_features = set(self.feature_cols) - set(available_features)
        
        if missing_features:
            print(f"\n⚠️  Warning: {len(missing_features)} features missing in test set")
            print(f"  Missing: {list(missing_features)[:5]}..." if len(missing_features) > 5 else f"  Missing: {list(missing_features)}")
        
        # Check for lagged features in test set
        lagged_cols = [col for col in test_clean.columns if col.startswith('lagged_')]
        if lagged_cols:
            print(f"\n🔄 Test set contains lagged features: {lagged_cols}")
            print("  These provide information from previous day for prediction")
        
        # Fill missing values in test set
        for col in available_features:
            median_val = test_clean[col].median() if test_clean[col].notna().any() else 0
            test_clean[col].fillna(median_val, inplace=True)
        
        # Add missing features as zeros (if any)
        for col in missing_features:
            test_clean[col] = 0
        
        # Ensure column order matches training
        X_test = test_clean[self.feature_cols].values
        
        # Scale test features
        X_test_scaled = self.scaler_features.transform(X_test)
        
        # Create sequences for LSTM test data
        # For test set, we need to handle the sequence creation differently
        # since we don't have future targets
        if len(X_test_scaled) >= lookback:
            X_test_lstm = []
            for i in range(lookback - 1, len(X_test_scaled)):
                X_test_lstm.append(X_test_scaled[i-lookback+1:i+1])
            X_test_lstm = np.array(X_test_lstm)
            test_indices = np.arange(lookback - 1, len(X_test_scaled))
        else:
            # If test set is smaller than lookback, pad with training data
            combined = np.vstack([X_train_scaled[-lookback:], X_test_scaled])
            X_test_lstm = []
            for i in range(lookback - 1, len(combined)):
                X_test_lstm.append(combined[i-lookback+1:i+1])
            X_test_lstm = np.array(X_test_lstm)
            test_indices = np.arange(len(X_test_scaled))
        
        print(f"\n🎯 Test Set Prepared:")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - LSTM test sequences: {X_test_lstm.shape}")
        if 'is_scored' in test_clean.columns:
            print(f"  - Rows to be scored: {test_clean['is_scored'].sum()}")
        
        return {
            # Training data
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'X_train_lstm': X_train_lstm,
            'X_val_lstm': X_val_lstm,
            'y_train_lstm': y_train_lstm,
            'y_val_lstm': y_val_lstm,
            # Test data
            'X_test': X_test_scaled,
            'X_test_lstm': X_test_lstm,
            'test_indices': test_indices,
            'test_df': test_clean,
            # Metadata
            'feature_cols': self.feature_cols
        }
    
    def create_sequences(self, X, y, lookback):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def train_autokeras_lstm(self, data, max_trials=20, epochs=50):
        """Train LSTM using AutoKeras for automatic hyperparameter tuning"""
        print("\n" + "="*70)
        print("🤖 Training LSTM Model with AutoKeras AutoML")
        print("="*70)
        print(f"Configuration: max_trials={max_trials}, epochs={epochs}")
        print("AutoKeras will search for optimal LSTM architecture...")
        print("This may take several minutes...\n")
        
        # Create AutoKeras TimeseriesForecaster
        lstm_auto = ak.TimeseriesForecaster(
            lookback=data['X_train_lstm'].shape[1],
            predict_from=1,
            predict_until=1,
            max_trials=max_trials,
            loss='mse',
            metrics=['mae'],
            objective='val_loss',
            directory='autokeras_lstm',
            project_name='sp500_lstm',
            overwrite=True,
            seed=42
        )
        
        # Reshape data for AutoKeras (batch, features, timesteps)
        X_train_ak = np.transpose(data['X_train_lstm'], (0, 2, 1))
        X_val_ak = np.transpose(data['X_val_lstm'], (0, 2, 1))
        
        # Train the model
        lstm_auto.fit(
            X_train_ak,
            data['y_train_lstm'],
            validation_data=(X_val_ak, data['y_val_lstm']),
            epochs=epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        )
        
        # Get the best model
        self.lstm_model = lstm_auto.export_model()
        
        # Get best hyperparameters
        best_trial = lstm_auto.tuner.oracle.get_best_trials(1)[0]
        self.best_lstm_params = best_trial.hyperparameters.values
        
        print("\n✅ Best LSTM model found!")
        print(f"   Best validation loss: {best_trial.score:.6f}")
        print(f"   Best hyperparameters: {self.best_lstm_params}")
        
        return lstm_auto
    
    def train_autokeras_xgboost(self, data, n_trials=30):
        """Train XGBoost using automated hyperparameter search"""
        print("\n" + "="*70)
        print("🤖 Training XGBoost Model with Automated Search")
        print("="*70)
        print(f"Running {n_trials} trials to find optimal configuration...")
        print("This may take several minutes...\n")
        
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, randint
        
        # Define hyperparameter search space
        param_distributions = {
            'max_depth': randint(3, 11),
            'learning_rate': uniform(0.01, 0.29),
            'n_estimators': randint(100, 1001),
            'min_child_weight': randint(1, 11),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 5),
            'reg_alpha': uniform(0, 5),
            'reg_lambda': uniform(0, 5)
        }
        
        # Create base model
        xgb_base = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Random search with cross-validation
        random_search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_distributions,
            n_iter=n_trials,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit
        random_search.fit(data['X_train'], data['y_train'])
        
        # Get best model and parameters
        self.xgb_model = random_search.best_estimator_
        self.best_xgb_params = random_search.best_params_
        
        print("\n✅ Best XGBoost model found!")
        print(f"   Best validation MSE: {-random_search.best_score_:.6f}")
        print(f"   Best hyperparameters:")
        for param, value in self.best_xgb_params.items():
            print(f"     {param}: {value}")
        
        return random_search
    
    def evaluate_models(self, data):
        """Evaluate both models on validation and test sets"""
        print("\n" + "="*70)
        print("📊 Model Evaluation Results")
        print("="*70)
        
        # === VALIDATION SET EVALUATION ===
        print("\n1️⃣  VALIDATION SET PERFORMANCE:")
        print("-" * 70)
        
        # LSTM predictions on validation
        X_val_lstm_reshaped = np.transpose(data['X_val_lstm'], (0, 2, 1))
        lstm_val_preds = self.lstm_model.predict(X_val_lstm_reshaped, verbose=0).flatten()
        
        # XGBoost predictions on validation
        xgb_val_preds = self.xgb_model.predict(data['X_val'])
        
        # Calculate validation metrics
        lstm_val_mse = mean_squared_error(data['y_val_lstm'], lstm_val_preds)
        lstm_val_mae = mean_absolute_error(data['y_val_lstm'], lstm_val_preds)
        lstm_val_rmse = np.sqrt(lstm_val_mse)
        lstm_val_r2 = r2_score(data['y_val_lstm'], lstm_val_preds)
        
        xgb_val_mse = mean_squared_error(data['y_val'], xgb_val_preds)
        xgb_val_mae = mean_absolute_error(data['y_val'], xgb_val_preds)
        xgb_val_rmse = np.sqrt(xgb_val_mse)
        xgb_val_r2 = r2_score(data['y_val'], xgb_val_preds)
        
        val_results_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
            'AutoKeras LSTM': [f'{lstm_val_mse:.6f}', f'{lstm_val_rmse:.6f}', 
                              f'{lstm_val_mae:.6f}', f'{lstm_val_r2:.6f}'],
            'Optimized XGBoost': [f'{xgb_val_mse:.6f}', f'{xgb_val_rmse:.6f}', 
                                 f'{xgb_val_mae:.6f}', f'{xgb_val_r2:.6f}']
        })
        print("\n" + val_results_df.to_string(index=False))
        
        # === TEST SET PREDICTIONS ===
        print("\n\n2️⃣  TEST SET PREDICTIONS:")
        print("-" * 70)
        
        # LSTM predictions on test
        X_test_lstm_reshaped = np.transpose(data['X_test_lstm'], (0, 2, 1))
        lstm_test_preds = self.lstm_model.predict(X_test_lstm_reshaped, verbose=0).flatten()
        
        # XGBoost predictions on test
        xgb_test_preds = self.xgb_model.predict(data['X_test'])
        
        print(f"✓ Generated {len(lstm_test_preds)} LSTM predictions")
        print(f"✓ Generated {len(xgb_test_preds)} XGBoost predictions")
        
        # Create submission dataframes
        test_df = data['test_df'].copy()
        
        # For LSTM, align predictions with correct indices
        lstm_submission = test_df[['date_id']].copy()
        lstm_submission['forward_returns_pred'] = np.nan
        lstm_submission.loc[data['test_indices'], 'forward_returns_pred'] = lstm_test_preds
        
        # For XGBoost
        xgb_submission = test_df[['date_id']].copy()
        xgb_submission['forward_returns_pred'] = xgb_test_preds
        
        if 'is_scored' in test_df.columns:
            scored_mask = test_df['is_scored'] == True
            print(f"\n📝 Predictions for scored rows:")
            print(f"   LSTM - Scored predictions: {lstm_submission.loc[scored_mask, 'forward_returns_pred'].notna().sum()}")
            print(f"   XGBoost - Scored predictions: {xgb_submission.loc[scored_mask, 'forward_returns_pred'].notna().sum()}")
            
            print(f"\n   LSTM prediction stats (scored rows):")
            print(f"     Mean: {lstm_submission.loc[scored_mask, 'forward_returns_pred'].mean():.6f}")
            print(f"     Std:  {lstm_submission.loc[scored_mask, 'forward_returns_pred'].std():.6f}")
            
            print(f"\n   XGBoost prediction stats (scored rows):")
            print(f"     Mean: {xgb_submission.loc[scored_mask, 'forward_returns_pred'].mean():.6f}")
            print(f"     Std:  {xgb_submission.loc[scored_mask, 'forward_returns_pred'].std():.6f}")
        
        # Determine winner based on validation performance
        print("\n" + "="*70)
        if lstm_val_mse < xgb_val_mse:
            print("🏆 AutoKeras LSTM is the WINNER (Validation Set)!")
            improvement = ((xgb_val_mse - lstm_val_mse) / xgb_val_mse) * 100
            print(f"   LSTM outperforms XGBoost by {improvement:.2f}%")
        else:
            print("🏆 Optimized XGBoost is the WINNER (Validation Set)!")
            improvement = ((lstm_val_mse - xgb_val_mse) / lstm_val_mse) * 100
            print(f"   XGBoost outperforms LSTM by {improvement:.2f}%")
        print("="*70)
        
        return {
            'validation': {
                'lstm': {'mse': lstm_val_mse, 'rmse': lstm_val_rmse, 'mae': lstm_val_mae, 'r2': lstm_val_r2,
                        'predictions': lstm_val_preds, 'actuals': data['y_val_lstm']},
                'xgb': {'mse': xgb_val_mse, 'rmse': xgb_val_rmse, 'mae': xgb_val_mae, 'r2': xgb_val_r2,
                       'predictions': xgb_val_preds, 'actuals': data['y_val']}
            },
            'test': {
                'lstm_submission': lstm_submission,
                'xgb_submission': xgb_submission,
                'lstm_preds': lstm_test_preds,
                'xgb_preds': xgb_test_preds
            }
        }
    
    def plot_results(self, results):
        """Plot comprehensive comparison visualizations"""
        val_results = results['validation']
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Predictions vs Actuals - LSTM
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(val_results['lstm']['actuals'], val_results['lstm']['predictions'], 
                   alpha=0.5, s=30, c='blue', edgecolors='navy', linewidths=0.5)
        ax1.plot([val_results['lstm']['actuals'].min(), val_results['lstm']['actuals'].max()],
                [val_results['lstm']['actuals'].min(), val_results['lstm']['actuals'].max()],
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Returns', fontsize=10)
        ax1.set_ylabel('Predicted Returns', fontsize=10)
        ax1.set_title(f'AutoKeras LSTM (Validation)\nR²={val_results["lstm"]["r2"]:.4f}', 
                     fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Predictions vs Actuals - XGBoost
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(val_results['xgb']['actuals'], val_results['xgb']['predictions'], 
                   alpha=0.5, s=30, c='green', edgecolors='darkgreen', linewidths=0.5)
        ax2.plot([val_results['xgb']['actuals'].min(), val_results['xgb']['actuals'].max()],
                [val_results['xgb']['actuals'].min(), val_results['xgb']['actuals'].max()],
                'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Returns', fontsize=10)
        ax2.set_ylabel('Predicted Returns', fontsize=10)
        ax2.set_title(f'Optimized XGBoost (Validation)\nR²={val_results["xgb"]["r2"]:.4f}', 
                     fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Metrics Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        metrics = ['MSE', 'RMSE', 'MAE']
        lstm_vals = [val_results['lstm']['mse'], val_results['lstm']['rmse'], val_results['lstm']['mae']]
        xgb_vals = [val_results['xgb']['mse'], val_results['xgb']['rmse'], val_results['xgb']['mae']]
        
        x = np.arange(len(metrics))
        width = 0.35
        bars1 = ax3.bar(x - width/2, lstm_vals, width, label='LSTM', 
                       alpha=0.8, color='blue', edgecolor='navy')
        bars2 = ax3.bar(x + width/2, xgb_vals, width, label='XGBoost', 
                       alpha=0.8, color='green', edgecolor='darkgreen')
        ax3.set_ylabel('Error Value', fontsize=10)
        ax3.set_title('Validation Performance', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.5f}', ha='center', va='bottom', fontsize=7)
        
        # 4-6. Residuals
        ax4 = fig.add_subplot(gs[1, 0])
        lstm_residuals = val_results['lstm']['actuals'] - val_results['lstm']['predictions']
        ax4.hist(lstm_residuals, bins=50, alpha=0.7, color='blue', edgecolor='navy', density=True)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Residuals', fontsize=10)
        ax4.set_ylabel('Density', fontsize=10)
        ax4.set_title(f'LSTM Residuals\nMean={np.mean(lstm_residuals):.6f}', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        xgb_residuals = val_results['xgb']['actuals'] - val_results['xgb']['predictions']
        ax5.hist(xgb_residuals, bins=50, alpha=0.7, color='green', edgecolor='darkgreen', density=True)
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax5.set_xlabel('Residuals', fontsize=10)
        ax5.set_ylabel('Density', fontsize=10)
        ax5.set_title(f'XGBoost Residuals\nMean={np.mean(xgb_residuals):.6f}', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(lstm_residuals, bins=50, alpha=0.5, label='LSTM', color='blue', density=True)
        ax6.hist(xgb_residuals, bins=50, alpha=0.5, label='XGBoost', color='green', density=True)
        ax6.set_xlabel('Residuals', fontsize=10)
        ax6.set_ylabel('Density', fontsize=10)
        ax6.set_title('Residuals Comparison', fontsize=11, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Time Series Predictions
        ax7 = fig.add_subplot(gs[2, :2])
        sample_size = min(200, len(val_results['lstm']['actuals']))
        indices = np.arange(sample_size)
        ax7.plot(indices, val_results['lstm']['actuals'][:sample_size], 
                'o-', label='Actual', alpha=0.7, markersize=3, linewidth=1)
        ax7.plot(indices, val_results['lstm']['predictions'][:sample_size], 
                's-', label='LSTM', alpha=0.7, markersize=3, linewidth=1)
        ax7.plot(indices, val_results['xgb']['predictions'][:sample_size], 
                '^-', label='XGBoost', alpha=0.7, markersize=3, linewidth=1)
        ax7.set_xlabel('Sample Index', fontsize=10)
        ax7.set_ylabel('Returns', fontsize=10)
        ax7.set_title('Validation: Predictions vs Actuals (First 200 samples)', fontsize=11, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary Table
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        summary_data = [
            ['Metric', 'LSTM', 'XGBoost', 'Winner'],
            ['MSE', f"{val_results['lstm']['mse']:.6f}", 
             f"{val_results['xgb']['mse']:.6f}",
             '🏆' if val_results['lstm']['mse'] < val_results['xgb']['mse'] else ''],
            ['RMSE', f"{val_results['lstm']['rmse']:.6f}", 
             f"{val_results['xgb']['rmse']:.6f}",
             '🏆' if val_results['lstm']['rmse'] < val_results['xgb']['rmse'] else ''],
            ['MAE', f"{val_results['lstm']['mae']:.6f}", 
             f"{val_results['xgb']['mae']:.6f}",
             '🏆' if val_results['lstm']['mae'] < val_results['xgb']['mae'] else ''],
            ['R²', f"{val_results['lstm']['r2']:.6f}", 
             f"{val_results['xgb']['r2']:.6f}",
             '🏆' if val_results['lstm']['r2'] > val_results['xgb']['r2'] else '']
        ]
        
        table = ax8.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, 5):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax8.set_title('Performance Summary', fontsize=11, fontweight='bold', pad=20)
        
        plt.suptitle('AutoKeras LSTM vs Optimized XGBoost: Validation Results', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.savefig('autokeras_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'autokeras_model_comparison.png'")
        plt.show()
    
    def save_models_and_predictions(self, results):
        """Save trained models and predictions"""
        print("\n" + "="*70)
        print("💾 Saving Models and Predictions...")
        print("="*70)
        
        # Save LSTM model
        self.lstm_model.save('best_lstm_model.keras')
        print("✓ LSTM model saved as 'best_lstm_model.keras'")
        
        # Save XGBoost model
        self.xgb_model.save_model('best_xgboost_model.json')
        print("✓ XGBoost model saved as 'best_xgboost_model.json'")
        
        # Save hyperparameters
        import json
        params = {
            'lstm': self.best_lstm_params,
            'xgboost': self.best_xgb_params
        }
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(params, f, indent=4, default=str)
        print("✓ Hyperparameters saved as 'best_hyperparameters.json'")
        
        # Save test predictions
        lstm_sub = results['test']['lstm_submission']
        xgb_sub = results['test']['xgb_submission']
        
        lstm_sub.to_csv('lstm_test_predictions.csv', index=False)
        print("✓ LSTM test predictions saved as 'lstm_test_predictions.csv'")
        
        xgb_sub.to_csv('xgboost_test_predictions.csv', index=False)
        print("✓ XGBoost test predictions saved as 'xgboost_test_predictions.csv'")
        
        # Create a combined comparison file
        comparison_df = lstm_sub.copy()
        comparison_df = comparison_df.rename(columns={'forward_returns_pred': 'lstm_pred'})
        comparison_df['xgboost_pred'] = xgb_sub['forward_returns_pred']
        comparison_df['pred_difference'] = comparison_df['lstm_pred'] - comparison_df['xgboost_pred']
        
        comparison_df.to_csv('combined_predictions_comparison.csv', index=False)
        print("✓ Combined predictions saved as 'combined_predictions_comparison.csv'")
        
        print("\n" + "="*70)


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("S&P 500 Returns Prediction with AutoKeras")
    print("AutoML Comparison: LSTM vs XGBoost")
    print("="*70)
    
    # Initialize and load data
    comparison = AutoKerasModelComparison(
        train_path='train.csv',
        test_path='test.csv'
    )
    
    # Preprocess data (including test set)
    data = comparison.preprocess_data(lookback=10)
    
    # Train models with AutoKeras automatic hyperparameter tuning
    print("\n🤖 Starting AutoKeras AutoML Process...")
    print("This will automatically search for optimal architectures and hyperparameters")
    
    # Train LSTM with AutoKeras
    print("\n[1/2] Training LSTM model...")
    lstm_auto = comparison.train_autokeras_lstm(data, max_trials=20, epochs=50)
    
    # Train XGBoost with automated search
    print("\n[2/2] Training XGBoost model...")
    xgb_search = comparison.train_autokeras_xgboost(data, n_trials=30)
    
    # Evaluate on validation set and make test predictions
    results = comparison.evaluate_models(data)
    
    # Visualize validation results
    comparison.plot_results(results)
    
    # Save everything
    comparison.save_models_and_predictions(results)
    
    print("\n" + "="*70)
    print("✅ AutoKeras Analysis Complete!")
    print("="*70)
    print("\n📁 Generated Files:")
    print("  📊 autokeras_model_comparison.png - Comprehensive visualizations")
    print("  💾 best_lstm_model.keras - Trained LSTM model")
    print("  💾 best_xgboost_model.json - Trained XGBoost model")
    print("  📝 best_hyperparameters.json - Optimal hyperparameters")
    print("  📈 lstm_test_predictions.csv - LSTM predictions on test set")
    print("  📈 xgboost_test_predictions.csv - XGBoost predictions on test set")
    print("  📊 combined_predictions_comparison.csv - Side-by-side comparison")
    print("\n" + "="*70)
    print("\n💡 Next Steps:")
    print("  1. Review validation metrics to choose best model")
    print("  2. Check test predictions in CSV files")
    print("  3. Submit predictions to Kaggle competition")
    print("  4. The 'is_scored' rows in test set are what will be evaluated")
    print("\n" + "="*70)