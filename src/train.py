import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from . import config, preprocessing

# Toggle target transform to handle heavy tails. Set to True to enable signed-log transform.
TARGET_TRANSFORM = True

def signed_log_transform(y):
    # y: pandas Series or numpy array
    sign = np.sign(y)
    return sign * np.log1p(np.abs(y))

def signed_log_inverse(y_t):
    sign = np.sign(y_t)
    return sign * (np.expm1(np.abs(y_t)))

def train_model():
    # Get clean data
    df = preprocessing.load_and_preprocess()
    print(f"Data loaded, samples: {len(df)}")
    
    # Calculate features such as lags, rolling means etc.
    features = ['Data_Lag1', 'Data_Change_PrevDay', 'Data_Rolling_Mean']
    target = 'Price_Change'

    X = df[features]
    y = df[target]

    # Chronological (time-series) split: first 80% train, last 20% test
    n = len(df)
    split_idx = int(n * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Scale features (fit on train, apply to both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optionally transform target to reduce impact of heavy tails
    if TARGET_TRANSFORM:
        y_train_used = signed_log_transform(y_train.values)
        y_test_used = signed_log_transform(y_test.values)
    else:
        y_train_used = y_train.values
        y_test_used = y_test.values

    # Train model
    print("Training the random forest model")
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train_scaled, y_train_used)

    # Predict
    y_pred_used = model.predict(X_test_scaled)

    # If transformed, inverse before computing metrics and plotting
    if TARGET_TRANSFORM:
        y_pred = signed_log_inverse(y_pred_used)
        y_test_vals = y_test.values
    else:
        y_pred = y_pred_used
        y_test_vals = y_test.values

    mse = mean_squared_error(y_test_vals, y_pred)
    r2 = r2_score(y_test_vals, y_pred)
    print(f"Evaluation results -> MSE: {mse:.6f}, R2: {r2:.4f}")

    # Save model and scaler
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)
    joblib.dump(scaler, os.path.join(config.MODELS_DIR, 'scaler.pkl'))
    print(f"Model saved to {config.MODEL_FILE} and scaler saved to {os.path.join(config.MODELS_DIR, 'scaler.pkl')}")

    # Save metrics and basic metadata
    metrics = {
        'mse': float(mse),
        'r2': float(r2),
        'n_samples': int(len(df)),
        'features': features,
        'target_transform': bool(TARGET_TRANSFORM)
    }
    with open(os.path.join(config.MODELS_DIR, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    # Plot: Actual vs Predicted (for test set)
    try:
        sns.set(style='whitegrid')
        # Prediction plot (sorted by actual)
        plt.figure(figsize=(10,6))
        order = np.argsort(y_test_vals)
        plt.plot(y_test_vals[order], label='Actual', marker='o')
        plt.plot(np.array(y_pred)[order], label='Predicted', marker='x')
        plt.legend()
        plt.title('Actual vs Predicted Price Change (test set)')
        plt.xlabel('Sample (sorted by actual)')
        plt.ylabel('Price Change')
        plot_path = os.path.join(config.MODELS_DIR, 'prediction_plot.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved prediction plot to {plot_path}")

        # Feature importances
        try:
            fi = model.feature_importances_
            plt.figure(figsize=(8,4))
            sns.barplot(x=fi, y=features)
            plt.title('Feature Importances')
            fi_path = os.path.join(config.MODELS_DIR, 'feature_importance.png')
            plt.tight_layout()
            plt.savefig(fi_path)
            plt.close()
            print(f"Saved feature importance to {fi_path}")
        except Exception as e:
            print(f"Could not save feature importance: {e}")

        # Residuals scatter and histogram
        residuals = y_test_vals - np.array(y_pred)
        plt.figure(figsize=(8,4))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title('Residuals vs Predicted')
        resid_path = os.path.join(config.MODELS_DIR, 'residuals_scatter.png')
        plt.tight_layout()
        plt.savefig(resid_path)
        plt.close()
        print(f"Saved residuals scatter to {resid_path}")

        plt.figure(figsize=(8,4))
        sns.histplot(residuals, bins=50, kde=True)
        plt.title('Residuals Distribution')
        hist_path = os.path.join(config.MODELS_DIR, 'residuals_hist.png')
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved residuals histogram to {hist_path}")

        # Save test predictions and residuals for inspection
        out_df = X_test.copy()
        out_df['actual'] = y_test_vals
        out_df['predicted'] = list(y_pred)
        out_df['residual'] = residuals
        out_csv = os.path.join(config.MODELS_DIR, 'test_predictions.csv')
        out_df.to_csv(out_csv, index=True)
        print(f"Saved test predictions to {out_csv}")

    except Exception as e:
        print(f"Could not save plots or diagnostics: {e}")

        # --- Extended diagnostics: train an extended-features model and save additional plots ---
        try:
            ext_features = ['Data_Lag1','Data_Lag2','Data_Lag3','Data_Lag4','Data_Lag5',
                            'Data_Change_PrevDay','Data_Pct_Change','Data_Rolling_Mean','Data_Rolling_STD']
            X_ext = df[ext_features]
            y_ext = df[target]

            # Chronological split
            split_idx = int(len(df) * 0.8)
            X_ext_train = X_ext.iloc[:split_idx]
            X_ext_test = X_ext.iloc[split_idx:]
            y_ext_train = y_ext.iloc[:split_idx]
            y_ext_test = y_ext.iloc[split_idx:]

            scaler_ext = StandardScaler()
            X_ext_train_s = scaler_ext.fit_transform(X_ext_train)
            X_ext_test_s = scaler_ext.transform(X_ext_test)

            # Optionally transform target
            if TARGET_TRANSFORM:
                y_ext_train_used = signed_log_transform(y_ext_train.values)
            else:
                y_ext_train_used = y_ext_train.values

            model_ext = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=0)
            model_ext.fit(X_ext_train_s, y_ext_train_used)

            # Predict and inverse if needed
            y_ext_pred_used = model_ext.predict(X_ext_test_s)
            if TARGET_TRANSFORM:
                y_ext_pred = signed_log_inverse(y_ext_pred_used)
            else:
                y_ext_pred = y_ext_pred_used

            mse_ext = mean_squared_error(y_ext_test.values, y_ext_pred)
            r2_ext = r2_score(y_ext_test.values, y_ext_pred)
            metrics_ext = {'mse': float(mse_ext), 'r2': float(r2_ext), 'n_samples': len(df), 'features': ext_features}
            with open(os.path.join(config.MODELS_DIR,'metrics_extended.json'),'w') as fh:
                json.dump(metrics_ext, fh, indent=2)

            # Save extended model and scaler
            joblib.dump(model_ext, os.path.join(config.MODELS_DIR,'FF_Model_extended.pkl'))
            joblib.dump(scaler_ext, os.path.join(config.MODELS_DIR,'scaler_extended.pkl'))

            # Time-series plots: Data and Price
            plt.figure(figsize=(12,5))
            plt.plot(df['Date'], df['Data'], label='Data')
            plt.plot(df['Date'], df['Price'], label='Price')
            plt.legend()
            plt.title('Time Series: Data and Price')
            ts_path = os.path.join(config.MODELS_DIR,'timeseries_data_price.png')
            plt.tight_layout(); plt.savefig(ts_path); plt.close()

            # Correlation heatmap
            corr_df = df[ext_features + [target]].corr()
            plt.figure(figsize=(8,6))
            sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm')
            ch_path = os.path.join(config.MODELS_DIR,'correlation_heatmap.png')
            plt.tight_layout(); plt.savefig(ch_path); plt.close()

            # Scatter plots: feature vs target (subset)
            for f in ext_features:
                plt.figure(figsize=(6,4))
                plt.scatter(df[f], df[target], s=8, alpha=0.6)
                plt.xlabel(f); plt.ylabel(target)
                plt.title(f'Scatter: {f} vs {target}')
                spath = os.path.join(config.MODELS_DIR,f'scatter_{f}.png')
                plt.tight_layout(); plt.savefig(spath); plt.close()

            # QQ-plot of residuals
            import scipy.stats as stats
            residuals_ext = y_ext_test.values - np.array(y_ext_pred)
            plt.figure(figsize=(6,5))
            stats.probplot(residuals_ext, dist='norm', plot=plt)
            qq_path = os.path.join(config.MODELS_DIR,'qqplot_residuals.png')
            plt.tight_layout(); plt.savefig(qq_path); plt.close()

            print(f"Saved extended diagnostics and model. Extended MSE: {mse_ext:.6f}, R2: {r2_ext:.4f}")

        except Exception as e:
            print(f"Extended diagnostics failed: {e}")

if __name__ == "__main__":
    train_model()