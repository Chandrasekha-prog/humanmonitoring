from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import os
import json
import kagglehub
import threading  # Added to prevent Render startup timeout
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

# FIX: Corrected __name__ syntax
app = Flask(__name__)

# Global variables
models = {}
label_encoder = None
feature_names = None
mean_values = None
feature_importance = None
X_train = None
y_train = None
scaler = None
selector = None
selected_features = None
model_performance = {}
sensor_data_buffer = []
BUFFER_SIZE = 50  
is_training = False  # Added flag to track initialization state
training_error = None

def load_and_train_models():
    """Load data and train multiple models"""
    global models, label_encoder, feature_names, mean_values, feature_importance, X_train, y_train, scaler, selector, selected_features, model_performance, is_training, training_error
    
    is_training = True
    try:
        print("📥 Loading data...")  
        # Download dataset  
        path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")  
        csv_filenames = [f for f in os.listdir(path) if f.endswith('.csv')]  

        # Load train and test datasets  
        train_file = next((f for f in csv_filenames if 'train' in f.lower()), csv_filenames[0])  
        test_file = next((f for f in csv_filenames if 'test' in f.lower()), csv_filenames[0])  

        # Use full dataset for better training  
        if train_file and test_file:  
            train_df = pd.read_csv(os.path.join(path, train_file))  
            test_df = pd.read_csv(os.path.join(path, test_file))  
            df = pd.concat([train_df, test_df], ignore_index=True)  
            print(f"✅ Combined data loaded! Shape: {df.shape}")  
        else:  
            df = pd.read_csv(os.path.join(path, csv_filenames[0]))  
            print(f"✅ Data loaded! Shape: {df.shape}")  

        # Preprocessing  
        X = df.drop(['Activity'], axis=1)  
        y = df['Activity']  
        label_encoder = LabelEncoder()  
        y_encoded = label_encoder.fit_transform(y)  

        # Split data with stratification  
        X_train, X_test, y_train, y_test = train_test_split(  
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded  
        )  

        # Feature scaling - IMPORTANT for SVM  
        scaler = StandardScaler()  
        X_train_scaled = scaler.fit_transform(X_train)  
        X_test_scaled = scaler.transform(X_test)  

        # Feature selection using RandomForest  
        print("🔍 Selecting important features...")  
        temp_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  
        temp_model.fit(X_train_scaled, y_train)  

        # Select features with importance > median  
        selector = SelectFromModel(temp_model, threshold='median', prefit=True)  
        X_train_selected = selector.transform(X_train_scaled)  
        X_test_selected = selector.transform(X_test_scaled)  

        selected_features = X.columns[selector.get_support()].tolist()  
        print(f"✅ Selected {len(selected_features)} important features out of {len(X.columns)}")  

        # Store feature info  
        feature_names = X.columns.tolist()  
        mean_values = X_train.mean().to_dict()  
        feature_importance = pd.DataFrame({  
            'feature': feature_names,  
            'importance': temp_model.feature_importances_  
        }).sort_values('importance', ascending=False)  

        # ===== TRAIN MULTIPLE MODELS =====  
        print("\n🤖 Training multiple models...")  

        models = {}  
        model_performance = {}  

        # 1. Random Forest  
        print("  - Training Random Forest...")  
        rf_model = RandomForestClassifier(  
            n_estimators=200,  
            max_depth=20,  
            min_samples_split=5,  
            min_samples_leaf=2,  
            max_features='sqrt',  
            random_state=42,  
            n_jobs=-1,  
            class_weight='balanced'  
        )  
        rf_model.fit(X_train_selected, y_train)  
        rf_score = rf_model.score(X_test_selected, y_test)  
        models['Random Forest'] = rf_model  
        model_performance['Random Forest'] = {  
            'accuracy': rf_score,  
            'cv_score': cross_val_score(rf_model, X_train_selected, y_train, cv=5).mean()  
        }  
        print(f"    ✅ Random Forest Accuracy: {rf_score:.4f}")  

        # 2. SVM  
        print("  - Training SVM...")  
        svm_model = SVC(  
            kernel='rbf',  
            C=10,  
            gamma='scale',  
            probability=True,  
            random_state=42,  
            class_weight='balanced'  
        )  
        svm_model.fit(X_train_selected, y_train)  
        svm_score = svm_model.score(X_test_selected, y_test)  
        models['SVM'] = svm_model  
        model_performance['SVM'] = {  
            'accuracy': svm_score,  
            'cv_score': cross_val_score(svm_model, X_train_selected, y_train, cv=5).mean()  
        }  
        print(f"    ✅ SVM Accuracy: {svm_score:.4f}")  

        # 3. XGBoost  
        if XGB_AVAILABLE:  
            print("  - Training XGBoost...")  
            xgb_model = XGBClassifier(  
                n_estimators=200,  
                max_depth=6,  
                learning_rate=0.1,  
                subsample=0.8,  
                colsample_bytree=0.8,  
                random_state=42,  
                use_label_encoder=False,  
                eval_metric='mlogloss',  
                n_jobs=-1  
            )  
            xgb_model.fit(X_train_selected, y_train)  
            xgb_score = xgb_model.score(X_test_selected, y_test)  
            models['XGBoost'] = xgb_model  
            model_performance['XGBoost'] = {  
                'accuracy': xgb_score,  
                'cv_score': cross_val_score(xgb_model, X_train_selected, y_train, cv=5).mean()  
            }  
            print(f"    ✅ XGBoost Accuracy: {xgb_score:.4f}")  
        else:  
            print("  ⚠️ XGBoost not available - skipping")  

        best_model_name = max(model_performance, key=lambda x: model_performance[x]['accuracy'])  
        print(f"\n🏆 Best Model: {best_model_name} with accuracy {model_performance[best_model_name]['accuracy']:.4f}")  
        print("✅ Application ready!")
        
    except Exception as e:
        training_error = str(e)
        print(f"❌ Critical pipeline error: {training_error}")
    finally:
        is_training = False

# FIX: Run inside a background thread so Render can start the web server instantly without 30s timeouts
print("🚀 Launching background training thread...")
threading.Thread(target=load_and_train_models, daemon=True).start()

@app.route('/')
def index():
    """Serve the main page"""
    # Defensive check if model pipeline isn't finished processing yet
    if not models and is_training:
        return "<h3>Models are currently training in the background. Please refresh in a minute! 🤖</h3>", 202
    if training_error:
        return f"<h3>Pipeline initialization error: {training_error}</h3>", 500

    top_features = feature_importance.head(20).to_dict('records')
    activities = label_encoder.classes_.tolist()  
    
    performance_data = []  
    for name, perf in model_performance.items():  
        performance_data.append({  
            'name': name,  
            'accuracy': f"{perf['accuracy']:.2%}",  
            'cv_score': f"{perf['cv_score']:.2%}"  
        })  

    best_model = max(model_performance, key=lambda x: model_performance[x]['accuracy'])  
    
    user_agent = request.headers.get('User-Agent', '').lower()  
    is_mobile = any(device in user_agent for device in ['mobile', 'android', 'iphone', 'ipad'])  

    return render_template('index.html',   
                         features=top_features,  
                         activities=activities,  
                         total_features=len(feature_names),  
                         selected_features=len(selected_features),  
                         models=performance_data,  
                         best_model=best_model,  
                         xgb_available=XGB_AVAILABLE,  
                         is_mobile=is_mobile)

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using selected model"""
    if not models:
        return jsonify({'error': 'Models have not finished training yet'}), 503
        
    try:
        data = request.json
        prediction_type = data.get('type', 'manual')
        model_name = data.get('model', 'Random Forest')

        if model_name not in models:  
            model_name = list(models.keys())[0]  

        model = models[model_name]  

        if prediction_type == 'realtime':  
            features = data.get('features', {})  
            input_values = mean_values.copy()  
            for key, value in features.items():  
                if key in input_values:  
                    input_values[key] = float(value)  

            input_df = pd.DataFrame([input_values])  
            X_scaled = scaler.transform(input_df)  
            X_selected = selector.transform(X_scaled)  

            prediction = model.predict(X_selected)[0]  
            proba = model.predict_proba(X_selected)[0]  

            top_indices = np.argsort(proba)[-3:][::-1]  
            top_predictions = []  
            for idx in top_indices:  
                top_predictions.append({  
                    'activity': label_encoder.inverse_transform([idx])[0],  
                    'probability': float(proba[idx] * 100)  
                })  

            return jsonify({  
                'prediction': label_encoder.inverse_transform([prediction])[0],  
                'confidence': float(np.max(proba) * 100),  
                'top_predictions': top_predictions,  
                'all_probabilities': [float(p * 100) for p in proba],  
                'model_used': model_name,  
                'timestamp': datetime.now().isoformat()  
            })  

        elif prediction_type == 'csv':  
            csv_content = data.get('content', '')  
            if not csv_content:  
                return jsonify({'error': 'No CSV content provided'}), 400  

            from io import StringIO  
            df_upload = pd.read_csv(StringIO(csv_content))  

            if 'Activity' in df_upload.columns:  
                df_upload = df_upload.drop(['Activity'], axis=1)  

            missing_features = set(feature_names) - set(df_upload.columns)  
            if missing_features:  
                return jsonify({  
                    'error': f'Missing features: {list(missing_features)[:5]}...'  
                }), 400  

            X_scaled = scaler.transform(df_upload)  
            X_selected = selector.transform(X_scaled)  

            predictions = model.predict(X_selected)  
            probabilities = model.predict_proba(X_selected)  

            results = []  
            for i, pred in enumerate(predictions):  
                results.append({  
                    'row': i + 1,  
                    'activity': label_encoder.inverse_transform([pred])[0],  
                    'confidence': float(np.max(probabilities[i]) * 100)  
                })  

            return jsonify({'results': results, 'count': len(results), 'model_used': model_name})  

        elif prediction_type == 'manual':  
            input_values = mean_values.copy()  
            for key, value in data.get('features', {}).items():  
                if key in input_values:  
                    input_values[key] = float(value)  

            input_df = pd.DataFrame([input_values])  
            X_scaled = scaler.transform(input_df)  
            X_selected = selector.transform(X_scaled)  

            prediction = model.predict(X_selected)[0]  
            proba = model.predict_proba(X_selected)[0]  

            top_indices = np.argsort(proba)[-3:][::-1]  
            top_predictions = []  
            for idx in top_indices:  
                top_predictions.append({  
                    'activity': label_encoder.inverse_transform([idx])[0],  
                    'probability': float(proba[idx] * 100)  
                })  

            return jsonify({  
                'prediction': label_encoder.inverse_transform([prediction])[0],  
                'confidence': float(np.max(proba) * 100),  
                'top_predictions': top_predictions,  
                'all_probabilities': [float(p * 100) for p in proba],  
                'model_used': model_name  
            })  

        elif prediction_type == 'sample':  
            sample_size = min(10, len(X_train))  
            sample_indices = np.random.choice(len(X_train), sample_size, replace=False)  
            sample_data = X_train.iloc[sample_indices]  
            sample_labels = y_train[sample_indices]  

            X_scaled = scaler.transform(sample_data)  
            X_selected = selector.transform(X_scaled)  

            predictions = model.predict(X_selected)  
            probabilities = model.predict_proba(X_selected)  

            results = []  
            for i, (idx, pred) in enumerate(zip(sample_indices, predictions)):  
                results.append({  
                    'sample_id': i + 1,  
                    'true_activity': label_encoder.inverse_transform([sample_labels[i]])[0],  
                    'predicted_activity': label_encoder.inverse_transform([pred])[0],  
                    'confidence': float(np.max(probabilities[i]) * 100)  
                })  

            return jsonify({'results': results, 'model_used': model_name})  

        return jsonify({'error': 'Invalid prediction type'}), 400  

    except Exception as e:  
        print(f"❌ Error in prediction: {str(e)}")  
        import traceback  
        traceback.print_exc()  
        return jsonify({'error': str(e)}), 500

@app.route('/compare_models')
def compare_models():
    """Compare all models"""
    if not model_performance:
        return jsonify([])
    comparison = []
    for name, perf in model_performance.items():
        comparison.append({
            'model': name,
            'accuracy': perf['accuracy'],
            'cv_score': perf['cv_score']
        })
    return jsonify(comparison)

@app.route('/feature_importance')
def get_feature_importance():
    """Return feature importance data"""
    if feature_importance is None:
        return jsonify([])
    top_features = feature_importance.head(30).to_dict('records')
    return jsonify(top_features)

# FIX: Static files router corrected
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Render configures ports via environment variables dynamically
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

          