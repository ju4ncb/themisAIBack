from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd
import numpy as np
from model_types import model_types
from fairlearn.metrics import MetricFrame, selection_rate, mean_prediction
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import io
import base64

#comienzo
app = Flask(__name__)

@app.route('/run-model', methods=['POST'])
def run_model():
    data = request.get_json()
    
    model_type = data.get('modelType')
    target = data.get('target')
    features = data.get('features')
    raw_data = data.get('data')
    sensitive_feature = data.get('sensitiveFeature', None)


    df = pd.DataFrame(raw_data)
    X = df[features]
    X = pd.get_dummies(X, drop_first=True)
    y = pd.to_numeric(df[target], errors='coerce')

    # Encode sensitive feature if present and categorical
    if sensitive_feature and sensitive_feature in df.columns:
        sensitive_series = df[sensitive_feature]
        if sensitive_series.dtype == 'object' or str(sensitive_series.dtype).startswith('category'):
            sensitive_encoded, sensitive_labels = pd.factorize(sensitive_series)
        else:
            sensitive_encoded = sensitive_series.values
    else:
        sensitive_encoded = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type not in model_types:
        return jsonify({'error': 'Unsupported model type'}), 400
    
    try:
        model = model_types[model_type]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Evaluate basic performance
    mse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = {
        'mse': mse,
        'r2': r2,
    }

    # Fairness metrics
    if sensitive_encoded is not None:
        sensitive_test = pd.Series(sensitive_encoded, index=df.index).loc[X_test.index]
        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "mean_prediction": mean_prediction
            },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test
        )
        fairness_dict = mf.by_group.to_dict()
        # Map numeric keys to original labels
        fairness_dict_named = {
            metric: {sensitive_labels[key] if isinstance(key, (int, np.integer)) and key < len(sensitive_labels) else key: value
                     for key, value in group_dict.items()}
            for metric, group_dict in fairness_dict.items()
        }
        results['fairness'] = fairness_dict_named

        plt.figure(figsize=(6, 4))
        fairness_means = mf.by_group['mean_prediction']
        fairness_means.plot(kind='bar', color='skyblue')
        plt.ylabel('Mean Prediction')
        plt.xlabel(sensitive_feature)
        plt.title('Mean Prediction by Sensitive Group')
        plt.tight_layout()
        buf_fairness = io.BytesIO()
        plt.savefig(buf_fairness, format='png')
        plt.close()
        buf_fairness.seek(0)
        fairness_plot_base64 = base64.b64encode(buf_fairness.read()).decode('utf-8')
        results['fairness_plot'] = fairness_plot_base64

    # For regression: plot predicted vs actual values (scatter plot)
    matplotlib.use('Agg')

    # Overfitting plot: train vs test predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = y_pred  # already computed

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_train, y=y_train_pred, color='blue', alpha=0.5, label='Train')
    sns.scatterplot(x=y_test, y=y_test_pred, color='orange', alpha=0.7, label='Test')
    min_val = min(y.min(), y_train_pred.min(), y_test_pred.min())
    max_val = max(y.max(), y_train_pred.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Train vs Test')
    plt.legend()
    plt.tight_layout()
    plt.xticks(np.linspace(min_val, max_val, num=6))
    plt.yticks(np.linspace(min_val, max_val, num=6))

    buf_overfit = io.BytesIO()
    plt.savefig(buf_overfit, format='png')
    plt.close()
    buf_overfit.seek(0)
    overfit_plot_base64 = base64.b64encode(buf_overfit.read()).decode('utf-8')
    results['overfitting_plot'] = overfit_plot_base64

    return jsonify(results)

@app.route('/model-types', methods=['GET'])
def get_model_types():
    return jsonify(list(model_types.keys()))

if __name__ == '__main__':
    app.run(port=8000, debug=True)
    