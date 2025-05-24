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
matplotlib.use('Agg')
graph_types = ["scatter", "hist", "box", "bar"]


@app.route('/run-model', methods=['POST'])
def run_model():
    data = request.get_json()
    
    model_type = data.get('modelType')
    target = data.get('target')
    features = data.get('features')
    raw_data = data.get('data')
    sensitive_feature = data.get('sensitiveFeature', None)

    df = pd.DataFrame(raw_data)
    df = df.dropna(subset=features + [target])
    X = df[features]
    X = pd.get_dummies(X, drop_first=True)
    # Determine if target is categorical or numeric
    if df[target].dtype == 'object' or str(df[target].dtype).startswith('category'):
        # Get only the first dummy column as a Series
        y = pd.get_dummies(df[target], drop_first=False).iloc[:, 0]
    else:
        y = pd.to_numeric(df[target], errors='coerce')

    print(target, df.columns)

    # Check if there are enough samples to split
    if len(X) == 0 or len(y) == 0:
        return jsonify({'error': 'No valid data available after dropping missing values.'}), 400
    if len(X) < 2:
        return jsonify({'error': 'Not enough data to split into train and test sets.'}), 400

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
        print(f"Error during model fitting or prediction: {str(e)}")
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
        # Show x-ticks as original sensitive group labels if available
        if sensitive_feature and sensitive_feature in df.columns and 'sensitive_labels' in locals():
            plt.xticks(ticks=range(len(sensitive_labels)), labels=sensitive_labels, rotation=45, ha='right')
        plt.xlabel(sensitive_feature)
        plt.title('Mean Prediction by Sensitive Group')
        plt.tight_layout()
        buf_fairness = io.BytesIO()
        plt.savefig(buf_fairness, format='png')
        plt.close()
        buf_fairness.seek(0)
        fairness_plot_base64 = base64.b64encode(buf_fairness.read()).decode('utf-8')
        results['fairness_plot'] = fairness_plot_base64

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

@app.route('/graph-types', methods=['GET'])
def get_graph_types():
    return jsonify(graph_types)

@app.route('/explore-data', methods=["POST"])
def explore_data():
    data = request.get_json()
    raw_data = data.get('data')
    x_values = data.get('x')
    y = data.get('y')
    if isinstance(y, list):
        y = y[0] if len(y) > 0 else None

    graph_type = data.get('graphType')
    hue = data.get('hue', None)

    df = pd.DataFrame(raw_data)
    # Convert columns that can be turned into numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    plt.figure(figsize=(6, 4))


    if graph_type == 'scatter':
        sns.scatterplot(x=df[x_values].squeeze(), y=df[y], hue=df[hue] if hue else None, palette="hls")
        plt.xlabel(x_values[0])
        plt.ylabel(y)
        plt.title(f'Diagrama de dispersión de {y} vs {x_values[0]}')
    elif graph_type == 'hist':
        for x_col in x_values:
            if hue and hue in df.columns:
                for hue_value in df[hue].unique():
                    subset = df[df[hue] == hue_value]
                    sns.histplot(subset[x_col], kde=True, element="step", alpha=0.5, label=f"{x_col} ({hue_value})")
            else:
                sns.histplot(df[x_col], kde=True, element="step", alpha=0.5, label=x_col)
        plt.xlabel(", ".join(str(col) for col in x_values))
        plt.title(f'Histograma de {", ".join(str(col) for col in x_values)}')
        plt.legend()
    elif graph_type == 'box':
        for x_col in x_values:
            sns.boxplot(
                x=x_col,
                y=y,
                data=df,
                hue=df[hue] if hue else None,
                palette="hls"
            )
        plt.ylabel(y)
        plt.xlabel(", ".join(x_values))
        plt.title(f"Diagrama de cajas y bigotes de {', '.join(x_values)}")
    elif graph_type == 'bar' and y:
        for x_col in x_values:
            sns.barplot(x=df[x_col], y=df[y], hue=df[hue] if hue else None, palette="hls")
        plt.xlabel(", ".join(x_values))
        plt.ylabel(y)
        plt.title(f'Diagrama de barras de {y} por {", ".join(x_values)}')
    else:
        plt.close()
        return jsonify({'error': 'Unsupported graph type or missing parameters'}), 400

    # Rotate x-tick labels if there are more than 5 unique ticks
    ax = plt.gca()
    x_tick_labels = ax.get_xticklabels()
    if len(x_tick_labels) > 5:
        plt.setp(x_tick_labels, rotation=45, ha='right')
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({'image': img_base64})

if __name__ == '__main__':
    app.run(port=8000, debug=True)
    