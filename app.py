from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from model_types import model_types
from fairlearn.metrics import MetricFrame, selection_rate, mean_prediction

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
    y = df[target]

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
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = {
        'mse': mse,
        'r2': r2,
    }

    # Fairness metrics
    if sensitive_feature and sensitive_feature in df.columns:
        sensitive_test = X_test[sensitive_feature]
        mf = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "mean_prediction": mean_prediction
            },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test
        )
        results['fairness'] = mf.by_group.to_dict()
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
