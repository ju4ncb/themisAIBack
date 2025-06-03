from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import MetricFrame, selection_rate, mean_prediction
from model_types import regression_models, classifier_models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import io
import base64

#comienzo
app = Flask(__name__)
matplotlib.use('Agg')
graph_types = {"bivariable": ["reg", "hist", "box", "bar"], "univariable": ["hist", "box", "violin", "count"], "multivariable": ["corr"]}


@app.route('/run-model', methods=['POST'])
def run_model():
    data = request.get_json()
    
    model_type = data.get('modelType')
    target = data.get('target')
    features = data.get('features')
    raw_data = data.get('data')
    sensitive_feature = data.get('sensitiveFeature', None)

    # Create DataFrame from raw data
    df = pd.DataFrame(raw_data)

    # Drop rows with missing values in target and features
    df = df.dropna(subset=[target] + features)

    original_df = df.copy()

    if model_type in regression_models:
        y = df[target].astype(float)
        X = df[features]
    elif model_type in classifier_models:
        y = df[target]
        X = df[features]
    else:
        return jsonify({'error': 'Unsupported model type'}), 400

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Save class labels for classifiers
    class_labels = le.classes_.tolist()

    # Encode categorical features in X
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype).startswith('category'):
            X[col] = le.fit_transform(X[col].astype(str))

    # Check if there are enough samples to split
    if len(X) == 0 or len(y) == 0:
        return jsonify({'error': 'No valid data available after dropping missing values.'}), 400
    if len(X) < 2:
        return jsonify({'error': 'Not enough data to split into train and test sets.'}), 400

    if sensitive_feature and sensitive_feature in original_df.columns:
        sensitive_labels_original = sorted(original_df[sensitive_feature].dropna().unique().tolist())

    # Encode sensitive feature if present and categorical
    if sensitive_feature and sensitive_feature in original_df.columns:
        sensitive_series = original_df[sensitive_feature]
        if sensitive_series.dtype == 'object' or str(sensitive_series.dtype).startswith('category'):
            sensitive_encoded, sensitive_labels = pd.factorize(sensitive_series)
        else:
            sensitive_encoded = sensitive_series.values
    else:
        sensitive_encoded = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type in regression_models:
        model = regression_models[model_type]
        try:
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
        # Use numeric ticks if target is numeric, else use class labels
        if class_labels is not None:
            plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45)
            plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45)
        else:
            plt.xticks(np.linspace(min_val, max_val, num=6))
            plt.yticks(np.linspace(min_val, max_val, num=6))

    elif model_type in classifier_models:
        model = classifier_models[model_type]
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"Error during model fitting or prediction: {str(e)}")
            return jsonify({'error': str(e)}), 400

        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')  # for multi-class support

        results = {
            'accuracy': accuracy,
            'f1_score': f1,
        }

        # Overfitting visualization: Confusion matrices using Seaborn
        y_train_pred = model.predict(X_train)
        y_test_pred = y_pred  # already computed

        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axs[0])
        axs[0].set_xticks(np.arange(len(class_labels)))
        axs[0].set_yticks(np.arange(len(class_labels)))
        axs[0].set_xticklabels(class_labels, rotation=45)
        axs[0].set_yticklabels(class_labels)
        axs[0].set_title("Train Confusion Matrix")
        axs[0].set_xlabel("Predicted")
        axs[0].set_ylabel("Actual")

        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Oranges', ax=axs[1])
        axs[1].set_xticks(np.arange(len(class_labels)))
        axs[1].set_yticks(np.arange(len(class_labels)))
        axs[1].set_xticklabels(class_labels, rotation=45)
        axs[1].set_yticklabels(class_labels)
        axs[1].set_title("Test Confusion Matrix")
        axs[1].set_xlabel("Predicted")
        axs[1].set_ylabel("Actual")

        plt.tight_layout()

    else:
        return jsonify({'error': 'Unsupported model type'}), 400
    
    buf_overfit = io.BytesIO()
    plt.savefig(buf_overfit, format='png')
    plt.close()
    buf_overfit.seek(0)
    overfit_plot_base64 = base64.b64encode(buf_overfit.read()).decode('utf-8')
    results['overfitting_plot'] = overfit_plot_base64

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
        if sensitive_feature and sensitive_feature in original_df.columns and 'sensitive_labels' in locals():
            plt.xticks(ticks=range(len(sensitive_labels_original)), labels=sensitive_labels_original, rotation=45, ha='right')
        plt.xlabel(sensitive_feature)
        plt.title('Mean Prediction by Sensitive Group')
        plt.tight_layout()
        buf_fairness = io.BytesIO()
        plt.savefig(buf_fairness, format='png')
        plt.close()
        buf_fairness.seek(0)
        fairness_plot_base64 = base64.b64encode(buf_fairness.read()).decode('utf-8')
        results['fairness_plot'] = fairness_plot_base64

    return jsonify(results)

@app.route('/model-types', methods=['GET'])
def get_model_types():
    model_types = {
        "regression": list(regression_models.keys()),
        "classifier": list(classifier_models.keys())
    }
    return jsonify(model_types)

@app.route('/graph-types', methods=['GET'])
def get_graph_types():
    return jsonify(graph_types)

@app.route('/multivariable', methods=["POST"])
def multivariable():
    data = request.get_json()
    cols = data.get('cols')
    raw_data = data.get('data')

    df = pd.DataFrame(raw_data)
    # Recorre las columnas no numéricas y aplica LabelEncoder
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            return jsonify({"error": f"No se pudo codificar la columna {col}: {e}"}), 400

    plt.figure(figsize=(12, 8))

    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')

    # Rotate x-tick labels if there are more than 4 unique ticks
    ax = plt.gca()
    x_tick_labels = ax.get_xticklabels()
    if len(x_tick_labels) > 4:
        plt.setp(x_tick_labels, rotation=45, ha='right')
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({'image': img_base64})

@app.route('/univariable', methods=["POST"])
def univariable():
    data = request.get_json()
    x = data.get('x')
    graph_type = data.get('graphType')
    raw_data = data.get('data')

    df = pd.DataFrame(raw_data)
    # Convert columns that can be turned into numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    plt.figure(figsize=(6, 4))
    # Plot with match-case
    match graph_type:
        case 'hist':
            sns.histplot(data=df, x=x, kde=True, palette="hls")
        case 'box':
            sns.boxplot(data=df, x=x, palette="hls")
        case 'count':
            sns.countplot(data=df, x=x, palette="hls")
        case 'violin':
            sns.violinplot(data=df, x=x, palette="hls")
        case _:
            return jsonify({'error': f'Plot type "{graph_type}" not recognized'}), 400

    ax = plt.gca()

    # If numeric, limit xticks to 10
    if pd.api.types.is_numeric_dtype(df[x]):
        # Get current xticks
        xticks = ax.get_xticks()
        if len(xticks) > 10:
            # Choose up to 10 evenly spaced ticks
            limited_xticks = np.linspace(min(xticks), max(xticks), 10)
            ax.set_xticks(limited_xticks)
            ax.set_xticklabels([f'{tick:.2f}' for tick in limited_xticks], rotation=45, ha='right')
    else:
        # If not numeric, rotate if too many categories
        if len(df[x].unique()) > 4:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({'image': img_base64})

@app.route('/bivariable', methods=["POST"])
def bivariable():
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    graph_type = data.get('graphType')
    hue = data.get('hue', None)
    raw_data = data.get('data')

    df = pd.DataFrame(raw_data)
    # Convert columns that can be turned into numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    plt.figure(figsize=(6, 4))

    match graph_type:
        case 'reg':
            if hue:
                unique_values = df[hue].unique()
                palette = sns.color_palette("hls", len(unique_values))
                for i, val in enumerate(unique_values):
                    subset = df[df[hue] == val]
                    sns.regplot(x=subset[x], y=subset[y], color=palette[i], label=str(val))
                plt.legend(title=hue)
            else:
                sns.regplot(x=df[x], y=df[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f'Diagrama de regresión de {y} vs {x}')
    
        case 'hist':
            if hue and hue in df.columns:
                for hue_value in df[hue].unique():
                    subset = df[df[hue] == hue_value]
                    sns.histplot(
                        subset[x],
                        kde=True,
                        element="step",
                        alpha=0.5,
                        label=f"{x} ({hue_value})"
                    )
            else:
                sns.histplot(df[x], kde=True, element="step", alpha=0.5, label=x)
    
            plt.xlabel(x)
            plt.title(f'Histograma de {x} vs {y}')
            plt.legend()
    
        case 'box':
            sns.boxplot(
                x=x,
                y=y,
                data=df,
                hue=df[hue] if hue else None,
                palette="hls"
            )
            plt.ylabel(y)
            plt.xlabel(x)
            plt.title(f"Diagrama de cajas y bigotes de {x} vs {y}")
    
        case 'bar' if y:
            sns.barplot(
                x=df[x],
                y=df[y],
                hue=df[hue] if hue else None,
                palette="hls"
            )
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f'Diagrama de barras de {x} vs {y}')
    
        case _:
            plt.close()
            return jsonify({'error': 'Unsupported graph type or missing parameters'}), 400

    # Rotate x-tick labels if there are more than 4 unique ticks
    ax = plt.gca()
    x_tick_labels = ax.get_xticklabels()
    if len(x_tick_labels) > 4:
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
    