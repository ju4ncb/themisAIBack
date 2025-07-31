from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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

matplotlib.use('Agg')

app = FastAPI()
app.title = "ThemisAIBackend"
app.version = "2.0.0"
app.description = """Backend para Themis, una herramienta de IA para la detección de sesgos en modelos de machine learning.
- Hecho por: @Farita1  --> GitHub"""

graph_types = {
    "bivariable": ["reg", "hist", "box", "bar"],
    "univariable": ["hist", "box", "violin", "count"],
    "multivariable": ["corr"]
}

class RunModelInput(BaseModel):
    modelType: str
    target: str
    features: list[str]
    data: list[dict]
    sensitiveFeature: str | None = None

class GraphInput(BaseModel):
    x: str
    graphType: str
    data: list[dict]

class BivariableInput(BaseModel):
    x: str
    y: str
    graphType: str
    data: list[dict]
    hue: str | None = None

class MultivariableInput(BaseModel):
    cols: list[str]
    data: list[dict]

@app.get("/model-types")
async def get_model_types():
    return JSONResponse(content={
        "regression": list(regression_models.keys()),
        "classifier": list(classifier_models.keys())
    })

@app.get("/graph-types")
async def get_graph_types():
    return JSONResponse(content=graph_types)

@app.post("/run-model")
async def run_model(payload: RunModelInput):
    model_type = payload.modelType
    target = payload.target
    features = payload.features
    raw_data = payload.data
    sensitive_feature = payload.sensitiveFeature

    df = pd.DataFrame(raw_data)
    df = df.dropna(subset=[target] + features)
    original_df = df.copy()

    if model_type in regression_models:
        y = df[target].astype(float)
        X = df[features]
    elif model_type in classifier_models:
        y = df[target]
        X = df[features]
    else:
        raise HTTPException(status_code=400, detail='Unsupported model type')

    le = LabelEncoder()
    y = le.fit_transform(y)
    class_labels = le.classes_.tolist()

    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype).startswith('category'):
            X[col] = le.fit_transform(X[col].astype(str))

    if len(X) == 0 or len(y) == 0:
        raise HTTPException(status_code=400, detail='No valid data available after dropping missing values.')
    if len(X) < 2:
        raise HTTPException(status_code=400, detail='Not enough data to split into train and test sets.')

    if sensitive_feature and sensitive_feature in original_df.columns:
        sensitive_labels_original = sorted(original_df[sensitive_feature].dropna().unique().tolist())

    if sensitive_feature and sensitive_feature in original_df.columns:
        sensitive_series = original_df[sensitive_feature]
        if sensitive_series.dtype == 'object' or str(sensitive_series.dtype).startswith('category'):
            sensitive_encoded, sensitive_labels = pd.factorize(sensitive_series)
        else:
            sensitive_encoded = sensitive_series.values
    else:
        sensitive_encoded = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    if model_type in regression_models:
        model = regression_models[model_type]
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        mse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.update({'mse': mse, 'r2': r2})

        y_train_pred = model.predict(X_train)
        y_test_pred = y_pred

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

    elif model_type in classifier_models:
        model = classifier_models[model_type]
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.update({'accuracy': accuracy, 'f1_score': f1})

        y_train_pred = model.predict(X_train)
        y_test_pred = y_pred

        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axs[0])
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Oranges', ax=axs[1])

        for ax, title in zip(axs, ["Train Confusion Matrix", "Test Confusion Matrix"]):
            ax.set_xticks(np.arange(len(class_labels)))
            ax.set_yticks(np.arange(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=45)
            ax.set_yticklabels(class_labels)
            ax.set_title(title)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.tight_layout()

    else:
        raise HTTPException(status_code=400, detail='Unsupported model type')

    buf_overfit = io.BytesIO()
    plt.savefig(buf_overfit, format='png')
    plt.close()
    buf_overfit.seek(0)
    results['overfitting_plot'] = base64.b64encode(buf_overfit.read()).decode('utf-8')

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
        fairness_dict_named = {
            metric: {
                sensitive_labels[key] if isinstance(key, (int, np.integer)) and key < len(sensitive_labels) else key: value
                for key, value in group_dict.items()
            }
            for metric, group_dict in fairness_dict.items()
        }
        results['fairness'] = fairness_dict_named

        plt.figure(figsize=(6, 4))
        mf.by_group['mean_prediction'].plot(kind='bar', color='skyblue')
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
        results['fairness_plot'] = base64.b64encode(buf_fairness.read()).decode('utf-8')

    return JSONResponse(content=results)

@app.post("/multivariable")
async def multivariable(payload: MultivariableInput):
    df = pd.DataFrame(payload.data)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo codificar la columna {col}: {e}")

    plt.figure(figsize=(12, 8))
    sns.heatmap(df[payload.cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    ax = plt.gca()
    if len(ax.get_xticklabels()) > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return JSONResponse(content={"image": img_base64})

@app.post("/univariable")
async def univariable(payload: GraphInput):
    df = pd.DataFrame(payload.data)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    plt.figure(figsize=(6, 4))
    match payload.graphType:
        case 'hist':
            sns.histplot(data=df, x=payload.x, kde=True, palette="hls")
        case 'box':
            sns.boxplot(data=df, x=payload.x, palette="hls")
        case 'count':
            sns.countplot(data=df, x=payload.x, palette="hls")
        case 'violin':
            sns.violinplot(data=df, x=payload.x, palette="hls")
        case _:
            raise HTTPException(status_code=400, detail=f'Plot type "{payload.graphType}" not recognized')

    ax = plt.gca()
    if pd.api.types.is_numeric_dtype(df[payload.x]):
        xticks = ax.get_xticks()
        if len(xticks) > 10:
            limited_xticks = np.linspace(min(xticks), max(xticks), 10)
            ax.set_xticks(limited_xticks)
            ax.set_xticklabels([f'{tick:.2f}' for tick in limited_xticks], rotation=45, ha='right')
    else:
        if len(df[payload.x].unique()) > 4:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return JSONResponse(content={"image": img_base64})

@app.post("/bivariable")
async def bivariable(payload: BivariableInput):
    df = pd.DataFrame(payload.data)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    plt.figure(figsize=(6, 4))
    match payload.graphType:
        case 'reg':
            if payload.hue:
                unique_values = df[payload.hue].unique()
                palette = sns.color_palette("hls", len(unique_values))
                for i, val in enumerate(unique_values):
                    subset = df[df[payload.hue] == val]
                    sns.regplot(x=subset[payload.x], y=subset[payload.y], color=palette[i], label=str(val))
                plt.legend(title=payload.hue)
            else:
                sns.regplot(x=df[payload.x], y=df[payload.y])
            plt.xlabel(payload.x)
            plt.ylabel(payload.y)
            plt.title(f'Diagrama de regresión de {payload.y} vs {payload.x}')

        case 'hist':
            if payload.hue and payload.hue in df.columns:
                for hue_value in df[payload.hue].unique():
                    subset = df[df[payload.hue] == hue_value]
                    sns.histplot(subset[payload.x], kde=True, element="step", alpha=0.5, label=f"{payload.x} ({hue_value})")
            else:
                sns.histplot(df[payload.x], kde=True, element="step", alpha=0.5, label=payload.x)
            plt.xlabel(payload.x)
            plt.title(f'Histograma de {payload.x} vs {payload.y}')
            plt.legend()

        case 'box':
            sns.boxplot(x=payload.x, y=payload.y, data=df, hue=df[payload.hue] if payload.hue else None, palette="hls")
            plt.ylabel(payload.y)
            plt.xlabel(payload.x)
            plt.title(f"Diagrama de cajas y bigotes de {payload.x} vs {payload.y}")

        case 'bar' if payload.y:
            sns.barplot(x=df[payload.x], y=df[payload.y], hue=df[payload.hue] if payload.hue else None, palette="hls")
            plt.xlabel(payload.x)
            plt.ylabel(payload.y)
            plt.title(f'Diagrama de barras de {payload.x} vs {payload.y}')

        case _:
            raise HTTPException(status_code=400, detail='Unsupported graph type or missing parameters')

    ax = plt.gca()
    if len(ax.get_xticklabels()) > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return JSONResponse(content={"image": img_base64})
