from sklearn.linear_model import LinearRegression # Lineal
from sklearn.preprocessing import PolynomialFeatures # Polinomial
from sklearn.svm import SVR # Regresión por soporte de vectores
from sklearn.tree import DecisionTreeRegressor # Árbol de decisión
from sklearn.ensemble import RandomForestRegressor # Random forest
from sklearn.linear_model import BayesianRidge # Regresión bayesiana
from sklearn.linear_model import SGDRegressor # Gradiente estocástico
from sklearn.gaussian_process import GaussianProcessRegressor # Proceso gaussiano
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

regression_models = {
    "linear": LinearRegression(),
    "poly": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "svr": make_pipeline(StandardScaler(), SVR(kernel='rbf')),
    "decision_tree_r": DecisionTreeRegressor(),
    "random_forest_r": RandomForestRegressor(),
    "bayesian": BayesianRidge(),
    "sgd_r": make_pipeline(StandardScaler(), SGDRegressor()),
    "gaussian": GaussianProcessRegressor()
}

classifier_models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "sgd_c": make_pipeline(StandardScaler(), SGDClassifier()),
    "svc": make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True)),
    "decision_tree_c": DecisionTreeClassifier(),
    "random_forest_c": RandomForestClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "naive_bayes": GaussianNB(),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier())
}