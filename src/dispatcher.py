from sklearn import ensemble
import xgboost as xgb
MODELS = {
    "bagging": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=0),
    "boosting": xgb.XGBClassifier(n_estimators=200),
}