import joblib
import pandas as pd

model = joblib.load("models/model_lgb.pkl")

print("Model type:", type(model))
print("\nModel structure:")
if hasattr(model, 'calibrated_classifiers_'):
    print("Has calibrated classifiers")
    estimator = model.calibrated_classifiers_[0].estimator
    print("Estimator type:", type(estimator))
    if hasattr(estimator, 'feature_names_in_'):
        print("\nExpected features:")
        print(estimator.feature_names_in_)
        print("\nNumber of features:", len(estimator.feature_names_in_))
