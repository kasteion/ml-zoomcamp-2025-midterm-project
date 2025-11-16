import pickle
import xgboost as xgb
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any

app = FastAPI(title="winner-prediction")

dv_file = 'dv.bin'
model_file = 'eta=0.03max_depth=6min_child_weight=30.bin'

with open(dv_file, 'rb') as f:
    dv = pickle.load(f)

with open(model_file, 'rb') as f:
    model = pickle.load(f)


def predict_single(match):
    X = dv.transform([match])
    d = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    probability = model.predict(d)
    return float(probability)

@app.post("/predict")
def predict(match: Dict[str, Any]):
    probability = predict_single(match)
    return {
        "win_probabilty": probability,
        "winner": "player_1" if probability >= 0.5 else "player_2"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)