import pickle
import xgboost as xgb
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field, field_validator
import pycountry
from typing import Optional

COUNTRY_CODES = {country.alpha_2 for country in pycountry.countries}

class Match(BaseModel):
    player1_civilization: int = Field(ge=0, le=36)
    player1_rating: float
    player1_country: str
    player2_civilization: int = Field(ge=0, le=36)
    player2_rating: float
    player2_country: str

    @field_validator('player1_country', 'player2_country')
    @classmethod
    def validate_country(cls, v):
        v = v.upper()
        if v not in COUNTRY_CODES:
            raise ValueError(f"{v} is not a valid country code")
        return v


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
    return float(probability[0])

@app.post("/predict")
def predict(match_payload: Match):
    match = Match.model_validate(match_payload.model_dump())
    probability = predict_single({
        'civ_p1': match.player1_civilization,
        'civ_p2': match.player2_civilization,
        'rating_p1': match.player1_rating,
        'rating_p2': match.player2_rating,
        'country_p1': match.player1_country,
        'country_p2': match.player2_country,
        'rating_diff': match.player1_rating - match.player2_rating
    })
    return {
        "win_probabilty": probability,
        "winner": "player_1" if probability >= 0.5 else "player_2"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)