# Age of Empires II Winner Predictor

Predicting the winner of an Age of Empires II match is a challenging task that involves analyzing the skill, civilization choice, map type, and behavior of two competing players. Competitive AoE2 matches contain rich strategic complexity, and even small differences in player rating, civilization strengths, or map characteristics can influence the outcome.

The goal of this project is to build a machine learning model capable of predicting the match winner based on pre-game and early-game information. This turns AoE2 competitive data into a binary classification problem, where the model must determine whether a given player (Player 1) is likely to win against their opponent (Player 2).

To achieve this, I used a dataset containing thousands of real AoE2 ranked matches, with fields describing:

- Player attributes such as rating, rating change, country, color, and Steam ID
- Civilization selections and their names (for both the winning and losing player)
- Map information (map_type, map_type.name)
- Match outcome (winner vs. loser)
- Match metadata such as duration, matchup type, and timestamps

The machine learning workflow includes:

- Exploratory Data Analysis (EDA) to understand player behavior, rating distributions, civilization popularity, map effects, and outcome patterns
- Data cleaning and feature engineering, including encoding categorical variables, normalizing numeric fields, and creating useful matchup-level features
- Training multiple ML models including Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Selecting XGBoost as the best-performing model, based on accuracy and generalization metrics
- Deploying the final model using FastAPI, making predictions accessible via an HTTP endpoint

This project aims to demonstrate real-world machine learning techniques applied to a competitive gaming domain, while providing insights into how different factors influence victory in Age of Empires II.

## Dataset

For this project I used the following dataset: [data/rankedrm.csv](https://github.com/kasteion/ml-zoomcamp-2025-midterm-project/blob/main/data/rankedrm.csv)

## Installation and Setup

This project uses uv for dependency management, virtual environments, and project execution.

If you don’t have uv installed yet, follow the official instructions:

- https://github.com/astral-sh/uv

1. Clone the repo

```bash
git clone https://github.com/kasteion/ml-zoomcamp-2025-midterm-project.git
cd ml-zoomcamp-2025-midterm-project
```

2. Install dependencies

```bash
uv sync --locked
```

3. Run the train script

```bash
uv run train.py
```

4. Run the prediction service

```bash
uv run predict.py
```

## Docker

1. Build the Docker image

```bash
docker build -t aoe-winner-prediction .
```

2. Run the Docker container

```bash
docker run -it --rm -p 9696:9696 aoe-winner-prediction
```

## URL for testing

https://ml-zoomcamp-2025-midterm-project-blue-sun-3900.fly.dev/predict

```bash
curl -X 'POST' \
  'https://ml-zoomcamp-2025-midterm-project-blue-sun-3900.fly.dev/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "player1_civilization": 5,
  "player1_rating": 1410.0,
  "player1_country": "US",
  "player2_civilization": 4,
  "player2_rating": 1405.0,
  "player2_country": "RU"
}'
```
