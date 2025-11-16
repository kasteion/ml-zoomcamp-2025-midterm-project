import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle

# Model parameters
eta = 0.03
max_depth=6
min_child_weight=30

# Output file
output_dv = 'dv.bin'
output_model = f'eta={eta}max_depth={max_depth}min_child_weight={min_child_weight}.bin'

# Load data
df = pd.read_csv('data/rankedrm.csv', encoding='latin-1')

# Preparing data
df = df.rename(
    columns={
        "steam_id.win": "steam_id_win", 
        "name.win": "name_win", 
        "country.win": "country_win",
        "rating.win": "rating_win",
        "rating_change.win": "rating_change_win",
        "civ.win": "civ_win",
        "color.win": "color_win",
        "steam_id.lose": "steam_id_lose",
        "name.lose": "name_lose",
        "country.lose": "country_lose",
        "rating.lose": "rating_lose",
        "rating_change.lose": "rating_change_lose",
        "civ.lose": "civ_lose",
        "color.lose": "color_lose",
        "map_type.name": "map_type_name",
        "civ.lose.name": "civ_lose_name",
        "civ.win.name": "civ_win_name"
})
columns_to_exclude = ['steam_id_win', 'name_win', 'rating_change_win', 'color_win', 'steam_id_lose', 'name_lose', 'rating_change_lose', 'color_lose']
df = df.loc[:, ~df.columns.isin(columns_to_exclude)]
df.rating_win = df.rating_win.fillna(df.rating_win.mean())
df.rating_lose = df.rating_lose.fillna(df.rating_lose.mean())
df_clean = df.dropna()
p1_win = pd.DataFrame({
    "civ_p1": df_clean["civ_win"],
    "civ_p2": df_clean["civ_lose"],
    "rating_p1": df_clean["rating_win"],
    "rating_p2": df_clean["rating_lose"],
    "country_p1": df_clean["country_win"],
    "country_p2": df_clean["country_lose"],
    "map_type": df_clean["map_type"],
    "map_type_name": df_clean["map_type_name"],
    "duration": df_clean["duration"],
    "target": 1
})
p1_lose = pd.DataFrame({
    "civ_p1": df_clean["civ_lose"],
    "civ_p2": df_clean["civ_win"],
    "rating_p1": df_clean["rating_lose"],
    "rating_p2": df_clean["rating_win"],
    "country_p1": df_clean["country_lose"],
    "country_p2": df_clean["country_win"],
    "map_type": df_clean["map_type"],
    "map_type_name": df_clean["map_type_name"],
    "duration": df_clean["duration"],
    "target": 0
})
df_clean = pd.concat([p1_win, p1_lose], ignore_index=True)
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
df_clean["rating_diff"] = df_clean["rating_p1"] - df_clean["rating_p2"]

# Spliting dataset
df_full_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
y_full_train = df_full_train.target
y_test = df_test.target
y_train = df_train.target
y_val = df_val.target

df_full_train = df_full_train.drop(columns=['target'])
df_train = df_train.drop(columns=['target'])
df_val = df_val.drop(columns=['target'])


# Training model
numerical_features = list(df_train.dtypes[df_train.dtypes != 'object'].keys())
categorical_features = list(df_clean.dtypes[df_clean.dtypes == 'object'].keys())
dicts = df_train[categorical_features + numerical_features].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dicts)
features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(features))
xgb_params = {
    'eta': eta,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1
}
model = xgb.train(xgb_params, dtrain, num_boost_round=200)

# Exporting model
with open(output_dv, 'wb') as file:
    pickle.dump(dv, file)
with open(output_model, 'wb') as file:
    pickle.dump(model, file)