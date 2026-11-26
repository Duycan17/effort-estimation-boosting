import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor

SEED = 42
TEST_SIZE = 0.2
N_ITER = 100
N_FOLDS = 5

CATEGORIES = ["Bug", "Feature", "Improvement", "Unknown"]
COMPONENTS = ["General", "API", "network", "performance", "security", "authentication", "backend", "database", "UI"]
TAWOS_NUMERIC = [
    "title_length_chars", "title_word_count", "body_length_chars", "body_word_count",
    "has_description_code", "description_code_length_chars",
    "comment_count_before_estimation", "unique_commenters_before_estimation",
    "comment_text_length_before_estimation", "comment_code_length_before_estimation",
    "changelog_count_before_estimation", "changed_fields_before_estimation",
    "project_age_days_at_creation", "project_prior_issues",
    "creator_prior_created_issues_global", "creator_prior_created_issues_project",
    "reporter_prior_reported_issues_global", "reporter_prior_reported_issues_project",
    "contributors_expertise_global", "contributors_expertise_project",
    "created_year", "created_month", "created_weekday", "created_hour",
]
CHOET_NUMERIC = [
    "title_length_chars", "title_word_count", "description_length_chars", "description_word_count",
    "total_length_chars", "desc_title_ratio", "has_code", "has_url", "exclamation_count", "question_count",
]
CODE_PATTERN = re.compile(r"(```|\{code\}|<code|<pre|@Override|public\s+\w+\s*\(|;\s*$)", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://|www\.")


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories):
        self.categories = categories

    def fit(self, X, y=None):
        self.lookup_ = {label.lower(): index for index, label in enumerate(self.categories)}
        return self

    def transform(self, X):
        values = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(np.asarray(X).ravel())
        encoded = np.zeros((len(values), len(self.categories)), dtype=float)
        for row, cell in enumerate(values):
            for part in str(cell).split(","):
                index = self.lookup_.get(part.strip().lower())
                if index is not None:
                    encoded[row, index] = 1.0
        return encoded

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"component_{label}" for label in self.categories])


def blank(series):
    return series.isna() | series.astype(str).str.strip().isin(["", "nan", "None", "NaN"])


def read_table(path):
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path)
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as stream:
            return pd.read_csv(io.TextIOWrapper(stream, encoding="utf-8"))


def scores(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MdAE": float(median_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def ages_frame(path):
    frame = read_table(path)
    frame = frame.dropna(subset=["Time Taken (days)"]).copy()
    frame = frame.loc[~blank(frame["Category"]) & ~blank(frame["Component"])].copy()
    frame = frame.drop_duplicates(subset=["Number"], keep="first")
    frame = frame.drop_duplicates(keep="first")
    known = {label.lower(): label for label in CATEGORIES}
    recognized = {label.lower() for label in COMPONENTS}
    frame["Category"] = frame["Category"].map(lambda value: known.get(str(value).strip().lower(), "Unknown"))
    raw_component = frame["Component"].astype(str)
    frame["num_components"] = raw_component.map(
        lambda value: max(1, sum(part.strip().lower() in recognized for part in str(value).split(",")))
    ).astype(float)
    frame["comments_count"] = pd.to_numeric(frame["Comments"], errors="coerce").fillna(0.0)
    frame["Component"] = raw_component
    frame = frame.loc[frame["Contributors Expertise"].notna() & (frame["Contributors Expertise"] != 0)].reset_index(drop=True)
    features = frame[["Category", "Component", "Contributors Expertise", "num_components", "comments_count"]].copy()
    target = np.log1p(frame["Time Taken (days)"].to_numpy(dtype=float))
    return features, target


def ages_pipeline(model):
    preprocessor = ColumnTransformer(
        [
            ("category", OneHotEncoder(categories=[CATEGORIES], handle_unknown="ignore", sparse_output=False), ["Category"]),
            ("component", MultiHotEncoder(COMPONENTS), ["Component"]),
            ("numeric", MinMaxScaler(), ["Contributors Expertise", "num_components", "comments_count"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def tawos_frame(path):
    frame = read_table(path)
    frame["story_point"] = pd.to_numeric(frame["story_point"], errors="coerce")
    valid = (
        frame["story_point"].notna()
        & (frame["story_point"] >= 0)
        & ~blank(frame["repository"])
        & ~blank(frame["project_key"])
        & (~blank(frame["title"]) | ~blank(frame["body"]))
    )
    frame = frame.loc[valid].copy()
    frame["_row_id"] = (
        frame["repository"].astype(str).str.strip()
        + "::"
        + frame["project_key"].astype(str).str.strip()
        + "::"
        + frame["number"].astype(str).str.strip()
    )
    frame = frame.drop_duplicates(subset=["_row_id"], keep="first").reset_index(drop=True)
    created = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    features = pd.DataFrame({
        "repository": frame["repository"].astype(str).str.strip(),
        "project_key": frame["project_key"].astype(str).str.strip(),
    })
    for column in TAWOS_NUMERIC[:20]:
        features[column] = pd.to_numeric(frame[column], errors="coerce")
    features["created_year"] = created.dt.year.astype(float)
    features["created_month"] = created.dt.month.astype(float)
    features["created_weekday"] = created.dt.weekday.astype(float)
    features["created_hour"] = created.dt.hour.astype(float)
    return features, np.log1p(frame["story_point"].to_numpy(dtype=float))


def choet_frame(path):
    files = sorted(Path(path).glob("*.csv"))
    parts = []
    for file in files:
        part = pd.read_csv(file)
        part["project"] = file.stem
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["storypoint"] = pd.to_numeric(frame["storypoint"], errors="coerce")
    frame["title"] = frame["title"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    valid = (
        frame["storypoint"].notna()
        & (frame["storypoint"] > 0)
        & (frame["storypoint"] <= 100)
        & ((frame["title"].str.strip() != "") | (frame["description"].str.strip() != ""))
    )
    frame = frame.loc[valid].drop_duplicates(subset=["issuekey"], keep="first").reset_index(drop=True)
    rows = []
    for title, description in zip(frame["title"], frame["description"]):
        text = f"{title}\n{description}"
        title_chars, description_chars = len(title), len(description)
        rows.append({
            "project": None,
            "title_length_chars": title_chars,
            "title_word_count": len(title.split()),
            "description_length_chars": description_chars,
            "description_word_count": len(description.split()),
            "total_length_chars": title_chars + description_chars,
            "desc_title_ratio": description_chars / max(title_chars, 1),
            "has_code": float(bool(CODE_PATTERN.search(text))),
            "has_url": float(bool(URL_PATTERN.search(text))),
            "exclamation_count": float((title + description).count("!")),
            "question_count": float((title + description).count("?")),
        })
    features = pd.DataFrame(rows)
    features["project"] = frame["project"].astype(str).str.strip().to_numpy()
    return features, np.log1p(frame["storypoint"].to_numpy(dtype=float))


def tabular_pipeline(model, categoricals, numerics):
    preprocessor = ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categoricals),
            ("numeric", SimpleImputer(strategy="median"), numerics),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def models():
    return [
        ("Gradient Boosting", lambda: GradientBoostingRegressor(random_state=SEED), {
            "n_estimators": randint(100, 501),
            "learning_rate": uniform(0.02, 0.13),
            "max_depth": randint(2, 6),
            "min_samples_leaf": randint(1, 21),
            "subsample": uniform(0.70, 0.30),
        }),
        ("LightGBM", lambda: LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1), {
            "n_estimators": randint(150, 701),
            "learning_rate": uniform(0.01, 0.12),
            "num_leaves": randint(15, 65),
            "min_child_samples": randint(10, 81),
            "colsample_bytree": uniform(0.60, 0.40),
            "reg_lambda": uniform(0, 5),
        }),
        ("XGBoost", lambda: XGBRegressor(random_state=SEED, verbosity=0, n_jobs=1), {
            "n_estimators": randint(200, 801),
            "learning_rate": uniform(0.01, 0.09),
            "max_depth": randint(3, 9),
            "min_child_weight": randint(1, 11),
            "subsample": uniform(0.70, 0.30),
            "colsample_bytree": uniform(0.60, 0.40),
            "reg_lambda": uniform(1, 19),
        }),
        ("HistGradientBoosting", lambda: HistGradientBoostingRegressor(random_state=SEED), {
            "max_iter": randint(100, 501),
            "learning_rate": uniform(0.02, 0.13),
            "max_leaf_nodes": randint(15, 64),
            "min_samples_leaf": randint(10, 61),
            "l2_regularization": uniform(0, 5),
        }),
        ("CatBoost", lambda: CatBoostRegressor(random_seed=SEED, verbose=0, allow_writing_files=False), {
            "iterations": randint(200, 801),
            "learning_rate": uniform(0.02, 0.13),
            "depth": randint(4, 9),
            "l2_leaf_reg": uniform(1, 9),
            "random_strength": uniform(0, 2),
        }),
    ]


def load(dataset, path):
    if dataset == "ages":
        features, target = ages_frame(path)
        return features, target, lambda model: ages_pipeline(model)
    if dataset == "tawos":
        features, target = tawos_frame(path)
        return features, target, lambda model: tabular_pipeline(model, ["repository", "project_key"], TAWOS_NUMERIC)
    features, target = choet_frame(path)
    return features, target, lambda model: tabular_pipeline(model, ["project"], CHOET_NUMERIC)


def run(dataset, path, output, jobs):
    features, target, pipeline_of = load(dataset, path)
    train, test = train_test_split(np.arange(len(target)), test_size=TEST_SIZE, random_state=SEED)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    selected = {}
    for name, factory, space in models():
        search = RandomizedSearchCV(
            pipeline_of(factory()),
            {f"model__{parameter}": distribution for parameter, distribution in space.items()},
            n_iter=N_ITER,
            scoring="neg_mean_absolute_error",
            cv=list(folds.split(train)),
            refit=True,
            random_state=SEED,
            n_jobs=jobs,
            error_score="raise",
        )
        search.fit(features.iloc[train], target[train])
        prediction = np.asarray(search.predict(features.iloc[test]), dtype=float)
        rows.append({"Model": name, **scores(target[test], prediction)})
        selected[name] = {
            key.removeprefix("model__"): value.item() if hasattr(value, "item") else value
            for key, value in search.best_params_.items()
        }
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        print(f"{dataset} {name} MAE={rows[-1]['MAE']:.4f}", flush=True)
    (output / "selected_parameters.json").write_text(json.dumps(selected, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ages", "tawos", "choet"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    run(args.dataset, args.data, args.output, args.jobs)


if __name__ == "__main__":
    main()
