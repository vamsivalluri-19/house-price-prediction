from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.data import load_or_download_dataset
from src.evaluate import compute_regression_metrics
from src.preprocessing import HousingPreprocessor
from src.utils import MODELS_DIR, RunArtifacts, save_json, setup_logging, timestamp_now

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBRegressor = None  # type: ignore


logger = setup_logging("train")


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_splits: int = 5


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_eda(df: pd.DataFrame, plots_dir: Path) -> dict[str, Any]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    eda_report: dict[str, Any] = {}

    missing_values = df.isna().sum().sort_values(ascending=False)
    eda_report["missing_values"] = missing_values[missing_values > 0].to_dict()

    duplicates = int(df.duplicated().sum())
    eda_report["duplicate_rows"] = duplicates

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    sample_cols = [c for c in ["price", "area_sqft", "bedrooms", "bathrooms", "age_of_property"] if c in df.columns]
    if len(sample_cols) > 1:
        sns.pairplot(df[sample_cols].dropna(), diag_kind="kde")
        plt.savefig(plots_dir / "pairplot.png", dpi=200)
        plt.close("all")

    for col in ["price", "area_sqft", "bedrooms", "bathrooms", "age_of_property"]:
        if col in numeric_df.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"dist_{col}.png", dpi=200)
            plt.close()

            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Outlier Detection - {col}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"outlier_{col}.png", dpi=200)
            plt.close()

    return eda_report


def build_models(random_state: int) -> dict[str, tuple[RegressorMixin, dict[str, list[Any]]]]:
    models: dict[str, tuple[RegressorMixin, dict[str, list[Any]]]] = {
        "linear_regression": (
            LinearRegression(),
            {
                "feature_selector__k": [20, 40, "all"],
            },
        ),
        "ridge": (
            Ridge(random_state=random_state),
            {
                "feature_selector__k": [20, 40, "all"],
                "model__alpha": [0.1, 1.0, 10.0],
            },
        ),
        "lasso": (
            Lasso(random_state=random_state, max_iter=5000),
            {
                "feature_selector__k": [20, 40, "all"],
                "model__alpha": [0.001, 0.01, 0.1],
            },
        ),
        "decision_tree": (
            DecisionTreeRegressor(random_state=random_state),
            {
                "feature_selector__k": [20, 40, "all"],
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 5, 10],
            },
        ),
        "random_forest": (
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            {
                "feature_selector__k": [20, 40, "all"],
                "model__n_estimators": [150, 300],
                "model__max_depth": [None, 15],
            },
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "feature_selector__k": [20, 40, "all"],
                "model__n_estimators": [150, 300],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }

    if HAS_XGBOOST and XGBRegressor is not None:
        models["xgboost"] = (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
            ),
            {
                "feature_selector__k": [20, 40, "all"],
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.03, 0.07],
            },
        )

    return models


def train_models(
    x_train: Any,
    y_train: pd.Series,
    x_test: Any,
    y_test: pd.Series,
    config: TrainConfig,
) -> tuple[dict[str, Any], str, Any, dict[str, Any]]:
    cv = KFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_state)
    rmse_scorer = make_scorer(_rmse, greater_is_better=False)
    models = build_models(config.random_state)

    leaderboard: dict[str, Any] = {}
    trained_estimators: dict[str, Any] = {}

    for name, (model, param_grid) in models.items():
        logger.info("Training model: %s", name)

        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ("feature_selector", SelectKBest(score_func=f_regression)),
                ("model", model),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=rmse_scorer,
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        grid.fit(x_train, y_train)
        best_estimator = grid.best_estimator_
        y_pred = best_estimator.predict(x_test)
        metrics = compute_regression_metrics(y_test.values, y_pred, n_features=x_train.shape[1])

        cv_scores = cross_val_score(
            best_estimator,
            x_train,
            y_train,
            scoring=rmse_scorer,
            cv=cv,
            n_jobs=-1,
        )

        leaderboard[name] = {
            "best_params": grid.best_params_,
            "cv_rmse_mean": float(-cv_scores.mean()),
            "cv_rmse_std": float(cv_scores.std()),
            "test_metrics": asdict(metrics),
        }
        trained_estimators[name] = best_estimator

    logger.info("Training stacking ensemble")
    base_estimators = [
        ("rf", trained_estimators["random_forest"]),
        ("gb", trained_estimators["gradient_boosting"]),
    ]
    if "xgboost" in trained_estimators:
        base_estimators.append(("xgb", trained_estimators["xgboost"]))

    stacker = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=config.random_state),
        n_jobs=-1,
    )
    stacker.fit(x_train, y_train)
    stack_pred = stacker.predict(x_test)
    stack_metrics = compute_regression_metrics(y_test.values, stack_pred, n_features=x_train.shape[1])

    leaderboard["stacking_ensemble"] = {
        "best_params": {"meta_model": "Ridge(alpha=1.0)"},
        "cv_rmse_mean": None,
        "cv_rmse_std": None,
        "test_metrics": asdict(stack_metrics),
    }
    trained_estimators["stacking_ensemble"] = stacker

    best_model_name = min(
        leaderboard,
        key=lambda name: leaderboard[name]["test_metrics"]["rmse"],
    )
    best_model = trained_estimators[best_model_name]

    return leaderboard, best_model_name, best_model, trained_estimators


def create_model_comparison_plot(leaderboard: dict[str, Any], plots_dir: Path) -> None:
    rows = [
        {
            "model": k,
            "RMSE": v["test_metrics"]["rmse"],
            "R2": v["test_metrics"]["r2"],
        }
        for k, v in leaderboard.items()
    ]
    frame = pd.DataFrame(rows).sort_values("RMSE")

    plt.figure(figsize=(11, 5))
    sns.barplot(data=frame, x="model", y="RMSE", hue="model", legend=False, palette="viridis")
    plt.xticks(rotation=25)
    plt.title("Model Comparison by RMSE")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison_rmse.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 5))
    sns.barplot(data=frame, x="model", y="R2", hue="model", legend=False, palette="mako")
    plt.xticks(rotation=25)
    plt.title("Model Comparison by R2")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison_r2.png", dpi=200)
    plt.close()


def create_prediction_diagnostics(
    model: Any,
    x_test: Any,
    y_test: pd.Series,
    plots_dir: Path,
) -> None:
    y_pred = model.predict(x_test)
    residuals = y_test.values - y_pred

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_test.values, y=y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(plots_dir / "prediction_vs_actual.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
    plt.axhline(0, linestyle="--", color="red")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(plots_dir / "residual_plot.png", dpi=200)
    plt.close()


def _plot_feature_importance(
    best_model: Any,
    feature_names: list[str],
    plots_dir: Path,
) -> dict[str, float]:
    importances: Optional[np.ndarray] = None

    if hasattr(best_model, "named_steps"):
        estimator = best_model.named_steps["model"]
        selector = best_model.named_steps["feature_selector"]

        if hasattr(selector, "get_support"):
            support = selector.get_support()
            selected_names = np.array(feature_names)[support].tolist()
        else:
            selected_names = feature_names

        if hasattr(estimator, "feature_importances_"):
            importances = np.array(estimator.feature_importances_)
            fn = selected_names
        elif hasattr(estimator, "coef_"):
            coef = np.array(estimator.coef_)
            importances = np.abs(coef.ravel())
            fn = selected_names
        else:
            return {}
    elif hasattr(best_model, "feature_importances_"):
        importances = np.array(best_model.feature_importances_)
        fn = feature_names
    else:
        return {}

    if importances is None:
        return {}

    top_n = min(20, len(importances))
    order = np.argsort(importances)[-top_n:]
    top_features = np.array(fn)[order]
    top_values = importances[order]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_values, y=top_features, hue=top_features, legend=False, palette="crest")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance.png", dpi=200)
    plt.close()

    return {str(name): float(val) for name, val in zip(top_features.tolist(), top_values.tolist())}


def run_training_pipeline(data_path: Optional[str] = None) -> RunArtifacts:
    config = TrainConfig()
    timestamp = timestamp_now()

    model_path = MODELS_DIR / f"best_model_{timestamp}.joblib"
    metrics_path = MODELS_DIR / f"metrics_{timestamp}.json"
    report_path = MODELS_DIR / f"model_report_{timestamp}.json"
    plots_dir = MODELS_DIR / f"plots_{timestamp}"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_download_dataset(Path(data_path) if data_path else None)
    eda_report = run_eda(df, plots_dir)

    preprocessor = HousingPreprocessor()
    x, y, engineered_df = preprocessor.fit_transform(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    leaderboard, best_name, best_model, trained_models = train_models(
        x_train,
        y_train,
        x_test,
        y_test,
        config,
    )

    create_model_comparison_plot(leaderboard, plots_dir)
    create_prediction_diagnostics(best_model, x_test, y_test, plots_dir)

    feature_importance = _plot_feature_importance(best_model, preprocessor.get_feature_names(), plots_dir)

    model_bundle = {
        "best_model_name": best_name,
        "best_model": best_model,
        "all_models": trained_models,
        "preprocessor": preprocessor,
        "training_data": engineered_df,
        "feature_importance": feature_importance,
    }
    joblib.dump(model_bundle, model_path)

    summary = {
        "best_model": best_name,
        "leaderboard": leaderboard,
        "feature_importance": feature_importance,
        "eda": eda_report,
        "plots_dir": str(plots_dir),
        "model_path": str(model_path),
    }

    save_json(summary, metrics_path)
    save_json(summary, report_path)

    logger.info("Training completed. Best model: %s", best_name)
    logger.info("Artifacts saved at: %s", MODELS_DIR)

    return RunArtifacts(
        model_path=model_path,
        metrics_path=metrics_path,
        plots_dir=plots_dir,
        report_path=report_path,
    )


if __name__ == "__main__":
    run_training_pipeline()
