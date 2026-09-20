"""Configuration management and type definitions for ChurnGuard."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.churn.utils import load_yaml_config


@dataclass
class ProjectConfig:
    name: str = "ChurnGuard"
    version: str = "1.0.0"
    description: str = "Customer Segmentation & Churn Prediction System"
    seed: int = 42


@dataclass
class PathsConfig:
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    exports_data_dir: str = "data/exports"
    models_dir: str = "models"
    powerbi_exports_dir: str = "powerbi/exports"
    database_url: str = "sqlite:///data/churnguard.db"
    sqlite_path: str = "data/churnguard.db"
    raw_transactions_file: str = "data/raw/online_retail_transactions.csv"
    processed_transactions_file: str = "data/processed/cleaned_transactions.parquet"
    processed_transactions_sample: str = "data/processed/cleaned_transactions_sample.csv"
    customer_master_file: str = "data/processed/customer_master.csv"
    customer_master_parquet: str = "data/processed/customer_master.parquet"
    model_file: str = "models/model.pkl"
    explainer_file: str = "models/shap_explainer.pkl"
    metrics_file: str = "models/metrics.json"
    threshold_file: str = "models/threshold.json"


@dataclass
class BusinessTargetsConfig:
    churn_window_days: int = 90
    target_revenue_at_risk_ratio: float = 0.07
    revenue_at_risk_tolerance: float = 0.015
    target_pareto_revenue_share: float = 0.65
    pareto_customer_percentile: float = 0.20


@dataclass
class SyntheticDataConfig:
    n_customers: int = 8000
    n_transactions: int = 250000
    date_range_months: int = 24
    pareto_alpha: float = 1.18
    base_churn_rate: float = 0.26
    cancellation_rate: float = 0.055
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    iqr_factor: float = 3.0
    min_monetary: float = 0.01


@dataclass
class RFMConfig:
    quintile_bins: int = 5
    kmeans_k_min: int = 3
    kmeans_k_max: int = 7
    optimal_k: int = 4
    segment_names: list[str] = field(
        default_factory=lambda: [
            "Champions",
            "Loyal Customers",
            "Potential Loyalist",
            "New Customers",
            "Promising",
            "Need Attention",
            "At Risk",
            "Can't Lose Them",
            "Hibernating",
            "Lost",
        ]
    )


@dataclass
class ModelConfig:
    target_column: str = "churned"
    features: list[str] = field(
        default_factory=lambda: [
            "recency",
            "frequency",
            "monetary",
            "avg_order_value",
            "tenure_days",
            "orders_per_month",
            "return_rate",
            "rfm_score",
            "segment",
            "kmeans_cluster",
        ]
    )
    numerical_features: list[str] = field(
        default_factory=lambda: [
            "recency",
            "frequency",
            "monetary",
            "avg_order_value",
            "tenure_days",
            "orders_per_month",
            "return_rate",
            "rfm_score",
        ]
    )
    categorical_features: list[str] = field(default_factory=lambda: ["segment", "kmeans_cluster"])
    test_size: float = 0.20
    cv_folds: int = 5
    random_state: int = 42
    imbalance_method: str = "balanced"
    tuning_n_iter: int = 40
    tuning_scoring: str = "average_precision"
    param_distributions: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": [100, 150, 200, 250],
            "max_depth": [6, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.8],
        }
    )


@dataclass
class PaletteConfig:
    primary: str = "#4F46E5"
    secondary: str = "#14B8A6"
    success: str = "#10B981"
    warning: str = "#F59E0B"
    danger: str = "#EF4444"
    background: str = "#F7F8FC"
    sidebar: str = "#0E1726"
    card_bg: str = "#FFFFFF"
    border: str = "#E2E8F0"
    text_primary: str = "#1E293B"
    text_secondary: str = "#64748B"
    risk_colors: dict[str, str] = field(
        default_factory=lambda: {
            "Low": "#10B981",
            "Medium": "#F59E0B",
            "High": "#EF4444",
            "Critical": "#991B1B",
        }
    )


@dataclass
class Settings:
    """Master application configuration container."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    business_targets: BusinessTargetsConfig = field(default_factory=BusinessTargetsConfig)
    synthetic_data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    rfm: RFMConfig = field(default_factory=RFMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    palette: PaletteConfig = field(default_factory=PaletteConfig)

    @classmethod
    def load(cls, config_path: str | Path = "config.yaml") -> "Settings":
        """Factory method to construct Settings from config.yaml.

        Args:
            config_path: Path to configuration YAML.

        Returns:
            Instantiated Settings object.
        """
        raw = load_yaml_config(config_path)

        proj_data = raw.get("project", {})
        paths_data = raw.get("paths", {})
        targets_data = raw.get("business_targets", {})
        synth_data = raw.get("synthetic_data", {})
        prep_data = raw.get("preprocessing", {})
        rfm_data = raw.get("rfm", {})
        model_data = raw.get("model", {})
        palette_data = raw.get("palette", {})

        tuning_data = model_data.get("tuning", {})

        return cls(
            project=ProjectConfig(
                name=proj_data.get("name", "ChurnGuard"),
                version=proj_data.get("version", "1.0.0"),
                description=proj_data.get("description", ""),
                seed=proj_data.get("seed", 42),
            ),
            paths=PathsConfig(**paths_data) if paths_data else PathsConfig(),
            business_targets=(
                BusinessTargetsConfig(**targets_data) if targets_data else BusinessTargetsConfig()
            ),
            synthetic_data=(
                SyntheticDataConfig(**synth_data) if synth_data else SyntheticDataConfig()
            ),
            preprocessing=PreprocessingConfig(**prep_data) if prep_data else PreprocessingConfig(),
            rfm=RFMConfig(
                quintile_bins=rfm_data.get("quintile_bins", 5),
                kmeans_k_min=rfm_data.get("kmeans_k_min", 3),
                kmeans_k_max=rfm_data.get("kmeans_k_max", 7),
                optimal_k=rfm_data.get("optimal_k", 4),
                segment_names=rfm_data.get("segment_names", RFMConfig().segment_names),
            ),
            model=ModelConfig(
                target_column=model_data.get("target_column", "churned"),
                features=model_data.get("features", ModelConfig().features),
                numerical_features=model_data.get(
                    "numerical_features", ModelConfig().numerical_features
                ),
                categorical_features=model_data.get(
                    "categorical_features", ModelConfig().categorical_features
                ),
                test_size=model_data.get("test_size", 0.20),
                cv_folds=model_data.get("cv_folds", 5),
                random_state=model_data.get("random_state", 42),
                imbalance_method=model_data.get("imbalance_method", "balanced"),
                tuning_n_iter=tuning_data.get("n_iter", 40),
                tuning_scoring=tuning_data.get("scoring", "average_precision"),
                param_distributions=tuning_data.get(
                    "param_distributions", ModelConfig().param_distributions
                ),
            ),
            palette=PaletteConfig(**palette_data) if palette_data else PaletteConfig(),
        )


# Global settings singleton
settings = Settings.load()
