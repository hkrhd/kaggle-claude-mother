"""
自動ノートブック生成システム

analyzerの技術分析結果を受けて、クラウド環境別に最適化された
実行可能なJupyter notebookコードを自動生成するシステム。
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re

class NotebookEnvironment(Enum):
    """ノートブック実行環境"""
    KAGGLE = "kaggle"
    COLAB = "colab"
    PAPERSPACE = "paperspace"

class TechniqueCategory(Enum):
    """技術カテゴリ"""
    ENSEMBLE = "ensemble"
    STACKING = "stacking"
    NEURAL_NETWORK = "neural_network"
    FEATURE_ENGINEERING = "feature_engineering"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"

@dataclass
class CodeTemplate:
    """コードテンプレート"""
    technique_name: str
    category: TechniqueCategory
    base_imports: List[str]
    environment_imports: Dict[NotebookEnvironment, List[str]]
    setup_code: str
    main_implementation: str
    evaluation_code: str
    submission_code: str
    gpu_requirements: bool = False
    estimated_runtime_minutes: int = 60

@dataclass
class NotebookConfig:
    """ノートブック設定"""
    competition_name: str
    competition_type: str  # "tabular", "computer_vision", "nlp", "time_series"
    target_environment: NotebookEnvironment
    techniques: List[str]
    dataset_info: Dict[str, Any] = None
    performance_target: float = 0.8
    time_limit_hours: int = 2
    gpu_enabled: bool = True

class NotebookGenerator:
    """自動ノートブック生成エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 技術別コードテンプレート
        self.technique_templates = self._initialize_technique_templates()
        
        # 環境別設定
        self.environment_configs = {
            NotebookEnvironment.KAGGLE: {
                "data_path_prefix": "/kaggle/input/",
                "output_path": "/kaggle/working/",
                "mount_code": "",
                "gpu_setup": "# GPU already configured in Kaggle",
                "submission_path": "/kaggle/working/submission.csv"
            },
            NotebookEnvironment.COLAB: {
                "data_path_prefix": "/content/drive/MyDrive/kaggle_data/",
                "output_path": "/content/drive/MyDrive/kaggle_output/",
                "mount_code": "from google.colab import drive\ndrive.mount('/content/drive')",
                "gpu_setup": "# GPU configured via Colab runtime settings", 
                "submission_path": "/content/drive/MyDrive/kaggle_output/submission.csv"
            },
            NotebookEnvironment.PAPERSPACE: {
                "data_path_prefix": "/datasets/",
                "output_path": "/outputs/",
                "mount_code": "",
                "gpu_setup": "import torch\nprint(f'GPU Available: {torch.cuda.is_available()}')",
                "submission_path": "/outputs/submission.csv"
            }
        }
    
    def _initialize_technique_templates(self) -> Dict[str, CodeTemplate]:
        """技術別テンプレート初期化"""
        
        templates = {}
        
        # Gradient Boosting Ensemble
        templates["gradient_boosting_ensemble"] = CodeTemplate(
            technique_name="gradient_boosting_ensemble",
            category=TechniqueCategory.ENSEMBLE,
            base_imports=[
                "import pandas as pd",
                "import numpy as np",
                "from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold",
                "from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error",
                "import xgboost as xgb",
                "import lightgbm as lgb",
                "import catboost as cb",
                "import warnings",
                "warnings.filterwarnings('ignore')"
            ],
            environment_imports={
                NotebookEnvironment.KAGGLE: ["from kaggle.competitions import competitions"],
                NotebookEnvironment.COLAB: ["!pip install xgboost lightgbm catboost"],
                NotebookEnvironment.PAPERSPACE: ["import os", "os.system('pip install xgboost lightgbm catboost')"]
            },
            setup_code="""
# Data loading and basic preprocessing
def load_data(data_path):
    train = pd.read_csv(f'{data_path}train.csv')
    test = pd.read_csv(f'{data_path}test.csv')
    return train, test

def basic_preprocessing(train, test):
    # Handle missing values
    numeric_columns = train.select_dtypes(include=[np.number]).columns
    train[numeric_columns] = train[numeric_columns].fillna(train[numeric_columns].median())
    test[numeric_columns] = test[numeric_columns].fillna(train[numeric_columns].median())
    
    # Encode categorical variables
    categorical_columns = train.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col not in ['id', 'target']:
            train[col] = train[col].astype('category').cat.codes
            test[col] = test[col].astype('category').cat.codes
    
    return train, test
""",
            main_implementation="""
# Multi-algorithm ensemble implementation
class GradientBoostingEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
        
        # XGBoost
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # LightGBM  
        self.models['lgb'] = lgb.LGBMRegressor(
            n_estimators=1000,
            num_leaves=31,
            learning_rate=0.01,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42
        )
        
        # CatBoost
        self.models['catboost'] = cb.CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.01,
            random_seed=42,
            verbose=False
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if X_val is not None:
                # Early stopping with validation set
                if name == 'xgb':
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)], 
                             early_stopping_rounds=50,
                             verbose=False)
                elif name == 'lgb':
                    model.fit(X_train, y_train,
                             eval_set=[(X_val, y_val)],
                             early_stopping_rounds=50,
                             verbose=False)
                else:
                    model.fit(X_train, y_train,
                             eval_set=[(X_val, y_val)],
                             early_stopping_rounds=50)
            else:
                model.fit(X_train, y_train)
            
            # Cross-validation scoring
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            scores[name] = -cv_scores.mean()
            print(f"{name} CV Score: {scores[name]:.4f}")
        
        # Calculate ensemble weights based on performance
        total_score = sum(1/score for score in scores.values())
        for name, score in scores.items():
            self.weights[name] = (1/score) / total_score
            print(f"{name} weight: {self.weights[name]:.3f}")
    
    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += pred * self.weights[name]
        
        return ensemble_pred

# Model training and prediction
ensemble = GradientBoostingEnsemble()
""",
            evaluation_code="""
# Model evaluation with cross-validation
def evaluate_model(model, X, y, cv_folds=5):
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        predictions = model.predict(X_fold_val)
        
        score = mean_squared_error(y_fold_val, predictions, squared=False)
        cv_scores.append(score)
        print(f"Fold {fold + 1} RMSE: {score:.4f}")
    
    print(f"\\nMean CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    return np.mean(cv_scores)
""",
            submission_code="""
# Generate submission
def generate_submission(model, test_data, submission_path):
    predictions = model.predict(test_data)
    
    submission = pd.DataFrame({
        'id': test_data.index,
        'target': predictions
    })
    
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    return submission
""",
            gpu_requirements=False,
            estimated_runtime_minutes=90
        )
        
        # Multi-level Stacking
        templates["multi_level_stacking"] = CodeTemplate(
            technique_name="multi_level_stacking",
            category=TechniqueCategory.STACKING,
            base_imports=[
                "import pandas as pd",
                "import numpy as np",
                "from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold",
                "from sklearn.linear_model import Ridge, Lasso",
                "from sklearn.ensemble import RandomForestRegressor",
                "from sklearn.metrics import mean_squared_error",
                "import xgboost as xgb",
                "import lightgbm as lgb",
                "from mlxtend.regressor import StackingRegressor"
            ],
            environment_imports={
                NotebookEnvironment.KAGGLE: [],
                NotebookEnvironment.COLAB: ["!pip install mlxtend"],
                NotebookEnvironment.PAPERSPACE: ["import os", "os.system('pip install mlxtend')"]
            },
            setup_code="""
# Multi-level stacking implementation
from sklearn.base import BaseEstimator, RegressorMixin

class MultiLevelStacking(BaseEstimator, RegressorMixin):
    def __init__(self):
        # Level 1 models (base learners)
        self.level1_models = {
            'xgb': xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.01),
            'lgb': lgb.LGBMRegressor(n_estimators=500, num_leaves=15, learning_rate=0.01),
            'rf': RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
        }
        
        # Level 2 models (meta learners)
        self.level2_models = {
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        # Level 3 model (final blender)
        self.level3_model = Ridge(alpha=0.5)
        
        self.level1_predictions = {}
        self.level2_predictions = {}
""",
            main_implementation="""
    def fit(self, X, y):
        # Level 1: Train base models with cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.level1_models.items():
            print(f"Training Level 1 model: {name}")
            
            # Out-of-fold predictions for level 2
            oof_predictions = np.zeros(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Fit model on fold training data
                model.fit(X_fold_train, y_fold_train)
                
                # Predict on fold validation data
                oof_predictions[val_idx] = model.predict(X_fold_val)
            
            self.level1_predictions[name] = oof_predictions
            
            # Refit on full dataset for prediction
            model.fit(X, y)
        
        # Level 2: Train meta models on level 1 predictions
        level1_features = pd.DataFrame(self.level1_predictions)
        
        for name, model in self.level2_models.items():
            print(f"Training Level 2 model: {name}")
            
            # Out-of-fold predictions for level 3
            oof_predictions = np.zeros(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_fold_train = level1_features.iloc[train_idx]
                X_fold_val = level1_features.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                oof_predictions[val_idx] = model.predict(X_fold_val)
            
            self.level2_predictions[name] = oof_predictions
            
            # Refit on full level 1 predictions
            model.fit(level1_features, y)
        
        # Level 3: Train final blender
        level2_features = pd.DataFrame(self.level2_predictions)
        print("Training Level 3 final blender")
        self.level3_model.fit(level2_features, y)
        
        return self
    
    def predict(self, X):
        # Level 1 predictions
        level1_preds = {}
        for name, model in self.level1_models.items():
            level1_preds[name] = model.predict(X)
        
        level1_features = pd.DataFrame(level1_preds)
        
        # Level 2 predictions
        level2_preds = {}
        for name, model in self.level2_models.items():
            level2_preds[name] = model.predict(level1_features)
        
        level2_features = pd.DataFrame(level2_preds)
        
        # Level 3 final prediction
        final_predictions = self.level3_model.predict(level2_features)
        
        return final_predictions

# Initialize multi-level stacking model
stacking_model = MultiLevelStacking()
""",
            evaluation_code="""
# Evaluate stacking model
def evaluate_stacking(model, X, y, cv_folds=3):
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\\nEvaluating fold {fold + 1}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit stacking model
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_val)
        
        # Calculate score
        score = mean_squared_error(y_val, predictions, squared=False)
        cv_scores.append(score)
        print(f"Fold {fold + 1} RMSE: {score:.4f}")
    
    print(f"\\nStacking Model CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    return np.mean(cv_scores)
""",
            submission_code="""
# Generate stacking submission
def generate_stacking_submission(model, test_data, submission_path):
    print("Generating stacking predictions...")
    predictions = model.predict(test_data)
    
    submission = pd.DataFrame({
        'id': test_data.index,
        'target': predictions
    })
    
    submission.to_csv(submission_path, index=False)
    print(f"Stacking submission saved to {submission_path}")
    print(f"Prediction stats: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    
    return submission
""",
            gpu_requirements=False,
            estimated_runtime_minutes=150
        )
        
        # Feature Engineering template
        templates["feature_engineering_automated"] = CodeTemplate(
            technique_name="feature_engineering_automated",
            category=TechniqueCategory.FEATURE_ENGINEERING,
            base_imports=[
                "import pandas as pd",
                "import numpy as np",
                "from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures",
                "from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression",
                "from sklearn.decomposition import PCA",
                "import itertools"
            ],
            environment_imports={
                NotebookEnvironment.KAGGLE: [],
                NotebookEnvironment.COLAB: ["!pip install feature-engine"],
                NotebookEnvironment.PAPERSPACE: ["import os", "os.system('pip install feature-engine')"]
            },
            setup_code="""
# Automated feature engineering class
class AutomatedFeatureEngineering:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.generated_features = []
        
    def generate_interaction_features(self, df, numeric_cols, max_interactions=100):
        print("Generating interaction features...")
        interaction_features = []
        
        # Pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                if len(interaction_features) >= max_interactions:
                    break
                
                # Multiplication
                feature_name = f"{col1}_x_{col2}"
                df[feature_name] = df[col1] * df[col2]
                interaction_features.append(feature_name)
                
                # Division (with zero handling)
                if not (df[col2] == 0).any():
                    feature_name = f"{col1}_div_{col2}"
                    df[feature_name] = df[col1] / df[col2]
                    interaction_features.append(feature_name)
        
        print(f"Generated {len(interaction_features)} interaction features")
        return interaction_features
""",
            main_implementation="""
    def apply_feature_engineering(self, train_df, test_df, target_col):
        print("Starting automated feature engineering...")
        
        # Identify column types
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Handle missing values
        print("Handling missing values...")
        for col in numeric_cols:
            median_val = train_df[col].median()
            train_df[col].fillna(median_val, inplace=True)
            test_df[col].fillna(median_val, inplace=True)
        
        for col in categorical_cols:
            mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else 'Unknown'
            train_df[col].fillna(mode_val, inplace=True)
            test_df[col].fillna(mode_val, inplace=True)
        
        # 2. Encode categorical variables
        print("Encoding categorical variables...")
        for col in categorical_cols:
            encoder = LabelEncoder()
            combined_values = pd.concat([train_df[col], test_df[col]], axis=0)
            encoder.fit(combined_values)
            
            train_df[col + '_encoded'] = encoder.transform(train_df[col])
            test_df[col + '_encoded'] = encoder.transform(test_df[col])
            self.encoders[col] = encoder
        
        # 3. Generate interaction features
        interaction_features = self.generate_interaction_features(train_df, numeric_cols[:10])  # Limit to top 10
        for feature in interaction_features:
            test_df[feature] = test_df[feature.split('_x_')[0]] * test_df[feature.split('_x_')[1]] if '_x_' in feature else test_df[feature.split('_div_')[0]] / test_df[feature.split('_div_')[1]]
        
        # 4. Polynomial features (degree 2, limited)
        print("Generating polynomial features...")
        top_features = numeric_cols[:5]  # Top 5 numeric features
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        
        poly_train = poly.fit_transform(train_df[top_features])
        poly_test = poly.transform(test_df[top_features])
        
        poly_feature_names = [f"poly_{i}" for i in range(poly_train.shape[1] - len(top_features))]
        
        for i, name in enumerate(poly_feature_names):
            train_df[name] = poly_train[:, len(top_features) + i]
            test_df[name] = poly_test[:, len(top_features) + i]
        
        # 5. Statistical features
        print("Generating statistical features...")
        train_df['numeric_mean'] = train_df[numeric_cols].mean(axis=1)
        train_df['numeric_std'] = train_df[numeric_cols].std(axis=1)
        train_df['numeric_skew'] = train_df[numeric_cols].skew(axis=1)
        
        test_df['numeric_mean'] = test_df[numeric_cols].mean(axis=1)
        test_df['numeric_std'] = test_df[numeric_cols].std(axis=1)
        test_df['numeric_skew'] = test_df[numeric_cols].skew(axis=1)
        
        # 6. Feature selection
        print("Selecting best features...")
        all_features = [col for col in train_df.columns if col != target_col and col not in ['id']]
        
        selector = SelectKBest(score_func=f_regression, k=min(100, len(all_features)))
        X_selected = selector.fit_transform(train_df[all_features], train_df[target_col])
        
        selected_features = [all_features[i] for i in selector.get_support(indices=True)]
        print(f"Selected {len(selected_features)} features")
        
        self.selected_features = selected_features
        
        return train_df[selected_features], test_df[selected_features]

# Initialize feature engineering
feature_engineer = AutomatedFeatureEngineering()
""",
            evaluation_code="""
# Evaluate feature engineering impact
def evaluate_feature_impact(original_features, engineered_features, target, model_class):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    # Baseline model with original features
    baseline_model = model_class()
    baseline_scores = cross_val_score(baseline_model, original_features, target, cv=5, scoring='neg_mean_squared_error')
    baseline_rmse = np.sqrt(-baseline_scores.mean())
    
    # Enhanced model with engineered features  
    enhanced_model = model_class()
    enhanced_scores = cross_val_score(enhanced_model, engineered_features, target, cv=5, scoring='neg_mean_squared_error')
    enhanced_rmse = np.sqrt(-enhanced_scores.mean())
    
    improvement = (baseline_rmse - enhanced_rmse) / baseline_rmse * 100
    
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Enhanced RMSE: {enhanced_rmse:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    return {
        'baseline_rmse': baseline_rmse,
        'enhanced_rmse': enhanced_rmse,
        'improvement_percent': improvement
    }
""",
            submission_code="""
# Generate submission with engineered features
def generate_feature_submission(model, test_features, submission_path):
    predictions = model.predict(test_features)
    
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'target': predictions
    })
    
    submission.to_csv(submission_path, index=False)
    print(f"Feature-engineered submission saved to {submission_path}")
    print(f"Features used: {test_features.shape[1]}")
    
    return submission
""",
            gpu_requirements=False,
            estimated_runtime_minutes=120
        )
        
        return templates
    
    async def generate_technique_notebooks(
        self,
        technique_name: str,
        competition_type: str,
        target_environments: List[str],
        config: Optional[NotebookConfig] = None
    ) -> Dict[str, str]:
        """技術別ノートブック生成"""
        
        self.logger.info(f"ノートブック生成開始: {technique_name}")
        
        # テンプレート取得
        template = self.technique_templates.get(technique_name)
        if not template:
            raise ValueError(f"Unknown technique: {technique_name}")
        
        notebooks = {}
        
        for env_str in target_environments:
            try:
                environment = NotebookEnvironment(env_str)
                notebook_code = await self._generate_environment_notebook(
                    template, environment, competition_type, config
                )
                notebooks[env_str] = notebook_code
                
            except ValueError:
                self.logger.warning(f"Unknown environment: {env_str}")
                continue
        
        self.logger.info(f"ノートブック生成完了: {len(notebooks)}環境")
        return notebooks
    
    async def _generate_environment_notebook(
        self,
        template: CodeTemplate,
        environment: NotebookEnvironment,
        competition_type: str,
        config: Optional[NotebookConfig]
    ) -> str:
        """環境別ノートブック生成"""
        
        env_config = self.environment_configs[environment]
        
        # ノートブック構造構築
        notebook_cells = []
        
        # 1. Header cell
        header = self._generate_header_cell(template, environment, competition_type)
        notebook_cells.append(header)
        
        # 2. Environment setup cell
        setup_cell = self._generate_setup_cell(template, environment, env_config)
        notebook_cells.append(setup_cell)
        
        # 3. Data loading cell
        data_loading_cell = self._generate_data_loading_cell(environment, env_config, competition_type)
        notebook_cells.append(data_loading_cell)
        
        # 4. Preprocessing cell
        preprocessing_cell = self._generate_preprocessing_cell(template, competition_type)
        notebook_cells.append(preprocessing_cell)
        
        # 5. Main implementation cell
        main_cell = self._generate_main_implementation_cell(template)
        notebook_cells.append(main_cell)
        
        # 6. Training cell
        training_cell = self._generate_training_cell(template, environment)
        notebook_cells.append(training_cell)
        
        # 7. Evaluation cell
        evaluation_cell = self._generate_evaluation_cell(template)
        notebook_cells.append(evaluation_cell)
        
        # 8. Submission cell
        submission_cell = self._generate_submission_cell(template, env_config)
        notebook_cells.append(submission_cell)
        
        # Jupyter notebook JSON構造作成
        notebook_json = {
            "nbformat": 4,
            "nbformat_minor": 2,
            "metadata": self._generate_notebook_metadata(environment, template),
            "cells": notebook_cells
        }
        
        return json.dumps(notebook_json, indent=2)
    
    def _generate_header_cell(self, template: CodeTemplate, environment: NotebookEnvironment, competition_type: str) -> Dict[str, Any]:
        """ヘッダーセル生成"""
        
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {template.technique_name.replace('_', ' ').title()}\n",
                f"\n",
                f"**Competition Type:** {competition_type}\n",
                f"**Environment:** {environment.value}\n",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                f"**Estimated Runtime:** {template.estimated_runtime_minutes} minutes\n",
                f"**GPU Required:** {'Yes' if template.gpu_requirements else 'No'}\n",
                f"\n",
                f"---\n",
                f"\n",
                f"This notebook implements {template.technique_name} using automated code generation.\n"
            ]
        }
    
    def _generate_setup_cell(self, template: CodeTemplate, environment: NotebookEnvironment, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """セットアップセル生成"""
        
        code_lines = []
        
        # Environment-specific mount/setup
        if env_config["mount_code"]:
            code_lines.append(env_config["mount_code"])
            code_lines.append("")
        
        # Base imports
        code_lines.extend(template.base_imports)
        code_lines.append("")
        
        # Environment-specific imports
        env_imports = template.environment_imports.get(environment, [])
        if env_imports:
            code_lines.extend(env_imports)
            code_lines.append("")
        
        # GPU setup
        if template.gpu_requirements:
            code_lines.append(env_config["gpu_setup"])
            code_lines.append("")
        
        # Configuration
        code_lines.extend([
            f"# Environment configuration",
            f"DATA_PATH = '{env_config['data_path_prefix']}'",
            f"OUTPUT_PATH = '{env_config['output_path']}'",
            f"SUBMISSION_PATH = '{env_config['submission_path']}'",
            "",
            "print(f'Environment: {environment.value}')",
            "print(f'Data path: {DATA_PATH}')",
            "print(f'Output path: {OUTPUT_PATH}')"
        ])
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        }
    
    def _generate_data_loading_cell(self, environment: NotebookEnvironment, env_config: Dict[str, Any], competition_type: str) -> Dict[str, Any]:
        """データ読み込みセル生成"""
        
        code_lines = [
            "# Data loading",
            "print('Loading data...')",
            "",
            "try:",
            "    train_df = pd.read_csv(f'{DATA_PATH}train.csv')",
            "    test_df = pd.read_csv(f'{DATA_PATH}test.csv')",
            "    ",
            "    print(f'Train shape: {train_df.shape}')",
            "    print(f'Test shape: {test_df.shape}')",
            "    ",
            "    # Display basic info",
            "    print('\\nTrain data info:')",
            "    print(train_df.info())",
            "    print('\\nFirst few rows:')",
            "    print(train_df.head())",
            "    ",
            "except FileNotFoundError as e:",
            "    print(f'Data file not found: {e}')",
            "    print('Please ensure the competition data is available in the specified path')",
            "    # Create dummy data for testing",
            "    print('Creating dummy data for testing...')",
            "    train_df = pd.DataFrame({",
            "        'feature_1': np.random.randn(1000),",
            "        'feature_2': np.random.randn(1000),",
            "        'target': np.random.randn(1000)",
            "    })",
            "    test_df = pd.DataFrame({",
            "        'feature_1': np.random.randn(500),",
            "        'feature_2': np.random.randn(500)",
            "    })"
        ]
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        }
    
    def _generate_preprocessing_cell(self, template: CodeTemplate, competition_type: str) -> Dict[str, Any]:
        """前処理セル生成"""
        
        code_lines = [
            "# Data preprocessing",
            template.setup_code,
            "",
            "# Apply preprocessing",
            "train_processed, test_processed = basic_preprocessing(train_df, test_df)",
            "",
            "print('Preprocessing completed')",
            "print(f'Processed train shape: {train_processed.shape}')",
            "print(f'Processed test shape: {test_processed.shape}')"
        ]
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        }
    
    def _generate_main_implementation_cell(self, template: CodeTemplate) -> Dict[str, Any]:
        """メイン実装セル生成"""
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": template.main_implementation.split('\n')
        }
    
    def _generate_training_cell(self, template: CodeTemplate, environment: NotebookEnvironment) -> Dict[str, Any]:
        """学習セル生成"""
        
        code_lines = [
            "# Model training",
            "print('Starting model training...')",
            "",
            "# Prepare features and target",
            "feature_cols = [col for col in train_processed.columns if col not in ['target', 'id']]",
            "X = train_processed[feature_cols]",
            "y = train_processed['target'] if 'target' in train_processed.columns else train_processed.iloc[:, -1]",
            "",
            "# Split for validation",
            "from sklearn.model_selection import train_test_split",
            "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)",
            "",
            "# Train model",
            f"model = {template.technique_name.split('_')[0]}_model",  # Simplified model reference
            "model.fit(X_train, y_train, X_val, y_val)",
            "",
            "print('Model training completed')"
        ]
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        }
    
    def _generate_evaluation_cell(self, template: CodeTemplate) -> Dict[str, Any]:
        """評価セル生成"""
        
        code_lines = [
            "# Model evaluation",
            template.evaluation_code,
            "",
            "# Run evaluation",
            "cv_score = evaluate_model(model, X, y)",
            "print(f'Final CV Score: {cv_score:.4f}')"
        ]
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        }
    
    def _generate_submission_cell(self, template: CodeTemplate, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """提出セル生成"""
        
        code_lines = [
            "# Generate submission",
            template.submission_code,
            "",
            "# Create submission",
            "test_features = test_processed[[col for col in test_processed.columns if col in feature_cols]]",
            "submission = generate_submission(model, test_features, SUBMISSION_PATH)",
            "",
            "print('Submission generation completed!')"
        ]
        
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        }
    
    def _generate_notebook_metadata(self, environment: NotebookEnvironment, template: CodeTemplate) -> Dict[str, Any]:
        """ノートブックメタデータ生成"""
        
        metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        }
        
        # Environment-specific metadata
        if environment == NotebookEnvironment.KAGGLE:
            metadata["kaggle"] = {
                "accelerator": "gpu" if template.gpu_requirements else "none",
                "dataSources": [],
                "dockerImageVersionId": 30127,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": template.gpu_requirements
            }
        
        elif environment == NotebookEnvironment.COLAB:
            metadata["colab"] = {
                "name": f"{template.technique_name}_colab.ipynb",
                "provenance": [],
                "collapsed_sections": [],
                "machine_shape": "hm" if template.gpu_requirements else "standard"
            }
            if template.gpu_requirements:
                metadata["accelerator"] = "GPU"
        
        elif environment == NotebookEnvironment.PAPERSPACE:
            metadata["paperspace"] = {
                "machine_type": "Free-GPU" if template.gpu_requirements else "Free-CPU",
                "runtime": "python3"
            }
        
        return metadata
    
    async def generate_ensemble_notebook(
        self,
        techniques: List[str],
        competition_type: str,
        environment: NotebookEnvironment,
        ensemble_method: str = "weighted_average"
    ) -> str:
        """アンサンブルノートブック生成"""
        
        self.logger.info(f"アンサンブルノートブック生成: {len(techniques)}技術")
        
        # アンサンブル用テンプレート作成
        ensemble_template = self._create_ensemble_template(techniques, ensemble_method)
        
        # 環境別ノートブック生成
        notebook_code = await self._generate_environment_notebook(
            ensemble_template, environment, competition_type, None
        )
        
        return notebook_code
    
    def _create_ensemble_template(self, techniques: List[str], ensemble_method: str) -> CodeTemplate:
        """アンサンブル用テンプレート作成"""
        
        base_imports = [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.model_selection import cross_val_score, KFold",
            "from sklearn.metrics import mean_squared_error",
            "from sklearn.linear_model import Ridge",
            "import warnings",
            "warnings.filterwarnings('ignore')"
        ]
        
        # 各技術の最適な重みを学習するコード
        main_implementation = f"""
class TechniqueEnsemble:
    def __init__(self, techniques={techniques}):
        self.techniques = techniques
        self.models = {{}}
        self.weights = {{}}
        self.meta_model = Ridge(alpha=1.0)
        
        # Initialize individual models based on techniques
        for technique in techniques:
            if 'gradient_boosting' in technique:
                import xgboost as xgb
                self.models[technique] = xgb.XGBRegressor()
            elif 'stacking' in technique:
                from sklearn.ensemble import RandomForestRegressor
                self.models[technique] = RandomForestRegressor()
            # Add more technique mappings as needed
    
    def fit(self, X, y):
        # Train individual models and collect predictions
        predictions = []
        
        for name, model in self.models.items():
            print(f"Training {{name}}...")
            model.fit(X, y)
            pred = model.predict(X)
            predictions.append(pred)
        
        # Stack predictions for meta-learning
        stacked_predictions = np.column_stack(predictions)
        
        # Train meta-model on stacked predictions
        self.meta_model.fit(stacked_predictions, y)
        
        return self
    
    def predict(self, X):
        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Stack and use meta-model
        stacked_predictions = np.column_stack(predictions)
        ensemble_prediction = self.meta_model.predict(stacked_predictions)
        
        return ensemble_prediction

# Initialize ensemble
ensemble_model = TechniqueEnsemble()
"""
        
        return CodeTemplate(
            technique_name=f"ensemble_{len(techniques)}_techniques",
            category=TechniqueCategory.ENSEMBLE,
            base_imports=base_imports,
            environment_imports={},
            setup_code="# Ensemble of multiple techniques",
            main_implementation=main_implementation,
            evaluation_code="# Evaluate ensemble performance",
            submission_code="# Generate ensemble submission",
            gpu_requirements=any("neural" in tech for tech in techniques),
            estimated_runtime_minutes=60 * len(techniques)
        )