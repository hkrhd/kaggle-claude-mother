"""
システム設定管理

全エージェント・統合システムの設定を一元管理し、
環境別設定・セキュリティ・パフォーマンス調整を提供。
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml


class Environment(Enum):
    """実行環境"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """ログレベル"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AgentConfig:
    """エージェント設定"""
    enabled: bool = True
    max_concurrent_tasks: int = 3
    timeout_minutes: int = 60
    retry_attempts: int = 3
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """システム全体設定"""
    
    # 基本設定
    environment: Environment = Environment.DEVELOPMENT
    system_name: str = "Claude Mother System"
    version: str = "1.0.0"
    
    # GitHub設定
    github_token: str = ""
    repo_name: str = ""
    
    # ログ設定
    log_level: LogLevel = LogLevel.INFO
    log_file_path: str = "logs/system.log"
    max_log_file_size_mb: int = 100
    log_retention_days: int = 30
    
    # リソース制限
    max_concurrent_competitions: int = 3
    default_gpu_budget_hours: float = 50.0
    default_api_calls_limit: int = 10000
    max_execution_time_hours: float = 72.0
    
    # エージェント設定
    agent_configs: Dict[str, AgentConfig] = field(default_factory=dict)
    
    # セキュリティ設定
    api_rate_limit_per_hour: int = 5000
    max_issue_creation_per_hour: int = 100
    enable_audit_logging: bool = True
    
    # パフォーマンス設定
    monitoring_interval_seconds: int = 30
    health_check_interval_seconds: int = 300
    auto_scaling_enabled: bool = False
    
    # 通知設定
    notification_channels: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # 実験設定
    enable_experimental_features: bool = False
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    # 競技別設定
    competition_settings: Dict[str, Any] = field(default_factory=dict)
    
    # クラウドプラットフォーム設定
    cloud_platforms: Dict[str, Any] = field(default_factory=dict)
    
    # ストレージ設定
    storage: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # デフォルトエージェント設定
        if not self.agent_configs:
            self.agent_configs = {
                "planner": AgentConfig(
                    max_concurrent_tasks=2,
                    timeout_minutes=30,
                    resource_limits={"max_memory_mb": 2000}
                ),
                "analyzer": AgentConfig(
                    max_concurrent_tasks=3,
                    timeout_minutes=45,
                    resource_limits={"max_api_calls_per_hour": 1000}
                ),
                "executor": AgentConfig(
                    max_concurrent_tasks=5,
                    timeout_minutes=180,
                    resource_limits={"max_gpu_hours": 20.0}
                ),
                "monitor": AgentConfig(
                    max_concurrent_tasks=1,
                    timeout_minutes=10,
                    resource_limits={"monitoring_frequency_seconds": 30}
                ),
                "retrospective": AgentConfig(
                    max_concurrent_tasks=1,
                    timeout_minutes=60,
                    resource_limits={"max_analysis_depth": "comprehensive"}
                )
            }
        
        # デフォルトアラート閾値
        if not self.alert_thresholds:
            self.alert_thresholds = {
                "cpu_usage_percent": 80.0,
                "memory_usage_percent": 85.0,
                "gpu_usage_percent": 90.0,
                "error_rate_percent": 5.0,
                "api_rate_limit_remaining": 100
            }
        
        # デフォルト機能フラグ
        if not self.feature_flags:
            self.feature_flags = {
                "enable_advanced_monitoring": True,
                "enable_auto_scaling": False,
                "enable_predictive_analysis": True,
                "enable_multi_competition": True,
                "enable_experimental_techniques": False
            }


class ConfigManager:
    """設定管理システム"""
    
    def __init__(self, config_file_path: str = "config/system.yaml"):
        self.config_file_path = config_file_path
        self.logger = logging.getLogger(__name__)
        self._config = None
    
    def load_config(self) -> SystemConfig:
        """設定読み込み"""
        
        try:
            # 環境変数から設定上書き
            config_data = self._load_from_file()
            config_data = self._override_from_env(config_data)
            
            # SystemConfigオブジェクト作成
            self._config = self._create_config_object(config_data)
            
            self.logger.info(f"設定読み込み完了: {self.config_file_path}")
            return self._config
            
        except Exception as e:
            self.logger.error(f"設定読み込み失敗: {e}")
            # デフォルト設定で続行
            self._config = SystemConfig()
            return self._config
    
    def _load_from_file(self) -> Dict[str, Any]:
        """ファイルから設定読み込み"""
        
        if not os.path.exists(self.config_file_path):
            self.logger.warning(f"設定ファイルが見つかりません: {self.config_file_path}")
            return {}
        
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                if self.config_file_path.endswith('.yaml') or self.config_file_path.endswith('.yml'):
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f) or {}
                    
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            return {}
    
    def _override_from_env(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """環境変数からの設定上書き"""
        
        # 重要な設定を環境変数から取得
        env_mappings = {
            "GITHUB_TOKEN": ["github_token"],
            "REPO_NAME": ["repo_name"],
            "ENVIRONMENT": ["environment"],
            "LOG_LEVEL": ["log_level"],
            "MAX_GPU_HOURS": ["default_gpu_budget_hours"],
            "MAX_CONCURRENT_COMPETITIONS": ["max_concurrent_competitions"]
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # ネストした設定パスに対応
                current = config_data
                for key in config_path[:-1]:
                    current = current.setdefault(key, {})
                
                # 型変換
                final_key = config_path[-1]
                if final_key in ["max_concurrent_competitions"]:
                    current[final_key] = int(env_value)
                elif final_key in ["default_gpu_budget_hours"]:
                    current[final_key] = float(env_value)
                else:
                    current[final_key] = env_value
        
        return config_data
    
    def _create_config_object(self, config_data: Dict[str, Any]) -> SystemConfig:
        """設定オブジェクト作成"""
        
        # Enumの変換
        if "environment" in config_data:
            config_data["environment"] = Environment(config_data["environment"])
        
        if "log_level" in config_data:
            config_data["log_level"] = LogLevel(config_data["log_level"])
        
        # AgentConfig変換
        if "agent_configs" in config_data:
            agent_configs = {}
            for agent_name, agent_data in config_data["agent_configs"].items():
                agent_configs[agent_name] = AgentConfig(**agent_data)
            config_data["agent_configs"] = agent_configs
        
        return SystemConfig(**config_data)
    
    def save_config(self, config: SystemConfig):
        """設定保存"""
        
        try:
            # 設定をディクショナリに変換
            config_dict = self._config_to_dict(config)
            
            # ディレクトリ作成
            os.makedirs(os.path.dirname(self.config_file_path), exist_ok=True)
            
            # ファイル保存
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                if self.config_file_path.endswith('.yaml') or self.config_file_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"設定保存完了: {self.config_file_path}")
            
        except Exception as e:
            self.logger.error(f"設定保存失敗: {e}")
            raise
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """設定オブジェクトをディクショナリに変換"""
        
        result = {}
        
        # 基本フィールド
        for field_name, field_value in config.__dict__.items():
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            elif isinstance(field_value, dict) and field_name == "agent_configs":
                # AgentConfig特別処理
                agent_configs = {}
                for agent_name, agent_config in field_value.items():
                    agent_configs[agent_name] = agent_config.__dict__
                result[field_name] = agent_configs
            else:
                result[field_name] = field_value
        
        return result
    
    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """エージェント別設定取得"""
        
        if not self._config:
            self.load_config()
        
        return self._config.agent_configs.get(agent_name, AgentConfig())
    
    def update_agent_config(self, agent_name: str, config: AgentConfig):
        """エージェント設定更新"""
        
        if not self._config:
            self.load_config()
        
        self._config.agent_configs[agent_name] = config
        self.save_config(self._config)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """機能フラグ確認"""
        
        if not self._config:
            self.load_config()
        
        return self._config.feature_flags.get(feature_name, False)
    
    def get_alert_threshold(self, metric_name: str) -> float:
        """アラート閾値取得"""
        
        if not self._config:
            self.load_config()
        
        return self._config.alert_thresholds.get(metric_name, 0.0)
    
    def validate_config(self) -> List[str]:
        """設定検証"""
        
        if not self._config:
            self.load_config()
        
        errors = []
        
        # 必須設定チェック
        if not self._config.github_token:
            errors.append("GitHub token is required")
        
        if not self._config.repo_name:
            errors.append("Repository name is required")
        
        # リソース制限チェック
        if self._config.default_gpu_budget_hours <= 0:
            errors.append("GPU budget must be positive")
        
        if self._config.max_concurrent_competitions <= 0:
            errors.append("Max concurrent competitions must be positive")
        
        # エージェント設定チェック
        for agent_name, agent_config in self._config.agent_configs.items():
            if agent_config.timeout_minutes <= 0:
                errors.append(f"{agent_name} timeout must be positive")
        
        return errors


# グローバル設定管理インスタンス
config_manager = ConfigManager()

def get_config() -> SystemConfig:
    """システム設定取得"""
    return config_manager.load_config()

def get_agent_config(agent_name: str) -> AgentConfig:
    """エージェント設定取得"""
    return config_manager.get_agent_config(agent_name)

def is_feature_enabled(feature_name: str) -> bool:
    """機能フラグ確認"""
    return config_manager.is_feature_enabled(feature_name)