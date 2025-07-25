"""
Executor Agent システム

Kaggle競技自動化のための実行エージェント統合モジュール。
複数クラウド環境での並列実験実行・リソース最適化・ハイパーパラメータ調整を統合管理。
"""

from .executor_agent import ExecutorAgent

from .cloud_managers.kaggle_kernel_manager import KaggleKernelManager
from .cloud_managers.colab_execution_manager import ColabExecutionManager
from .cloud_managers.paperspace_manager import PaperspaceManager
from .cloud_managers.resource_optimizer import CloudResourceOptimizer

from .code_generators.notebook_generator import NotebookGenerator
from .code_generators.experiment_designer import ExperimentDesigner

from .execution_orchestrator.parallel_executor import ParallelExecutor

from .optimization.hyperparameter_tuner import HyperparameterTuner

__all__ = [
    "ExecutorAgent",
    "KaggleKernelManager",
    "ColabExecutionManager", 
    "PaperspaceManager",
    "CloudResourceOptimizer",
    "NotebookGenerator",
    "ExperimentDesigner",
    "ParallelExecutor",
    "HyperparameterTuner"
]

__version__ = "1.0.0"
__author__ = "Claude Mother System"
__description__ = "Multi-cloud Kaggle competition automation execution system"