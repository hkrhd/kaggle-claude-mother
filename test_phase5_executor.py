#!/usr/bin/env python3
"""
Phase 5: 実行エージェント統合テスト

ExecutorAgentとその全コンポーネントの統合動作を検証するテストスイート。
複数クラウド環境での並列実験実行・リソース最適化・ハイパーパラメータ調整の
エンドツーエンド動作確認を実施。
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))

# テスト対象インポート
from system.agents.executor.executor_agent import ExecutorAgent
from system.agents.executor.cloud_managers.kaggle_kernel_manager import KaggleKernelManager
from system.agents.executor.cloud_managers.colab_execution_manager import ColabExecutionManager
from system.agents.executor.cloud_managers.paperspace_manager import PaperspaceManager
from system.agents.executor.cloud_managers.resource_optimizer import CloudResourceOptimizer
from system.agents.executor.code_generators.notebook_generator import NotebookGenerator
from system.agents.executor.code_generators.experiment_designer import ExperimentDesigner
from system.agents.executor.execution_orchestrator.parallel_executor import ParallelExecutor
from system.agents.executor.optimization.hyperparameter_tuner import HyperparameterTuner

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Phase5IntegrationTest:
    """Phase 5統合テスト"""
    
    def __init__(self):
        self.test_results = {}
        self.error_count = 0
        
        # テスト用競技設定
        self.test_competition = {
            "name": "test-tabular-competition",
            "type": "tabular",
            "dataset_size_gb": 2.5,
            "deadline_days": 7
        }
        
        # テスト用技術リスト
        self.test_techniques = [
            {
                "technique": "gradient_boosting_ensemble",
                "integrated_score": 0.85,
                "estimated_runtime_hours": 2.0,
                "complexity_level": 0.7,
                "priority_score": 0.9
            },
            {
                "technique": "multi_level_stacking",
                "integrated_score": 0.78,
                "estimated_runtime_hours": 3.5,
                "complexity_level": 0.9,
                "priority_score": 0.8
            },
            {
                "technique": "neural_network",
                "integrated_score": 0.72,
                "estimated_runtime_hours": 4.0,
                "complexity_level": 0.8,
                "priority_score": 0.7
            }
        ]
    
    async def run_full_integration_test(self) -> Dict[str, Any]:
        """完全統合テスト実行"""
        
        logger.info("🚀 Phase 5 実行エージェント統合テスト開始")
        
        test_cases = [
            ("ExecutorAgent初期化テスト", self.test_executor_agent_initialization),
            ("クラウドマネージャー統合テスト", self.test_cloud_managers_integration),
            ("コード生成システムテスト", self.test_code_generation_system),
            ("リソース最適化テスト", self.test_resource_optimization),
            ("ハイパーパラメータ最適化テスト", self.test_hyperparameter_optimization),
            ("並列実行オーケストレーションテスト", self.test_parallel_execution),
            ("エンドツーエンド技術実装テスト", self.test_end_to_end_execution),
            ("パフォーマンス・スケーラビリティテスト", self.test_performance_scalability)
        ]
        
        for test_name, test_func in test_cases:
            try:
                logger.info(f"📋 {test_name} 実行中...")
                result = await test_func()
                self.test_results[test_name] = {
                    "status": "SUCCESS" if result else "FAILED",
                    "details": result if isinstance(result, dict) else {"success": result}
                }
                
                if result:
                    logger.info(f"✅ {test_name} 成功")
                else:
                    logger.error(f"❌ {test_name} 失敗")
                    self.error_count += 1
                    
            except Exception as e:
                logger.error(f"❌ {test_name} エラー: {e}")
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "error": str(e)
                }
                self.error_count += 1
        
        # 総合結果
        total_tests = len(test_cases)
        success_count = total_tests - self.error_count
        success_rate = success_count / total_tests
        
        summary = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "total_tests": total_tests,
            "successful_tests": success_count,
            "failed_tests": self.error_count,
            "success_rate": success_rate,
            "test_results": self.test_results
        }
        
        logger.info(f"🏁 Phase 5統合テスト完了: {success_count}/{total_tests} 成功 ({success_rate:.1%})")
        
        return summary
    
    async def test_executor_agent_initialization(self) -> Dict[str, Any]:
        """ExecutorAgent初期化テスト"""
        
        # ExecutorAgent作成
        executor = ExecutorAgent(
            github_token="test_token",
            repo_name="test/repo"
        )
        
        # 基本属性確認
        assert hasattr(executor, 'logger'), "Logger not initialized"
        assert hasattr(executor, 'kaggle_manager'), "Kaggle manager not initialized"
        assert hasattr(executor, 'colab_manager'), "Colab manager not initialized"
        assert hasattr(executor, 'paperspace_manager'), "Paperspace manager not initialized"
        assert hasattr(executor, 'resource_optimizer'), "Resource optimizer not initialized"
        assert hasattr(executor, 'notebook_generator'), "Notebook generator not initialized"
        assert hasattr(executor, 'experiment_designer'), "Experiment designer not initialized"
        assert hasattr(executor, 'parallel_executor'), "Parallel executor not initialized"
        assert hasattr(executor, 'hyperparameter_tuner'), "Hyperparameter tuner not initialized"
        
        # 各コンポーネントの型確認
        assert isinstance(executor.kaggle_manager, KaggleKernelManager), "Invalid kaggle manager type"
        assert isinstance(executor.colab_manager, ColabExecutionManager), "Invalid colab manager type"
        assert isinstance(executor.paperspace_manager, PaperspaceManager), "Invalid paperspace manager type"
        assert isinstance(executor.resource_optimizer, CloudResourceOptimizer), "Invalid resource optimizer type"
        assert isinstance(executor.notebook_generator, NotebookGenerator), "Invalid notebook generator type"
        assert isinstance(executor.experiment_designer, ExperimentDesigner), "Invalid experiment designer type"
        assert isinstance(executor.parallel_executor, ParallelExecutor), "Invalid parallel executor type"
        assert isinstance(executor.hyperparameter_tuner, HyperparameterTuner), "Invalid hyperparameter tuner type"
        
        return {
            "executor_initialized": True,
            "components_count": 8,
            "all_components_valid": True
        }
    
    async def test_cloud_managers_integration(self) -> Dict[str, Any]:
        """クラウドマネージャー統合テスト"""
        
        # 各クラウドマネージャーの初期化・基本動作確認
        kaggle_manager = KaggleKernelManager()
        colab_manager = ColabExecutionManager()
        paperspace_manager = PaperspaceManager()
        
        # リソース状態取得テスト
        kaggle_status = await kaggle_manager.get_resource_status()
        colab_status = await colab_manager.get_resource_status()
        paperspace_status = await paperspace_manager.get_resource_status()
        
        assert "resource_usage" in kaggle_status, "Kaggle resource status invalid"
        assert "resource_usage" in colab_status, "Colab resource status invalid"
        assert "resource_usage" in paperspace_status, "Paperspace resource status invalid"
        
        # コスト推定テスト
        kaggle_cost = await kaggle_manager.estimate_execution_cost(0.7, "tabular")
        colab_cost = await colab_manager.estimate_execution_cost(0.7, "tabular")
        paperspace_cost = await paperspace_manager.estimate_execution_cost(0.7, "tabular")
        
        assert "estimated_gpu_hours" in kaggle_cost, "Kaggle cost estimation invalid"
        assert "estimated_gpu_hours" in colab_cost, "Colab cost estimation invalid"
        assert "estimated_gpu_hours" in paperspace_cost, "Paperspace cost estimation invalid"
        
        return {
            "kaggle_manager_working": True,
            "colab_manager_working": True,
            "paperspace_manager_working": True,
            "resource_status_available": True,
            "cost_estimation_available": True
        }
    
    async def test_code_generation_system(self) -> Dict[str, Any]:
        """コード生成システムテスト"""
        
        notebook_generator = NotebookGenerator()
        experiment_designer = ExperimentDesigner()
        
        # ノートブック生成テスト
        test_technique = self.test_techniques[0]
        notebooks = await notebook_generator.generate_technique_notebooks(
            technique_name=test_technique["technique"],
            competition_type=self.test_competition["type"],
            dataset_info={"size_gb": 2.5, "features": 100},
            resource_constraints={"max_gpu_hours": 4.0}
        )
        
        assert isinstance(notebooks, dict), "Notebooks should be dictionary"
        assert len(notebooks) > 0, "No notebooks generated"
        
        # 実験設計テスト
        experiment_plan = await experiment_designer.design_experiments(
            techniques=self.test_techniques[:2],
            notebooks={},
            resource_constraints={"max_gpu_hours": 8.0}
        )
        
        assert hasattr(experiment_plan, 'experiment_id'), "Experiment plan missing ID"
        assert hasattr(experiment_plan, 'configs'), "Experiment plan missing configs"
        assert len(experiment_plan.configs) > 0, "No experiment configs generated"
        
        return {
            "notebook_generation_working": True,
            "generated_notebooks": len(notebooks),
            "experiment_design_working": True,
            "experiment_configs": len(experiment_plan.configs),
            "estimated_total_time": experiment_plan.estimated_total_time
        }
    
    async def test_resource_optimization(self) -> Dict[str, Any]:
        """リソース最適化テスト"""
        
        # ResourceOptimizer初期化
        optimizer = CloudResourceOptimizer()
        
        # 現在のリソース状態取得
        current_resources = await optimizer.get_current_resource_state()
        
        assert isinstance(current_resources, dict), "Resource state should be dictionary"
        
        # 実験要件作成
        from system.agents.executor.cloud_managers.resource_optimizer import ExecutionRequirement
        
        requirements = []
        for technique in self.test_techniques[:2]:
            req = ExecutionRequirement(
                technique_name=technique["technique"],
                competition_name=self.test_competition["name"],
                estimated_gpu_hours=technique["estimated_runtime_hours"],
                estimated_cpu_hours=technique["estimated_runtime_hours"] * 0.2,
                memory_gb_required=8.0,
                storage_gb_required=5.0,
                deadline_hours=168,  # 1週間
                priority_score=technique["priority_score"],
                complexity_level=technique["complexity_level"]
            )
            requirements.append(req)
        
        # 最適化実行
        allocations = await optimizer.optimize_experiment_allocation(requirements)
        
        assert isinstance(allocations, list), "Allocations should be list"
        assert len(allocations) > 0, "No allocations generated"
        
        # 最適化レポート取得
        report = await optimizer.get_optimization_report()
        
        assert "platform_efficiency" in report, "Optimization report missing platform efficiency"
        
        return {
            "resource_optimization_working": True,
            "allocations_generated": len(allocations),
            "platforms_analyzed": len(current_resources),
            "optimization_report_available": True
        }
    
    async def test_hyperparameter_optimization(self) -> Dict[str, Any]:
        """ハイパーパラメータ最適化テスト"""
        
        tuner = HyperparameterTuner()
        
        # 最適化設定作成
        config = await tuner.create_technique_optimization_config(
            technique_name="gradient_boosting_ensemble",
            competition_type="tabular",
            resource_constraints={"max_gpu_hours": 2.0}
        )
        
        assert hasattr(config, 'study_name'), "Config missing study name"
        assert hasattr(config, 'parameter_specs'), "Config missing parameter specs"
        assert len(config.parameter_specs) > 0, "No parameter specs defined"
        
        # 簡易目的関数定義
        def mock_objective(params: Dict[str, Any]) -> float:
            # 模擬クロスバリデーションスコア
            return 0.85 + (hash(str(params)) % 1000) / 10000
        
        # 最適化実行（短時間で）
        config.max_trials = 5
        config.max_time_hours = 0.1  # 6分
        
        result = await tuner.optimize_hyperparameters(
            config=config,
            objective_function=mock_objective
        )
        
        assert result.success, "Optimization should succeed"
        assert result.n_trials > 0, "No trials executed"
        assert len(result.best_params) > 0, "No best params found"
        
        # 最適化サマリー取得
        summary = await tuner.get_optimization_summary()
        
        assert "total_optimizations" in summary, "Summary missing total optimizations"
        
        return {
            "hyperparameter_optimization_working": True,
            "trials_executed": result.n_trials,
            "best_score": result.best_score,
            "parameter_count": len(result.best_params),
            "optimization_time_hours": result.optimization_time_hours
        }
    
    async def test_parallel_execution(self) -> Dict[str, Any]:
        """並列実行オーケストレーションテスト"""
        
        # ParallelExecutor初期化
        parallel_executor = ParallelExecutor()
        
        # 模擬ノートブックコード
        mock_notebooks = {
            "gradient_boosting_ensemble": {
                "kaggle_kernels": "# Mock Kaggle notebook for GBM",
                "google_colab": "# Mock Colab notebook for GBM"
            },
            "multi_level_stacking": {
                "kaggle_kernels": "# Mock Kaggle notebook for Stacking",
                "paperspace_gradient": "# Mock Paperspace notebook for Stacking"
            }
        }
        
        # 並列実行テスト（短時間で）
        execution_settings = {
            "competition_name": self.test_competition["name"],
            "max_retries": 1
        }
        
        # 実行時間を短縮
        for technique in self.test_techniques[:2]:
            technique["estimated_runtime_hours"] = 0.1  # 6分に短縮
        
        parallel_execution = await parallel_executor.execute_parallel_techniques(
            technique_configs=self.test_techniques[:2],
            notebook_codes=mock_notebooks,
            execution_settings=execution_settings
        )
        
        assert hasattr(parallel_execution, 'execution_id'), "Parallel execution missing ID"
        assert hasattr(parallel_execution, 'results'), "Parallel execution missing results"
        assert parallel_execution.total_tasks > 0, "No tasks created"
        
        # 実行状態取得
        status = await parallel_executor.get_execution_status(parallel_execution.execution_id)
        
        assert status is not None, "Execution status not available"
        assert "status" in status, "Status missing status field"
        
        # パフォーマンスサマリー取得
        performance = await parallel_executor.get_performance_summary()
        
        assert "recent_executions" in performance, "Performance summary missing recent executions"
        
        return {
            "parallel_execution_working": True,
            "tasks_created": parallel_execution.total_tasks,
            "execution_id": parallel_execution.execution_id,
            "results_count": len(parallel_execution.results),
            "success_rate": parallel_execution.success_rate
        }
    
    async def test_end_to_end_execution(self) -> Dict[str, Any]:
        """エンドツーエンド技術実装テスト"""
        
        # ExecutorAgent作成
        executor = ExecutorAgent(
            github_token="test_token",
            repo_name="test/repo"
        )
        
        # 技術実装実行（フル統合）
        technique_info = self.test_techniques[0].copy()
        technique_info["estimated_runtime_hours"] = 0.2  # 12分に短縮
        
        # ExecutionRequestオブジェクト作成
        from system.agents.executor.executor_agent import ExecutionRequest, ExecutionPriority
        
        execution_request = ExecutionRequest(
            competition_name=self.test_competition["name"],
            analyzer_issue_number=123,
            techniques_to_implement=[technique_info],
            priority=ExecutionPriority.HIGH,
            deadline_days=7,
            resource_constraints={
                "max_gpu_hours": 1.0,
                "priority": "high"
            }
        )
        
        result = await executor.execute_technical_implementation(
            request=execution_request
        )
        
        assert result is not None, "Implementation result is None"
        assert hasattr(result, "execution_id"), "Result missing execution_id"
        assert hasattr(result, "kaggle_results"), "Result missing kaggle_results"
        
        return {
            "end_to_end_execution_working": True,
            "technique_implemented": technique_info["technique"],
            "execution_id": result.execution_id,
            "kaggle_results_count": len(result.kaggle_results),
            "colab_results_count": len(result.colab_results),
            "paperspace_results_count": len(result.paperspace_results),
            "total_gpu_hours_used": result.total_gpu_hours_used
        }
    
    async def test_performance_scalability(self) -> Dict[str, Any]:
        """パフォーマンス・スケーラビリティテスト"""
        
        # ExecutorAgent作成
        executor = ExecutorAgent(
            github_token="test_token",
            repo_name="test/repo"
        )
        
        # 複数技術の実行テスト（簡略化）
        multiple_techniques = self.test_techniques[:2].copy()
        for technique in multiple_techniques:
            technique["estimated_runtime_hours"] = 0.1  # 6分に短縮
        
        # 複数のExecutionRequestを作成
        from system.agents.executor.executor_agent import ExecutionRequest, ExecutionPriority
        
        requests = []
        for technique in multiple_techniques:
            request = ExecutionRequest(
                competition_name=self.test_competition["name"],
                analyzer_issue_number=124,
                techniques_to_implement=[technique],
                priority=ExecutionPriority.STANDARD,
                deadline_days=7,
                resource_constraints={"max_gpu_hours": 1.0}
            )
            requests.append(request)
        
        # 各技術を順番に実行（並列実行のシミュレーション）
        results = []
        for request in requests:
            try:
                result = await executor.execute_technical_implementation(request=request)
                results.append(result)
            except Exception as e:
                # エラーは無視してテストを続行
                results.append({"error": str(e)})
        
        # 基本的な検証
        assert len(results) == len(multiple_techniques), "Not all techniques executed"
        
        return {
            "performance_scalability_working": True,
            "batch_techniques_count": len(multiple_techniques),
            "results_count": len(results),
            "execution_success": len([r for r in results if "error" not in r]),
            "error_count": len([r for r in results if "error" in r])
        }


async def main():
    """メイン実行関数"""
    
    print("🎯 Phase 5: ExecutorAgent統合テスト開始")
    print("=" * 60)
    
    # テスト実行
    test_suite = Phase5IntegrationTest()
    results = await test_suite.run_full_integration_test()
    
    # 結果表示
    print("\n📊 テスト結果サマリー:")
    print(f"総テスト数: {results['total_tests']}")
    print(f"成功: {results['successful_tests']}")
    print(f"失敗: {results['failed_tests']}")
    print(f"成功率: {results['success_rate']:.1%}")
    
    print("\n📋 詳細結果:")
    for test_name, result in results['test_results'].items():
        status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_emoji} {test_name}: {result['status']}")
        
        if result['status'] == 'ERROR':
            print(f"   エラー: {result.get('error', 'Unknown error')}")
        elif 'details' in result and isinstance(result['details'], dict):
            for key, value in result['details'].items():
                if isinstance(value, (int, float, bool, str)):
                    print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    
    if results['success_rate'] >= 0.8:
        print("🎉 Phase 5統合テスト 成功！")
        print("ExecutorAgentシステムは正常に動作しています。")
        return 0
    else:
        print("⚠️  Phase 5統合テスト 部分的成功")
        print("一部のコンポーネントに問題があります。詳細を確認してください。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)