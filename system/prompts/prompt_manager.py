"""
プロンプト管理システム

各エージェントのLLMプロンプトを.mdファイルから読み込み、
動的な変数置換・バージョン管理・最適化を行う統合管理システム。
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """プロンプトタイプ"""
    COMPETITION_SELECTION = "competition_selection"
    TECHNICAL_ANALYSIS = "technical_analysis"
    SUBMISSION_DECISION = "submission_decision"
    ANOMALY_DIAGNOSIS = "anomaly_diagnosis"
    RESOURCE_OPTIMIZATION = "resource_optimization"


@dataclass
class PromptTemplate:
    """プロンプトテンプレート"""
    name: str
    prompt_type: PromptType
    template_content: str
    variables: Dict[str, Any]
    version: str
    created_at: str
    optimized_for_medal: bool = True


class PromptManager:
    """プロンプト管理マネージャー"""
    
    def __init__(self, prompts_dir: str = None):
        self.logger = logging.getLogger(__name__)
        
        # プロンプトディレクトリ設定
        if prompts_dir is None:
            current_dir = Path(__file__).parent
            self.prompts_dir = current_dir
        else:
            self.prompts_dir = Path(prompts_dir)
        
        # プロンプトキャッシュ
        self.prompt_cache: Dict[str, PromptTemplate] = {}
        
        # メダル獲得最適化設定
        self.medal_optimization_enabled = True
        
        self.logger.info(f"プロンプト管理システム初期化: {self.prompts_dir}")
    
    def load_prompt(
        self, 
        prompt_type: PromptType, 
        agent_name: str = None
    ) -> Optional[PromptTemplate]:
        """プロンプト読み込み"""
        
        try:
            # プロンプトファイル名決定
            if agent_name:
                filename = f"{agent_name}_{prompt_type.value}.md"
            else:
                filename = f"{prompt_type.value}.md"
            
            filepath = self.prompts_dir / filename
            
            # キャッシュ確認
            cache_key = f"{agent_name}_{prompt_type.value}"
            if cache_key in self.prompt_cache:
                return self.prompt_cache[cache_key]
            
            # ファイル読み込み
            if not filepath.exists():
                self.logger.warning(f"プロンプトファイル未発見: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # メタデータ抽出
            variables = self._extract_variables(content)
            version = self._extract_version(content)
            
            # プロンプトテンプレート作成
            template = PromptTemplate(
                name=filename,
                prompt_type=prompt_type,
                template_content=content,
                variables=variables,
                version=version,
                created_at="2024-01-01",  # 実際の実装では file stats から取得
                optimized_for_medal=True
            )
            
            # キャッシュ保存
            self.prompt_cache[cache_key] = template
            
            self.logger.info(f"プロンプト読み込み完了: {filename}")
            return template
            
        except Exception as e:
            self.logger.error(f"プロンプト読み込みエラー: {e}")
            return None
    
    def render_prompt(
        self,
        template: PromptTemplate,
        context_data: Dict[str, Any],
        medal_focus: bool = True
    ) -> str:
        """プロンプトレンダリング（変数置換）"""
        
        try:
            rendered_content = template.template_content
            
            # メダル獲得最適化
            if medal_focus and self.medal_optimization_enabled:
                context_data = self._apply_medal_optimization(context_data)
            
            # 変数置換実行
            for variable, default_value in template.variables.items():
                value = context_data.get(variable, default_value)
                placeholder = "{{" + variable + "}}"
                rendered_content = rendered_content.replace(placeholder, str(value))
            
            # 残存プレースホルダー警告
            if "{{" in rendered_content:
                self.logger.warning("未置換プレースホルダーが残存")
            
            return rendered_content
            
        except Exception as e:
            self.logger.error(f"プロンプトレンダリングエラー: {e}")
            return template.template_content
    
    def get_optimized_prompt(
        self,
        prompt_type: PromptType,
        context_data: Dict[str, Any],
        agent_name: str = None
    ) -> str:
        """最適化プロンプト取得（ワンストップ）"""
        
        # プロンプトテンプレート読み込み
        template = self.load_prompt(prompt_type, agent_name)
        
        if template is None:
            self.logger.error(f"プロンプト取得失敗: {prompt_type.value}")
            return self._get_fallback_prompt(prompt_type)
        
        # レンダリング実行
        return self.render_prompt(template, context_data, medal_focus=True)
    
    def _extract_variables(self, content: str) -> Dict[str, Any]:
        """プロンプト内変数抽出"""
        
        import re
        variables = {}
        
        # {{variable_name}} 形式の変数を抽出
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, content)
        
        for match in matches:
            variables[match] = ""  # デフォルト値は空文字
        
        return variables
    
    def _extract_version(self, content: str) -> str:
        """バージョン情報抽出"""
        
        # メタデータコメントからバージョン抽出
        lines = content.split('\n')
        for line in lines[:10]:  # 先頭10行から探索
            if 'version:' in line.lower():
                return line.split(':')[-1].strip()
        
        return "1.0.0"
    
    def _apply_medal_optimization(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """メダル獲得最適化適用"""
        
        optimized_data = context_data.copy()
        
        # メダル獲得優先度を明示的に追加
        optimized_data['medal_priority'] = 'MAXIMUM'
        optimized_data['optimization_target'] = 'medal_acquisition'
        
        # 競合分析重視
        if 'competitive_analysis' not in optimized_data:
            optimized_data['competitive_analysis'] = 'enabled'
        
        # リスク評価強化
        if 'risk_tolerance' not in optimized_data:
            optimized_data['risk_tolerance'] = 'balanced_for_medal'
        
        return optimized_data
    
    def _get_fallback_prompt(self, prompt_type: PromptType) -> str:
        """フォールバックプロンプト"""
        
        fallback_prompts = {
            PromptType.TECHNICAL_ANALYSIS: """
# 技術分析タスク（フォールバック）

競技の技術要件を分析し、メダル獲得確率を最大化する技術選択を行ってください。

## 分析観点
1. 競技特性に適した技術手法
2. 実装複雑度 vs 効果のバランス
3. 競合優位性の確保
4. リソース制約内での最適化

JSON形式で技術推奨を出力してください。
""",
            PromptType.ANOMALY_DIAGNOSIS: """
# 異常診断タスク（フォールバック）

システム異常を診断し、迅速な対策を提案してください。

## 診断観点
1. 症状の根本原因特定
2. 影響度・緊急度評価
3. 具体的対策手順
4. 予防措置

JSON形式で診断結果を出力してください。
"""
        }
        
        return fallback_prompts.get(prompt_type, "汎用分析タスクを実行してください。")
    
    def create_prompt_directory(self):
        """プロンプトディレクトリ作成"""
        
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"プロンプトディレクトリ作成: {self.prompts_dir}")
    
    def get_prompt_stats(self) -> Dict[str, Any]:
        """プロンプト統計情報"""
        
        return {
            "total_prompts_cached": len(self.prompt_cache),
            "prompts_directory": str(self.prompts_dir),
            "medal_optimization_enabled": self.medal_optimization_enabled,
            "available_prompt_types": [pt.value for pt in PromptType]
        }