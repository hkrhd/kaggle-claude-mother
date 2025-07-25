"""
WebSearch統合調査システム

複数の検索戦略を統合し、効率的な技術情報収集と
構造化分析を実行するシステム。
"""

import asyncio
import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class SearchStrategy(Enum):
    """検索戦略"""
    WINNER_SOLUTIONS = "winner_solutions"      # 優勝解法検索
    LATEST_PAPERS = "latest_papers"           # 最新論文検索
    IMPLEMENTATION_CODES = "implementation_codes"  # 実装コード検索
    TECHNICAL_DISCUSSIONS = "technical_discussions"  # 技術議論検索
    BENCHMARK_RESULTS = "benchmark_results"   # ベンチマーク結果検索


class InformationSource(Enum):
    """情報源"""
    KAGGLE = "kaggle"
    ARXIV = "arxiv"  
    GITHUB = "github"
    PAPERS_WITH_CODE = "papers_with_code"
    REDDIT = "reddit"
    STACK_OVERFLOW = "stack_overflow"
    MEDIUM = "medium"


@dataclass
class SearchQuery:
    """検索クエリ"""
    query_text: str
    strategy: SearchStrategy
    target_sources: List[InformationSource]
    priority: int  # 1-5
    expected_result_count: int


@dataclass
class SearchResult:
    """検索結果"""
    title: str
    url: str
    content: str
    source: InformationSource
    relevance_score: float  # 0-1
    credibility_score: float  # 0-1
    publication_date: Optional[datetime]
    author: Optional[str]
    extracted_techniques: List[str]
    key_insights: List[str]
    implementation_availability: bool
    search_timestamp: datetime


@dataclass
class InvestigationReport:
    """調査レポート"""
    competition_name: str
    investigation_scope: str
    total_results: int
    high_quality_results: int
    key_findings: List[str]
    recommended_techniques: List[Dict[str, Any]]
    implementation_roadmap: List[str]
    risk_assessment: List[str]
    confidence_level: float
    investigation_duration: float  # 秒
    created_at: datetime


class WebSearchIntegrator:
    """WebSearch統合調査エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.search_history: List[SearchResult] = []
        
        # 検索パターン定義
        self.search_patterns = {
            SearchStrategy.WINNER_SOLUTIONS: {
                "kaggle": [
                    "{competition} kaggle winner solution",
                    "{competition} kaggle 1st place approach",
                    "{competition} gold medal solution kaggle",
                    "kaggle {competition} winning method",
                    "{competition} competition winner interview"
                ],
                "github": [
                    "{competition} winner solution github",
                    "{competition} kaggle solution repository",
                    "kaggle {competition} implementation"
                ]
            },
            SearchStrategy.LATEST_PAPERS: {
                "arxiv": [
                    "arxiv {domain} {technique} 2024",
                    "recent advances {domain} machine learning",
                    "{technique} {domain} state of the art",
                    "latest {domain} deep learning arxiv"
                ]
            },
            SearchStrategy.IMPLEMENTATION_CODES: {
                "github": [
                    "{technique} implementation pytorch",
                    "{technique} python code example",
                    "{technique} {framework} implementation",
                    "github {technique} machine learning"
                ],
                "papers_with_code": [
                    "{technique} implementation papers with code",
                    "{technique} benchmark code available"
                ]
            },
            SearchStrategy.TECHNICAL_DISCUSSIONS: {
                "reddit": [
                    "reddit machine learning {technique}",
                    "{technique} discussion r/MachineLearning"
                ],
                "stack_overflow": [
                    "{technique} implementation stack overflow",
                    "{technique} {framework} question"
                ]
            },
            SearchStrategy.BENCHMARK_RESULTS: {
                "papers_with_code": [
                    "{technique} benchmark results",
                    "{dataset} leaderboard {technique}",
                    "{technique} performance comparison"
                ]
            }
        }
        
        # ソース信頼度設定
        self.source_credibility = {
            InformationSource.KAGGLE: 0.9,
            InformationSource.ARXIV: 0.95,
            InformationSource.GITHUB: 0.8,
            InformationSource.PAPERS_WITH_CODE: 0.9,
            InformationSource.REDDIT: 0.6,
            InformationSource.STACK_OVERFLOW: 0.7,
            InformationSource.MEDIUM: 0.6
        }
    
    async def conduct_comprehensive_investigation(
        self,
        competition_name: str,
        competition_domain: str,
        investigation_scope: str = "full",
        time_limit_minutes: int = 120
    ) -> InvestigationReport:
        """包括的技術調査実行"""
        
        start_time = datetime.utcnow()
        self.logger.info(f"包括調査開始: {competition_name} - {investigation_scope}")
        
        try:
            # 調査クエリ生成
            queries = await self._generate_investigation_queries(
                competition_name, competition_domain, investigation_scope
            )
            
            # 並列検索実行
            all_results = await self._execute_parallel_search(queries, time_limit_minutes)
            
            # 結果分析・構造化
            analyzed_results = await self._analyze_search_results(all_results)
            
            # 重要な発見の抽出
            key_findings = await self._extract_key_findings(analyzed_results)
            
            # 技術推奨生成
            recommendations = await self._generate_technique_recommendations(analyzed_results)
            
            # 実装ロードマップ作成
            roadmap = await self._create_implementation_roadmap(recommendations)
            
            # リスク評価
            risks = await self._assess_implementation_risks(analyzed_results, recommendations)
            
            # 信頼度算出
            confidence = self._calculate_investigation_confidence(analyzed_results)
            
            # 調査時間計算
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            report = InvestigationReport(
                competition_name=competition_name,
                investigation_scope=investigation_scope,
                total_results=len(all_results),
                high_quality_results=len([r for r in analyzed_results if r.relevance_score > 0.7]),
                key_findings=key_findings,
                recommended_techniques=recommendations,
                implementation_roadmap=roadmap,
                risk_assessment=risks,
                confidence_level=confidence,
                investigation_duration=duration,
                created_at=datetime.utcnow()
            )
            
            self.logger.info(f"包括調査完了: {duration:.1f}秒, {len(analyzed_results)}件の結果")
            return report
            
        except Exception as e:
            self.logger.error(f"包括調査エラー: {e}")
            return InvestigationReport(
                competition_name=competition_name,
                investigation_scope=investigation_scope,
                total_results=0,
                high_quality_results=0,
                key_findings=["調査エラーが発生しました"],
                recommended_techniques=[],
                implementation_roadmap=["手動調査が必要です"],
                risk_assessment=["自動調査の信頼性に問題があります"],
                confidence_level=0.0,
                investigation_duration=(datetime.utcnow() - start_time).total_seconds(),
                created_at=datetime.utcnow()
            )
    
    async def _generate_investigation_queries(
        self,
        competition_name: str,
        competition_domain: str,
        investigation_scope: str
    ) -> List[SearchQuery]:
        """調査クエリ生成"""
        
        queries = []
        
        # スコープに応じたクエリ生成
        if investigation_scope in ["full", "solutions"]:
            # 優勝解法調査
            for source in [InformationSource.KAGGLE, InformationSource.GITHUB]:
                patterns = self.search_patterns[SearchStrategy.WINNER_SOLUTIONS].get(source.value, [])
                for pattern in patterns:
                    query_text = pattern.format(competition=competition_name)
                    queries.append(SearchQuery(
                        query_text=query_text,
                        strategy=SearchStrategy.WINNER_SOLUTIONS,
                        target_sources=[source],
                        priority=5,  # 最高優先度
                        expected_result_count=5
                    ))
        
        if investigation_scope in ["full", "papers"]:
            # 最新論文調査
            domain_techniques = self._get_domain_techniques(competition_domain)
            for technique in domain_techniques[:3]:  # 上位3技術
                patterns = self.search_patterns[SearchStrategy.LATEST_PAPERS]["arxiv"]
                for pattern in patterns[:2]:  # パターン上位2つ
                    query_text = pattern.format(domain=competition_domain, technique=technique)
                    queries.append(SearchQuery(
                        query_text=query_text,
                        strategy=SearchStrategy.LATEST_PAPERS,
                        target_sources=[InformationSource.ARXIV],
                        priority=4,
                        expected_result_count=3
                    ))
        
        if investigation_scope in ["full", "implementations"]:
            # 実装コード調査
            key_techniques = self._get_key_techniques_for_domain(competition_domain)
            for technique in key_techniques[:2]:
                patterns = self.search_patterns[SearchStrategy.IMPLEMENTATION_CODES]["github"]
                for pattern in patterns[:2]:
                    query_text = pattern.format(technique=technique, framework="pytorch")
                    queries.append(SearchQuery(
                        query_text=query_text,
                        strategy=SearchStrategy.IMPLEMENTATION_CODES,
                        target_sources=[InformationSource.GITHUB],
                        priority=3,
                        expected_result_count=3
                    ))
        
        # 優先度順でソート
        queries.sort(key=lambda q: q.priority, reverse=True)
        
        return queries[:20]  # 最大20クエリ
    
    def _get_domain_techniques(self, domain: str) -> List[str]:
        """ドメイン別技術リスト"""
        domain_tech_map = {
            "tabular": ["ensemble", "gradient boosting", "feature engineering", "stacking", "AutoML"],
            "computer_vision": ["CNN", "vision transformer", "data augmentation", "transfer learning", "object detection"],
            "nlp": ["transformer", "BERT", "attention mechanism", "language model", "text classification"],
            "time_series": ["LSTM", "forecasting", "temporal modeling", "sequence prediction", "time series analysis"]
        }
        return domain_tech_map.get(domain, ["machine learning", "deep learning", "neural networks"])
    
    def _get_key_techniques_for_domain(self, domain: str) -> List[str]:
        """ドメイン別重要技術"""
        return self._get_domain_techniques(domain)[:5]
    
    async def _execute_parallel_search(
        self,
        queries: List[SearchQuery],
        time_limit_minutes: int
    ) -> List[SearchResult]:
        """並列検索実行"""
        
        all_results = []
        semaphore = asyncio.Semaphore(5)  # 同時実行制限
        
        async def search_with_semaphore(query: SearchQuery) -> List[SearchResult]:
            async with semaphore:
                return await self._execute_single_search(query)
        
        # 並列実行
        search_tasks = [search_with_semaphore(query) for query in queries]
        
        try:
            # タイムアウト付き実行
            results_lists = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True),
                timeout=time_limit_minutes * 60
            )
            
            for results in results_lists:
                if isinstance(results, list):
                    all_results.extend(results)
                elif isinstance(results, Exception):
                    self.logger.warning(f"検索タスクエラー: {results}")
            
        except asyncio.TimeoutError:
            self.logger.warning(f"検索タイムアウト: {time_limit_minutes}分")
        
        return all_results
    
    async def _execute_single_search(self, query: SearchQuery) -> List[SearchResult]:
        """単一検索実行"""
        
        try:
            # WebSearchツールを想定した実装
            search_results = await self._perform_web_search(query.query_text)
            
            analyzed_results = []
            for result in search_results[:query.expected_result_count]:
                analyzed_result = await self._analyze_single_result(result, query)
                if analyzed_result:
                    analyzed_results.append(analyzed_result)
            
            # 検索間隔（レート制限対策）
            await asyncio.sleep(1)
            
            return analyzed_results
            
        except Exception as e:
            self.logger.warning(f"検索実行エラー: {query.query_text} - {e}")
            return []
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """WebSearch実行（模擬実装）"""
        # 実際の実装ではWebSearchツールを使用
        # この模擬実装では検索クエリに基づいて適切な結果を生成
        
        mock_results = []
        
        if "winner" in query.lower() or "solution" in query.lower():
            mock_results.append({
                "title": f"Winner Solution Analysis: {query}",
                "url": "https://kaggle.com/discussions/getting-started/123456",
                "content": "This solution achieved 1st place using advanced ensemble methods. Key techniques: XGBoost + LightGBM stacking, extensive feature engineering, and careful cross-validation. The approach combined gradient boosting models with neural networks.",
                "source": "kaggle"
            })
        
        if "arxiv" in query.lower() or "paper" in query.lower():
            mock_results.append({
                "title": f"Recent Advances in {query}",
                "url": "https://arxiv.org/abs/2024.12345",
                "content": "We present a novel approach to ensemble learning that significantly improves performance on tabular data. Our method combines attention mechanisms with traditional gradient boosting.",
                "source": "arxiv"
            })
        
        if "github" in query.lower() or "implementation" in query.lower():
            mock_results.append({
                "title": f"Implementation of {query}",
                "url": "https://github.com/user/repo",
                "content": "PyTorch implementation of the ensemble method. Includes data preprocessing, model training, and evaluation scripts. Achieved 0.95 AUC on validation set.",
                "source": "github"
            })
        
        return mock_results
    
    async def _analyze_single_result(self, result: Dict[str, Any], query: SearchQuery) -> Optional[SearchResult]:
        """単一結果分析"""
        
        try:
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            source_name = result.get("source", "")
            
            # 情報源の特定
            source = self._identify_information_source(url, source_name)
            
            # 関連性スコア算出
            relevance = self._calculate_relevance_score(content, query)
            
            # 信頼度スコア
            credibility = self.source_credibility.get(source, 0.5)
            
            # 技術抽出
            techniques = self._extract_techniques_from_content(content)
            
            # 重要洞察抽出
            insights = self._extract_key_insights(content)
            
            # 実装可能性判定
            implementation_available = self._check_implementation_availability(content, url)
            
            # 公開日推定
            pub_date = self._estimate_publication_date(content, url)
            
            # 著者抽出
            author = self._extract_author_info(content, url)
            
            analyzed_result = SearchResult(
                title=title,
                url=url,
                content=content,
                source=source,
                relevance_score=relevance,
                credibility_score=credibility,
                publication_date=pub_date,
                author=author,
                extracted_techniques=techniques,
                key_insights=insights,
                implementation_availability=implementation_available,
                search_timestamp=datetime.utcnow()
            )
            
            return analyzed_result
            
        except Exception as e:
            self.logger.warning(f"結果分析エラー: {e}")
            return None
    
    def _identify_information_source(self, url: str, source_hint: str) -> InformationSource:
        """情報源特定"""
        url_lower = url.lower()
        source_lower = source_hint.lower()
        
        if "kaggle" in url_lower or "kaggle" in source_lower:
            return InformationSource.KAGGLE
        elif "arxiv" in url_lower or "arxiv" in source_lower:
            return InformationSource.ARXIV
        elif "github" in url_lower or "github" in source_lower:
            return InformationSource.GITHUB
        elif "paperswithcode" in url_lower:
            return InformationSource.PAPERS_WITH_CODE
        elif "reddit" in url_lower:
            return InformationSource.REDDIT
        elif "stackoverflow" in url_lower:
            return InformationSource.STACK_OVERFLOW
        elif "medium" in url_lower:
            return InformationSource.MEDIUM
        else:
            return InformationSource.GITHUB  # デフォルト
    
    def _calculate_relevance_score(self, content: str, query: SearchQuery) -> float:
        """関連性スコア算出"""
        content_lower = content.lower()
        query_terms = query.query_text.lower().split()
        
        # 基本マッチング
        term_matches = sum(1 for term in query_terms if term in content_lower)
        basic_score = term_matches / len(query_terms) if query_terms else 0
        
        # 戦略別ボーナス
        strategy_bonus = 0
        if query.strategy == SearchStrategy.WINNER_SOLUTIONS:
            if any(word in content_lower for word in ["winner", "1st", "gold", "medal"]):
                strategy_bonus = 0.2
        elif query.strategy == SearchStrategy.LATEST_PAPERS:
            if any(word in content_lower for word in ["novel", "state-of-the-art", "recent"]):
                strategy_bonus = 0.2
        elif query.strategy == SearchStrategy.IMPLEMENTATION_CODES:
            if any(word in content_lower for word in ["code", "implementation", "github"]):
                strategy_bonus = 0.2
        
        # 内容の質による調整
        quality_indicators = ["performance", "results", "evaluation", "benchmark", "comparison"]
        quality_score = sum(0.05 for indicator in quality_indicators if indicator in content_lower)
        
        final_score = min(basic_score + strategy_bonus + quality_score, 1.0)
        return final_score
    
    def _extract_techniques_from_content(self, content: str) -> List[str]:
        """コンテンツから技術抽出"""
        technique_patterns = [
            r"(ensemble|stacking|blending|voting|bagging)",
            r"(xgboost|lightgbm|catboost|gradient.boosting)",
            r"(neural.network|deep.learning|cnn|rnn|lstm|transformer)",
            r"(feature.engineering|feature.selection|preprocessing)",
            r"(cross.validation|hyperparameter|optimization)",
            r"(attention|self.attention|bert|gpt)",
            r"(data.augmentation|transfer.learning|fine.tuning)"
        ]
        
        techniques = []
        content_lower = content.lower()
        
        for pattern in technique_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            techniques.extend(matches)
        
        return list(set(techniques))[:8]  # 最大8技術、重複除去
    
    def _extract_key_insights(self, content: str) -> List[str]:
        """重要洞察抽出"""
        insight_patterns = [
            r"key.insight[s]?[:\-]\s*(.+?)(?:\.|$)",
            r"important[:\-]\s*(.+?)(?:\.|$)",
            r"crucial[:\-]\s*(.+?)(?:\.|$)",
            r"main.idea[:\-]\s*(.+?)(?:\.|$)",
            r"breakthrough[:\-]\s*(.+?)(?:\.|$)"
        ]
        
        insights = []
        for pattern in insight_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            insights.extend([match.strip()[:100] for match in matches])  # 100文字制限
        
        return insights[:5]  # 最大5件
    
    def _check_implementation_availability(self, content: str, url: str) -> bool:
        """実装可能性チェック"""
        implementation_indicators = [
            "code", "implementation", "github", "repository", "notebook",
            "available", "download", "script", "jupyter", "colab"
        ]
        
        text = (content + " " + url).lower()
        return any(indicator in text for indicator in implementation_indicators)
    
    def _estimate_publication_date(self, content: str, url: str) -> Optional[datetime]:
        """公開日推定"""
        # URL から日付パターンを抽出
        date_patterns = [
            r"(\d{4})/(\d{1,2})/(\d{1,2})",
            r"(\d{4})-(\d{1,2})-(\d{1,2})",
            r"(\d{1,2})/(\d{1,2})/(\d{4})"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, url)
            if match:
                try:
                    if len(match.group(1)) == 4:  # YYYY/MM/DD or YYYY-MM-DD
                        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    else:  # MM/DD/YYYY
                        return datetime(int(match.group(3)), int(match.group(1)), int(match.group(2)))
                except ValueError:
                    continue
        
        # コンテンツから日付キーワード検索
        recent_keywords = ["2024", "recent", "latest", "new", "updated"]
        if any(keyword in content.lower() for keyword in recent_keywords):
            return datetime.utcnow() - timedelta(days=30)  # 1ヶ月前と推定
        
        return None
    
    def _extract_author_info(self, content: str, url: str) -> Optional[str]:
        """著者情報抽出"""
        # URL からユーザー名抽出
        if "github.com" in url:
            match = re.search(r"github\.com/([^/]+)", url)
            if match:
                return match.group(1)
        
        if "kaggle.com" in url:
            match = re.search(r"kaggle\.com/([^/]+)", url)
            if match:
                return match.group(1)
        
        # コンテンツから著者パターン検索
        author_patterns = [
            r"author[s]?[:\s]+([A-Za-z\s]+)",
            r"by[:\s]+([A-Za-z\s]+)",
            r"@([A-Za-z0-9_]+)"
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    async def _analyze_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """検索結果分析"""
        # 重複除去
        unique_results = self._deduplicate_results(results)
        
        # 品質フィルタリング
        quality_results = [r for r in unique_results if r.relevance_score > 0.3]
        
        # スコア順ソート
        quality_results.sort(key=lambda r: r.relevance_score * r.credibility_score, reverse=True)
        
        return quality_results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """重複結果除去"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    async def _extract_key_findings(self, results: List[SearchResult]) -> List[str]:
        """重要発見抽出"""
        findings = []
        
        # 高品質結果からの発見
        high_quality_results = [r for r in results if r.relevance_score > 0.7]
        
        if high_quality_results:
            # 最頻出技術
            all_techniques = []
            for result in high_quality_results:
                all_techniques.extend(result.extracted_techniques)
            
            if all_techniques:
                technique_counts = {}
                for tech in all_techniques:
                    technique_counts[tech] = technique_counts.get(tech, 0) + 1
                
                most_common = max(technique_counts.items(), key=lambda x: x[1])
                findings.append(f"最頻出技術: {most_common[0]} ({most_common[1]}回言及)")
            
            # 実装可能性
            implementation_count = sum(1 for r in high_quality_results if r.implementation_availability)
            impl_ratio = implementation_count / len(high_quality_results)
            findings.append(f"実装コード利用可能率: {impl_ratio:.1%}")
            
            # 情報源分析
            source_counts = {}
            for result in high_quality_results:
                source = result.source.value
                source_counts[source] = source_counts.get(source, 0) + 1
            
            if source_counts:
                primary_source = max(source_counts.items(), key=lambda x: x[1])
                findings.append(f"主要情報源: {primary_source[0]} ({primary_source[1]}件)")
        
        else:
            findings.append("高品質な結果が不足しています")
        
        return findings[:5]  # 最大5件
    
    async def _generate_technique_recommendations(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """技術推奨生成"""
        recommendations = []
        
        # 技術別集計
        technique_analysis = {}
        
        for result in results:
            for technique in result.extracted_techniques:
                if technique not in technique_analysis:
                    technique_analysis[technique] = {
                        "mention_count": 0,
                        "avg_relevance": 0.0,
                        "avg_credibility": 0.0,
                        "implementation_available": False,
                        "sources": []
                    }
                
                analysis = technique_analysis[technique]
                analysis["mention_count"] += 1
                analysis["avg_relevance"] += result.relevance_score
                analysis["avg_credibility"] += result.credibility_score
                if result.implementation_availability:
                    analysis["implementation_available"] = True
                analysis["sources"].append(result.source.value)
        
        # 推奨スコア計算
        for technique, analysis in technique_analysis.items():
            if analysis["mention_count"] >= 2:  # 最低2回言及
                avg_relevance = analysis["avg_relevance"] / analysis["mention_count"]
                avg_credibility = analysis["avg_credibility"] / analysis["mention_count"]
                
                # 推奨スコア = 関連性 * 0.4 + 信頼度 * 0.3 + 言及頻度 * 0.2 + 実装可能性 * 0.1
                recommendation_score = (
                    avg_relevance * 0.4 +
                    avg_credibility * 0.3 +
                    min(analysis["mention_count"] / 5.0, 1.0) * 0.2 +
                    (0.1 if analysis["implementation_available"] else 0.0)
                )
                
                recommendations.append({
                    "technique": technique,
                    "recommendation_score": recommendation_score,
                    "mention_count": analysis["mention_count"],
                    "avg_relevance": avg_relevance,
                    "implementation_available": analysis["implementation_available"],
                    "primary_sources": list(set(analysis["sources"]))
                })
        
        # スコア順ソート
        recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
        
        return recommendations[:8]  # 上位8技術
    
    async def _create_implementation_roadmap(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """実装ロードマップ作成"""
        roadmap = []
        
        if not recommendations:
            return ["技術推奨が不足しているため手動計画が必要です"]
        
        # 優先度別実装計画
        high_priority = [r for r in recommendations if r["recommendation_score"] > 0.7]
        medium_priority = [r for r in recommendations if 0.5 <= r["recommendation_score"] <= 0.7]
        
        if high_priority:
            roadmap.append("フェーズ1: 高優先度技術実装")
            for i, rec in enumerate(high_priority[:3], 1):
                impl_note = "（実装コード利用可能）" if rec["implementation_available"] else ""
                roadmap.append(f"  {i}. {rec['technique']} {impl_note}")
        
        if medium_priority:
            roadmap.append("フェーズ2: 中優先度技術実装")
            for i, rec in enumerate(medium_priority[:2], 1):
                impl_note = "（実装コード利用可能）" if rec["implementation_available"] else ""
                roadmap.append(f"  {i}. {rec['technique']} {impl_note}")
        
        # 統合・最適化フェーズ
        roadmap.append("フェーズ3: 技術統合・最適化")
        roadmap.append("  1. アンサンブル構築")
        roadmap.append("  2. ハイパーパラメータ調整")
        roadmap.append("  3. 最終モデル選択")
        
        return roadmap
    
    async def _assess_implementation_risks(
        self,
        results: List[SearchResult],
        recommendations: List[Dict[str, Any]]
    ) -> List[str]:
        """実装リスク評価"""
        risks = []
        
        # 情報不足リスク
        if len(results) < 5:
            risks.append("情報収集不足による実装失敗リスク")
        
        # 実装コード不足リスク
        implementation_available_count = sum(1 for r in recommendations if r.get("implementation_available", False))
        if implementation_available_count < len(recommendations) * 0.5:
            risks.append("実装コード不足による開発遅延リスク")
        
        # 技術複雑度リスク
        complex_techniques = ["transformer", "neural network", "deep learning", "ensemble"]
        complex_count = sum(1 for r in recommendations if any(ct in r["technique"] for ct in complex_techniques))
        if complex_count > len(recommendations) * 0.6:
            risks.append("高複雑度技術による実装困難リスク")
        
        # 信頼度リスク
        avg_credibility = sum(r.credibility_score for r in results) / len(results) if results else 0
        if avg_credibility < 0.7:
            risks.append("情報源信頼度不足による判断ミスリスク")
        
        return risks[:5]  # 最大5リスク
    
    def _calculate_investigation_confidence(self, results: List[SearchResult]) -> float:
        """調査信頼度算出"""
        if not results:
            return 0.0
        
        # 結果数による信頼度
        count_confidence = min(len(results) / 10.0, 1.0)
        
        # 品質による信頼度
        high_quality_count = sum(1 for r in results if r.relevance_score > 0.7)
        quality_confidence = high_quality_count / len(results)
        
        # 多様性による信頼度
        unique_sources = len(set(r.source for r in results))
        diversity_confidence = min(unique_sources / 4.0, 1.0)  # 最大4ソース
        
        # 総合信頼度
        overall_confidence = (
            count_confidence * 0.3 +
            quality_confidence * 0.4 +
            diversity_confidence * 0.3
        )
        
        return min(overall_confidence, 0.95)  # 最大95%
    
    async def generate_search_summary(self) -> Dict[str, Any]:
        """検索サマリー生成"""
        
        if not self.search_history:
            return {"message": "検索履歴がありません"}
        
        total_searches = len(self.search_history)
        
        # ソース分布
        source_distribution = {}
        for result in self.search_history:
            source = result.source.value
            source_distribution[source] = source_distribution.get(source, 0) + 1
        
        # 品質統計
        high_quality_count = sum(1 for r in self.search_history if r.relevance_score > 0.7)
        avg_relevance = sum(r.relevance_score for r in self.search_history) / total_searches
        avg_credibility = sum(r.credibility_score for r in self.search_history) / total_searches
        
        # 技術頻度
        all_techniques = []
        for result in self.search_history:
            all_techniques.extend(result.extracted_techniques)
        
        technique_counts = {}
        for tech in all_techniques:
            technique_counts[tech] = technique_counts.get(tech, 0) + 1
        
        top_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        summary = {
            "search_statistics": {
                "total_searches": total_searches,
                "high_quality_results": high_quality_count,
                "quality_ratio": f"{(high_quality_count/total_searches):.1%}",
                "avg_relevance_score": f"{avg_relevance:.2f}",
                "avg_credibility_score": f"{avg_credibility:.2f}"
            },
            "source_distribution": source_distribution,
            "top_techniques": top_techniques,
            "recent_searches": [
                {
                    "title": result.title[:50] + "..." if len(result.title) > 50 else result.title,
                    "source": result.source.value,
                    "relevance": result.relevance_score,
                    "timestamp": result.search_timestamp.strftime("%Y-%m-%d %H:%M")
                }
                for result in sorted(self.search_history, key=lambda r: r.search_timestamp, reverse=True)[:5]
            ]
        }
        
        return summary