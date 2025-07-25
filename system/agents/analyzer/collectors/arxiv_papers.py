"""
arXiv最新論文収集・分析システム

WebSearchを活用してarXivから最新の機械学習論文を収集し、
Kaggleコンペに応用可能な技術を特定・評価するシステム。
"""

import asyncio
import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class PaperCategory(Enum):
    """論文カテゴリ"""
    MACHINE_LEARNING = "cs.LG"
    COMPUTER_VISION = "cs.CV"
    NATURAL_LANGUAGE = "cs.CL"
    ARTIFICIAL_INTELLIGENCE = "cs.AI"
    STATISTICS = "stat.ML"
    DATA_STRUCTURES = "cs.DS"


class ApplicabilityLevel(Enum):
    """応用可能性レベル"""
    HIGH = "high"           # 直接応用可能
    MEDIUM = "medium"       # 適応・改良で応用可能
    LOW = "low"            # 研究レベル、実装困難
    THEORETICAL = "theoretical"  # 理論的、実装不適切


# TechnicalComplexityは technical_feasibility からインポート
from ..analyzers.technical_feasibility import TechnicalComplexity as ImplementationComplexity


@dataclass
class ArxivPaper:
    """arXiv論文情報"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    category: PaperCategory
    published_date: datetime
    updated_date: Optional[datetime]
    pdf_url: str
    abstract_url: str
    
    # 分析結果
    kaggle_applicability: ApplicabilityLevel
    implementation_complexity: ImplementationComplexity
    estimated_implementation_time: str
    key_techniques: List[str]
    potential_improvements: List[str]
    competition_types: List[str]  # 適用可能なコンペタイプ
    required_libraries: List[str]
    gpu_requirement: bool
    novelty_score: float  # 0-1, 新規性スコア
    practical_value: float  # 0-1, 実用価値スコア
    
    # メタデータ
    citation_count: Optional[int]
    github_implementations: List[str]
    related_papers: List[str]
    collected_at: datetime


@dataclass
class TechniqueExtraction:
    """技術抽出結果"""
    technique_name: str
    description: str
    mathematical_foundation: str
    implementation_hints: List[str]
    expected_performance_gain: str
    limitations: List[str]
    prerequisites: List[str]


class ArxivPaperCollector:
    """arXiv論文収集・分析エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.collected_papers: List[ArxivPaper] = []
        
        # 検索キーワード定義
        self.search_keywords = {
            "tabular": [
                "tabular data", "structured data", "feature engineering", 
                "gradient boosting", "ensemble methods", "AutoML"
            ],
            "computer_vision": [
                "computer vision", "image classification", "object detection",
                "convolutional neural networks", "vision transformer", "data augmentation"
            ],
            "nlp": [
                "natural language processing", "transformer", "BERT", "GPT",
                "text classification", "language models", "attention mechanism"
            ],
            "time_series": [
                "time series", "forecasting", "temporal data", "sequence modeling",
                "LSTM", "GRU", "time series analysis"
            ],
            "general_ml": [
                "machine learning", "deep learning", "neural networks",
                "optimization", "regularization", "transfer learning"
            ]
        }
        
        # 実装可能性判定キーワード
        self.implementation_indicators = {
            "high_applicability": [
                "code available", "implementation", "reproducible",
                "practical", "real-world", "benchmark", "sota", "state-of-the-art"
            ],
            "medium_applicability": [
                "method", "algorithm", "approach", "technique",
                "framework", "model", "architecture"
            ],
            "low_applicability": [
                "theoretical", "proof", "analysis", "study",
                "investigation", "survey", "review"
            ]
        }
        
        # 複雑度判定キーワード
        self.complexity_indicators = {
            "simple": [
                "simple", "basic", "straightforward", "easy",
                "standard", "conventional", "traditional"
            ],
            "moderate": [
                "novel", "new", "improved", "enhanced",
                "modified", "adapted", "extended"
            ],
            "complex": [
                "advanced", "sophisticated", "complex", "intricate",
                "multi-stage", "hierarchical", "end-to-end"
            ],
            "research": [
                "cutting-edge", "pioneering", "groundbreaking",
                "revolutionary", "unprecedented", "innovative"
            ]
        }
    
    async def collect_latest_papers(
        self,
        competition_domain: str,
        days_back: int = 30,
        max_papers: int = 20
    ) -> List[ArxivPaper]:
        """最新論文収集"""
        
        self.logger.info(f"arXiv論文収集開始: {competition_domain}, {days_back}日間")
        
        try:
            # ドメイン別キーワード取得
            domain_keywords = self.search_keywords.get(competition_domain, self.search_keywords["general_ml"])
            
            # 検索実行
            papers = await self._search_arxiv_papers(domain_keywords, days_back, max_papers)
            
            # 論文分析・フィルタリング
            analyzed_papers = []
            for paper_data in papers:
                paper = await self._analyze_paper(paper_data, competition_domain)
                if paper and paper.kaggle_applicability != ApplicabilityLevel.THEORETICAL:
                    analyzed_papers.append(paper)
            
            # 収集結果保存
            self.collected_papers.extend(analyzed_papers)
            
            self.logger.info(f"論文収集完了: {len(analyzed_papers)}件")
            return analyzed_papers
            
        except Exception as e:
            self.logger.error(f"論文収集エラー: {e}")
            return []
    
    async def _search_arxiv_papers(
        self,
        keywords: List[str],
        days_back: int,
        max_papers: int
    ) -> List[Dict[str, Any]]:
        """arXiv論文検索"""
        
        papers = []
        
        # 複数キーワードでの検索
        for keyword in keywords[:5]:  # 上位5キーワードに限定
            try:
                # WebSearchを使用した検索
                query = f"arXiv {keyword} machine learning 2024"
                search_results = await self._perform_arxiv_search(query)
                
                # 結果の構造化
                for result in search_results:
                    paper_data = await self._extract_paper_metadata(result)
                    if paper_data:
                        papers.append(paper_data)
                
                # レート制限対策
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"検索エラー: {keyword} - {e}")
                continue
        
        # 重複除去・日付フィルタリング
        unique_papers = self._deduplicate_papers(papers)
        recent_papers = self._filter_by_date(unique_papers, days_back)
        
        return recent_papers[:max_papers]
    
    async def _perform_arxiv_search(self, query: str) -> List[Dict[str, Any]]:
        """arXiv検索実行（模擬実装）"""
        # 実際の実装ではWebSearchツールを使用
        mock_results = [
            {
                "title": f"Advanced Ensemble Methods for {query}",
                "url": "https://arxiv.org/abs/2024.12345",
                "content": "We propose a novel ensemble method that combines gradient boosting with neural networks. Our approach achieves state-of-the-art results on benchmark datasets. Code is available at github.com/author/repo.",
                "snippet": "Novel ensemble method combining gradient boosting and neural networks",
                "date": "2024-01-15"
            },
            {
                "title": f"Transformer-based Feature Engineering for Tabular Data",
                "url": "https://arxiv.org/abs/2024.67890", 
                "content": "This paper introduces a transformer architecture specifically designed for tabular data. We show significant improvements on various classification tasks.",
                "snippet": "Transformer architecture for tabular data classification",
                "date": "2024-01-10"
            }
        ]
        
        return mock_results
    
    async def _extract_paper_metadata(self, search_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """論文メタデータ抽出"""
        
        try:
            url = search_result.get("url", "")
            title = search_result.get("title", "")
            content = search_result.get("content", "")
            
            # arXiv IDの抽出
            arxiv_id_match = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url)
            if not arxiv_id_match:
                return None
            
            arxiv_id = arxiv_id_match.group(1)
            
            # 著者名の抽出
            authors = self._extract_authors(content)
            
            # 日付の抽出
            pub_date = self._extract_publication_date(search_result.get("date", ""))
            
            # カテゴリの推定
            category = self._infer_category(title, content)
            
            return {
                "title": title,
                "authors": authors,
                "abstract": content,
                "arxiv_id": arxiv_id,
                "category": category,
                "published_date": pub_date,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "abstract_url": f"https://arxiv.org/abs/{arxiv_id}"
            }
            
        except Exception as e:
            self.logger.warning(f"メタデータ抽出エラー: {e}")
            return None
    
    def _extract_authors(self, content: str) -> List[str]:
        """著者名抽出"""
        # 簡易的な著者名抽出
        author_patterns = [
            r"Authors?[:\s]+([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)",
            r"By[:\s]+([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)"
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content)
            if match:
                authors = [name.strip() for name in match.group(1).split(",")]
                return authors[:5]  # 最大5名
        
        return ["Unknown"]
    
    def _extract_publication_date(self, date_str: str) -> datetime:
        """発行日抽出"""
        try:
            if date_str:
                # 様々な日付形式に対応
                date_formats = ["%Y-%m-%d", "%Y/%m/%d", "%d %b %Y"]
                for fmt in date_formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
        except:
            pass
        
        return datetime.utcnow() - timedelta(days=7)  # デフォルト：1週間前
    
    def _infer_category(self, title: str, content: str) -> PaperCategory:
        """カテゴリ推定"""
        text = (title + " " + content).lower()
        
        if any(keyword in text for keyword in ["computer vision", "image", "visual", "cnn"]):
            return PaperCategory.COMPUTER_VISION
        elif any(keyword in text for keyword in ["nlp", "language", "text", "transformer", "bert"]):
            return PaperCategory.NATURAL_LANGUAGE
        elif any(keyword in text for keyword in ["statistics", "statistical", "bayesian"]):
            return PaperCategory.STATISTICS
        elif any(keyword in text for keyword in ["machine learning", "ml", "neural network"]):
            return PaperCategory.MACHINE_LEARNING
        else:
            return PaperCategory.ARTIFICIAL_INTELLIGENCE
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重複論文除去"""
        seen_ids = set()
        unique_papers = []
        
        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if arxiv_id and arxiv_id not in seen_ids:
                seen_ids.add(arxiv_id)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _filter_by_date(self, papers: List[Dict[str, Any]], days_back: int) -> List[Dict[str, Any]]:
        """日付フィルタリング"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        filtered_papers = []
        for paper in papers:
            pub_date = paper.get("published_date", datetime.utcnow())
            if pub_date >= cutoff_date:
                filtered_papers.append(paper)
        
        return filtered_papers
    
    async def _analyze_paper(self, paper_data: Dict[str, Any], competition_domain: str) -> Optional[ArxivPaper]:
        """論文詳細分析"""
        
        try:
            # 基本情報
            title = paper_data["title"]
            abstract = paper_data["abstract"]
            
            # 応用可能性評価
            applicability = self._assess_kaggle_applicability(title, abstract)
            
            # 実装複雑度評価
            complexity = self._assess_implementation_complexity(title, abstract)
            
            # 実装時間推定
            impl_time = self._estimate_implementation_time(complexity, abstract)
            
            # 技術抽出
            techniques = self._extract_key_techniques(abstract)
            
            # 改善ポイント特定
            improvements = self._identify_potential_improvements(abstract)
            
            # 適用コンペタイプ
            comp_types = self._determine_applicable_competition_types(title, abstract, competition_domain)
            
            # 必要ライブラリ
            libraries = self._identify_required_libraries(abstract, techniques)
            
            # GPU要件
            gpu_required = self._assess_gpu_requirement(abstract, techniques)
            
            # 新規性・実用価値スコア
            novelty = self._calculate_novelty_score(title, abstract)
            practical_value = self._calculate_practical_value(abstract, techniques)
            
            # GitHub実装検索
            github_repos = await self._search_github_implementations(paper_data["arxiv_id"])
            
            paper = ArxivPaper(
                title=title,
                authors=paper_data["authors"],
                abstract=abstract,
                arxiv_id=paper_data["arxiv_id"],
                category=paper_data["category"],
                published_date=paper_data["published_date"],
                updated_date=None,
                pdf_url=paper_data["pdf_url"],
                abstract_url=paper_data["abstract_url"],
                kaggle_applicability=applicability,
                implementation_complexity=complexity,
                estimated_implementation_time=impl_time,
                key_techniques=techniques,
                potential_improvements=improvements,
                competition_types=comp_types,
                required_libraries=libraries,
                gpu_requirement=gpu_required,
                novelty_score=novelty,
                practical_value=practical_value,
                citation_count=None,  # 実装では引用数API使用
                github_implementations=github_repos,
                related_papers=[],
                collected_at=datetime.utcnow()
            )
            
            return paper
            
        except Exception as e:
            self.logger.error(f"論文分析エラー: {e}")
            return None
    
    def _assess_kaggle_applicability(self, title: str, abstract: str) -> ApplicabilityLevel:
        """Kaggle応用可能性評価"""
        text = (title + " " + abstract).lower()
        
        high_indicators = self.implementation_indicators["high_applicability"]
        medium_indicators = self.implementation_indicators["medium_applicability"]
        low_indicators = self.implementation_indicators["low_applicability"]
        
        high_score = sum(1 for indicator in high_indicators if indicator in text)
        medium_score = sum(1 for indicator in medium_indicators if indicator in text)
        low_score = sum(1 for indicator in low_indicators if indicator in text)
        
        if high_score >= 2:
            return ApplicabilityLevel.HIGH
        elif high_score >= 1 or medium_score >= 3:
            return ApplicabilityLevel.MEDIUM
        elif low_score >= 2:
            return ApplicabilityLevel.THEORETICAL
        else:
            return ApplicabilityLevel.LOW
    
    def _assess_implementation_complexity(self, title: str, abstract: str) -> ImplementationComplexity:
        """実装複雑度評価"""
        text = (title + " " + abstract).lower()
        
        complexity_scores = {}
        for level, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            complexity_scores[level] = score
        
        max_score_level = max(complexity_scores.items(), key=lambda x: x[1])[0]
        
        complexity_mapping = {
            "simple": ImplementationComplexity.SIMPLE,
            "moderate": ImplementationComplexity.MODERATE,
            "complex": ImplementationComplexity.COMPLEX,
            "research": ImplementationComplexity.RESEARCH
        }
        
        return complexity_mapping.get(max_score_level, ImplementationComplexity.MODERATE)
    
    def _estimate_implementation_time(self, complexity: ImplementationComplexity, abstract: str) -> str:
        """実装時間推定"""
        base_times = {
            ImplementationComplexity.SIMPLE: 3,     # 3日
            ImplementationComplexity.MODERATE: 7,   # 1週間
            ImplementationComplexity.COMPLEX: 14,   # 2週間
            ImplementationComplexity.RESEARCH: 30   # 1ヶ月
        }
        
        base_days = base_times[complexity]
        
        # 抽象の長さによる調整（複雑な手法ほど長い抽象）
        length_factor = min(len(abstract) / 1000, 1.5)
        
        estimated_days = int(base_days * length_factor)
        
        if estimated_days <= 7:
            return f"{estimated_days}日"
        elif estimated_days <= 30:
            return f"{estimated_days // 7}週間"
        else:
            return "1ヶ月以上"
    
    def _extract_key_techniques(self, abstract: str) -> List[str]:
        """重要技術抽出"""
        # 技術キーワードパターン
        technique_patterns = [
            r"(neural network|deep learning|CNN|RNN|LSTM|GRU|transformer)",
            r"(ensemble|boosting|bagging|stacking|voting)",
            r"(attention|self-attention|multi-head)",
            r"(batch normalization|dropout|regularization)",
            r"(data augmentation|transfer learning|fine-tuning)",
            r"(optimization|Adam|SGD|learning rate)",
            r"(feature engineering|feature selection|dimensionality reduction)"
        ]
        
        techniques = []
        abstract_lower = abstract.lower()
        
        for pattern in technique_patterns:
            matches = re.findall(pattern, abstract_lower)
            techniques.extend(matches)
        
        return list(set(techniques))[:10]  # 最大10件、重複除去
    
    def _identify_potential_improvements(self, abstract: str) -> List[str]:
        """改善ポイント特定"""
        improvement_patterns = [
            r"improve[s]? ([^.]+)",
            r"enhance[s]? ([^.]+)",
            r"better ([^.]+)",
            r"outperform[s]? ([^.]+)",
            r"achieve[s]? ([^.]+)"
        ]
        
        improvements = []
        for pattern in improvement_patterns:
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            improvements.extend([match.strip() for match in matches])
        
        return improvements[:5]  # 上位5件
    
    def _determine_applicable_competition_types(self, title: str, abstract: str, domain: str) -> List[str]:
        """適用可能コンペタイプ判定"""
        text = (title + " " + abstract).lower()
        applicable_types = []
        
        # ドメイン固有判定
        if domain == "tabular" or any(keyword in text for keyword in ["tabular", "structured", "classification", "regression"]):
            applicable_types.append("tabular")
        
        if domain == "computer_vision" or any(keyword in text for keyword in ["image", "vision", "cnn", "visual"]):
            applicable_types.append("computer_vision")
        
        if domain == "nlp" or any(keyword in text for keyword in ["text", "language", "nlp", "transformer"]):
            applicable_types.append("nlp")
        
        if any(keyword in text for keyword in ["time series", "temporal", "sequence", "forecasting"]):
            applicable_types.append("time_series")
        
        # 一般的な機械学習手法は全タイプに適用可能
        if any(keyword in text for keyword in ["ensemble", "optimization", "regularization"]):
            applicable_types.extend(["tabular", "computer_vision", "nlp"])
        
        return list(set(applicable_types)) or [domain]
    
    def _identify_required_libraries(self, abstract: str, techniques: List[str]) -> List[str]:
        """必要ライブラリ特定"""
        libraries = set()
        
        # 抽象からライブラリ名を直接抽出
        library_patterns = [
            r"(pytorch|tensorflow|keras|scikit-learn|xgboost|lightgbm|catboost)",
            r"(numpy|pandas|scipy|matplotlib|seaborn)",
            r"(transformers|huggingface|timm|torchvision)",
            r"(optuna|hyperopt|sklearn)"
        ]
        
        abstract_lower = abstract.lower()
        for pattern in library_patterns:
            matches = re.findall(pattern, abstract_lower)
            libraries.update(matches)
        
        # 技術に基づくライブラリ推定
        technique_to_library = {
            "neural network": ["pytorch", "tensorflow"],
            "transformer": ["transformers", "pytorch"],
            "boosting": ["xgboost", "lightgbm"],
            "ensemble": ["scikit-learn"],
            "optimization": ["optuna", "scipy"]
        }
        
        for technique in techniques:
            for tech_keyword, libs in technique_to_library.items():
                if tech_keyword in technique:
                    libraries.update(libs)
        
        return list(libraries)[:8]  # 最大8ライブラリ
    
    def _assess_gpu_requirement(self, abstract: str, techniques: List[str]) -> bool:
        """GPU要件評価"""
        gpu_indicators = [
            "neural network", "deep learning", "cnn", "rnn", "lstm",
            "transformer", "attention", "gpu", "cuda", "large-scale"
        ]
        
        text = (abstract + " " + " ".join(techniques)).lower()
        
        return any(indicator in text for indicator in gpu_indicators)
    
    def _calculate_novelty_score(self, title: str, abstract: str) -> float:
        """新規性スコア算出"""
        novelty_keywords = [
            "novel", "new", "first", "pioneering", "innovative",
            "breakthrough", "unprecedented", "original"
        ]
        
        text = (title + " " + abstract).lower()
        novelty_count = sum(1 for keyword in novelty_keywords if keyword in text)
        
        # 最大5個のキーワードで正規化
        return min(novelty_count / 5.0, 1.0)
    
    def _calculate_practical_value(self, abstract: str, techniques: List[str]) -> float:
        """実用価値スコア算出"""
        practical_keywords = [
            "practical", "real-world", "benchmark", "sota", "state-of-the-art",
            "performance", "accuracy", "improvement", "efficient", "effective"
        ]
        
        text = (abstract + " " + " ".join(techniques)).lower()
        practical_count = sum(1 for keyword in practical_keywords if keyword in text)
        
        # 技術数によるボーナス
        technique_bonus = min(len(techniques) / 10.0, 0.3)
        
        base_score = min(practical_count / 8.0, 0.7)
        
        return min(base_score + technique_bonus, 1.0)
    
    async def _search_github_implementations(self, arxiv_id: str) -> List[str]:
        """GitHub実装検索"""
        try:
            # WebSearchを使用してGitHub実装を検索
            query = f"github {arxiv_id} implementation"
            # 実際の実装ではWebSearchツールを使用
            # search_results = await web_search(query)
            
            # 模擬データ
            mock_repos = [
                f"https://github.com/author1/paper-{arxiv_id}",
                f"https://github.com/author2/{arxiv_id}-implementation"
            ]
            
            return mock_repos[:3]  # 最大3リポジトリ
            
        except Exception as e:
            self.logger.warning(f"GitHub実装検索エラー: {e}")
            return []
    
    async def get_papers_by_applicability(
        self,
        applicability: ApplicabilityLevel,
        max_results: int = 10
    ) -> List[ArxivPaper]:
        """応用可能性別論文取得"""
        
        filtered_papers = [
            paper for paper in self.collected_papers
            if paper.kaggle_applicability == applicability
        ]
        
        # 実用価値スコア順でソート
        filtered_papers.sort(key=lambda p: p.practical_value, reverse=True)
        
        return filtered_papers[:max_results]
    
    async def generate_technique_recommendations(
        self,
        competition_domain: str,
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """技術推奨生成"""
        
        # 対象ドメインの高適用性論文を抽出
        relevant_papers = [
            paper for paper in self.collected_papers
            if (competition_domain in paper.competition_types and
                paper.kaggle_applicability in [ApplicabilityLevel.HIGH, ApplicabilityLevel.MEDIUM])
        ]
        
        if not relevant_papers:
            return []
        
        # 技術別に分析
        technique_analysis = {}
        
        for paper in relevant_papers:
            for technique in paper.key_techniques:
                if technique not in technique_analysis:
                    technique_analysis[technique] = {
                        "papers": [],
                        "avg_practical_value": 0.0,
                        "avg_novelty": 0.0,
                        "complexity_distribution": {},
                        "implementation_count": 0
                    }
                
                analysis = technique_analysis[technique]
                analysis["papers"].append(paper)
                analysis["implementation_count"] += len(paper.github_implementations)
        
        # 推奨スコア計算
        recommendations = []
        
        for technique, analysis in technique_analysis.items():
            papers = analysis["papers"]
            
            avg_practical = sum(p.practical_value for p in papers) / len(papers)
            avg_novelty = sum(p.novelty_score for p in papers) / len(papers)
            paper_count = len(papers)
            
            # 推奨スコア = 実用価値 * 0.4 + 新規性 * 0.3 + 実装数 * 0.3
            recommendation_score = (
                avg_practical * 0.4 +
                avg_novelty * 0.3 +
                min(analysis["implementation_count"] / 5.0, 1.0) * 0.3
            )
            
            recommendations.append({
                "technique": technique,
                "recommendation_score": recommendation_score,
                "paper_count": paper_count,
                "avg_practical_value": avg_practical,
                "avg_novelty": avg_novelty,
                "implementation_availability": analysis["implementation_count"] > 0,
                "recent_papers": [p.title for p in papers[:3]],
                "estimated_implementation_time": papers[0].estimated_implementation_time if papers else "不明"
            })
        
        # スコア順でソート
        recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
        
        return recommendations[:max_recommendations]
    
    async def create_paper_summary_report(self) -> Dict[str, Any]:
        """論文サマリーレポート生成"""
        
        if not self.collected_papers:
            return {"error": "収集論文がありません"}
        
        total_papers = len(self.collected_papers)
        
        # 応用可能性分布
        applicability_dist = {}
        for paper in self.collected_papers:
            level = paper.kaggle_applicability.value
            applicability_dist[level] = applicability_dist.get(level, 0) + 1
        
        # 複雑度分布
        complexity_dist = {}
        for paper in self.collected_papers:
            level = paper.implementation_complexity.value
            complexity_dist[level] = complexity_dist.get(level, 0) + 1
        
        # カテゴリ分布
        category_dist = {}
        for paper in self.collected_papers:
            cat = paper.category.value
            category_dist[cat] = category_dist.get(cat, 0) + 1
        
        # GPU要件統計
        gpu_required_count = sum(1 for p in self.collected_papers if p.gpu_requirement)
        gpu_ratio = gpu_required_count / total_papers
        
        # 平均スコア
        avg_novelty = sum(p.novelty_score for p in self.collected_papers) / total_papers
        avg_practical = sum(p.practical_value for p in self.collected_papers) / total_papers
        
        # 最新技術トレンド（最頻出技術）
        all_techniques = []
        for paper in self.collected_papers:
            all_techniques.extend(paper.key_techniques)
        
        technique_counts = {}
        for tech in all_techniques:
            technique_counts[tech] = technique_counts.get(tech, 0) + 1
        
        top_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 高価値論文
        high_value_papers = sorted(
            self.collected_papers,
            key=lambda p: p.practical_value * p.novelty_score,
            reverse=True
        )[:5]
        
        report = {
            "collection_summary": {
                "total_papers": total_papers,
                "collection_period": f"{(max(p.published_date for p in self.collected_papers) - min(p.published_date for p in self.collected_papers)).days}日間",
                "last_updated": datetime.utcnow().isoformat()
            },
            "applicability_distribution": applicability_dist,
            "complexity_distribution": complexity_dist,
            "category_distribution": category_dist,
            "gpu_requirement_ratio": f"{gpu_ratio:.1%}",
            "average_scores": {
                "novelty": f"{avg_novelty:.2f}",
                "practical_value": f"{avg_practical:.2f}"
            },
            "trending_techniques": top_techniques,
            "high_value_papers": [
                {
                    "title": paper.title,
                    "arxiv_id": paper.arxiv_id,
                    "practical_value": paper.practical_value,
                    "novelty_score": paper.novelty_score,
                    "key_techniques": paper.key_techniques[:3]
                }
                for paper in high_value_papers
            ],
            "implementation_recommendations": await self._generate_implementation_priorities()
        }
        
        return report
    
    async def _generate_implementation_priorities(self) -> List[str]:
        """実装優先度生成"""
        high_applicability_papers = [
            p for p in self.collected_papers
            if p.kaggle_applicability == ApplicabilityLevel.HIGH
        ]
        
        if not high_applicability_papers:
            return ["高適用性論文が不足しています"]
        
        # 実装時間・価値でソート
        priorities = []
        
        for paper in high_applicability_papers[:5]:
            priority_text = f"{paper.title[:50]}... - {paper.estimated_implementation_time}, 価値スコア: {paper.practical_value:.2f}"
            priorities.append(priority_text)
        
        return priorities