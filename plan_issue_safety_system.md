# Issue安全連携システム実装計画書

## 概要
READMEの設計に基づくエージェント間連携の中核システム。GitHub Issue APIによる安全な自動連携、競合状態・デッドロック防止、原子性操作保証を担当する連携基盤システム。

## 実装アーキテクチャ

### 1. 技術スタック選択

#### Python + asyncio + aiohttp
**採用理由**: 
- GitHub API の高効率並列アクセス・レート制限対応
- 複数エージェント同時実行での競合状態回避
- 原子性操作・楽観的ロックの非同期実装

#### PyGithub + github3.py + requests
**採用理由**:
- GitHub Issue操作の包括API・高信頼性実装
- ETag/If-Match ヘッダによる楽観的ロック実装
- Webhook・リアルタイム通知システム構築

#### Redis + SQLite + filelock
**採用理由**:
- 分散ロック・セマフォによる排他制御
- エージェント実行状態・依存関係の永続化
- ローカル・分散両対応の柔軟な競合回避

### 2. コアモジュール設計

```
system/issue_safety_system/
├── __init__.py
├── issue_safety_manager.py      # メイン安全管理クラス
├── concurrency_control/
│   ├── atomic_operations.py     # 原子性操作・トランザクション管理
│   ├── deadlock_prevention.py   # デッドロック検出・防止
│   ├── lock_manager.py          # 分散ロック・セマフォ管理
│   └── conflict_resolver.py     # 競合解決・調停システム
├── issue_managers/
│   ├── safe_issue_creator.py    # 安全なIssue作成・重複防止
│   ├── issue_updater.py         # 楽観的ロックによる更新
│   ├── label_synchronizer.py    # ラベル同期・一貫性保証
│   └── notification_dispatcher.py # エージェント通知・起動管理
├── dependency_trackers/
│   ├── agent_dependency_graph.py # エージェント依存関係管理
│   ├── execution_order_enforcer.py # 実行順序強制・順序保証
│   ├── circular_dependency_detector.py # 循環依存検出・防止
│   └── timeout_manager.py       # タイムアウト・デッドライン管理
├── state_machines/
│   ├── agent_state_tracker.py   # エージェント状態追跡
│   ├── competition_state_manager.py # コンペ状態・フェーズ管理
│   ├── workflow_orchestrator.py  # ワークフロー・フロー制御
│   └── recovery_manager.py      # 障害復旧・状態復元
└── utils/
    ├── github_api_wrapper.py    # GitHub API安全ラッパー
    ├── retry_mechanism.py       # 指数バックオフ・リトライ
    └── audit_logger.py          # 操作ログ・監査証跡
```

**設計根拠**:
- **安全性最優先**: 競合・データ破損の完全防止
- **原子性保証**: 全操作の成功・失敗の一貫性確保
- **スケーラビリティ**: 複数コンペ・エージェント並行対応

### 3. 原子性操作・競合回避システム

#### 楽観的ロック・ETag活用
```python
class AtomicIssueOperations:
    def __init__(self, github_client):
        self.github = github_client
        self.retry_config = {
            "max_attempts": 5,
            "base_delay": 1.0,
            "max_delay": 16.0,
            "exponential_base": 2.0
        }
    
    async def create_issue_atomically(self, title, body, labels, assignees=None):
        """重複防止・原子的Issue作成"""
        # 重複チェック用一意識別子生成
        unique_identifier = self.generate_unique_identifier(title, labels)
        
        # 分散ロック取得（重複作成防止）
        async with self.acquire_creation_lock(unique_identifier):
            # 既存Issue重複チェック
            existing_issue = await self.check_duplicate_issue(title, labels)
            if existing_issue:
                return {
                    "issue": existing_issue,
                    "created": False,
                    "reason": "duplicate_exists"
                }
            
            # 原子的Issue作成実行
            try:
                issue = await self.github.create_issue(
                    title=title,
                    body=body,
                    labels=labels,
                    assignees=assignees or []
                )
                
                # 作成成功の確認・検証
                created_issue = await self.verify_issue_creation(issue.number)
                
                return {
                    "issue": created_issue,
                    "created": True,
                    "timestamp": datetime.utcnow()
                }
                
            except Exception as e:
                # 作成失敗時のクリーンアップ
                await self.cleanup_failed_creation(unique_identifier)
                raise
    
    async def update_issue_with_optimistic_lock(self, issue_number, updates):
        """楽観的ロックによる安全なIssue更新"""
        for attempt in range(self.retry_config["max_attempts"]):
            try:
                # 現在のIssue状態・ETag取得
                current_issue = await self.github.get_issue(issue_number)
                current_etag = current_issue.etag
                
                # 更新内容の準備・検証
                update_data = await self.prepare_update_data(current_issue, updates)
                
                # 楽観的ロック付き更新実行
                updated_issue = await self.github.update_issue(
                    issue_number=issue_number,
                    if_match=current_etag,  # 楽観的ロック
                    **update_data
                )
                
                return {
                    "issue": updated_issue,
                    "updated": True,
                    "attempt": attempt + 1
                }
                
            except ConflictError as e:
                # 競合検出時の指数バックオフ
                if attempt < self.retry_config["max_attempts"] - 1:
                    delay = self.calculate_backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise ConcurrencyConflictError(
                        f"Failed to update issue {issue_number} after {self.retry_config['max_attempts']} attempts"
                    )
            
            except Exception as e:
                # その他のエラー・即座失敗
                raise IssueUpdateError(f"Issue update failed: {str(e)}")
    
    async def batch_operation_atomically(self, operations):
        """複数Issue操作の原子的実行"""
        # 操作順序の依存関係解析
        operation_graph = self.build_operation_dependency_graph(operations)
        execution_order = self.topological_sort(operation_graph)
        
        # 全操作用分散ロック取得
        required_locks = self.identify_required_locks(operations)
        
        async with self.acquire_multiple_locks(required_locks):
            executed_operations = []
            rollback_operations = []
            
            try:
                # 順序保証された操作実行
                for operation in execution_order:
                    result = await self.execute_single_operation(operation)
                    executed_operations.append(result)
                    
                    # ロールバック用逆操作準備
                    rollback_op = self.prepare_rollback_operation(operation, result)
                    rollback_operations.insert(0, rollback_op)  # LIFO順
                
                return {
                    "success": True,
                    "executed_operations": executed_operations,
                    "total_operations": len(operations)
                }
                
            except Exception as e:
                # 失敗時の自動ロールバック
                await self.execute_rollback_operations(rollback_operations)
                raise BatchOperationError(
                    f"Batch operation failed, rolled back {len(rollback_operations)} operations: {str(e)}"
                )
```

**原子性根拠**:
- **一貫性保証**: 全操作成功または全操作失敗の一貫性
- **競合回避**: ETag楽観的ロックによる更新競合防止
- **自動復旧**: 失敗時の自動ロールバック・状態復元

#### デッドロック検出・防止システム
```python
class DeadlockPreventionSystem:
    def __init__(self):
        self.agent_dependency_graph = nx.DiGraph()
        self.execution_timeouts = {
            "planner": 1800,    # 30分
            "analyzer": 7200,   # 2時間
            "executor": 14400,  # 4時間
            "monitor": 28800,   # 8時間（継続監視）
            "retrospective": 3600  # 1時間
        }
        
    async def detect_potential_deadlock(self, agent_requests):
        """デッドロック可能性の事前検出"""
        # 現在の待機・実行状態の分析
        current_state = await self.get_current_agent_states()
        
        # 新しいリクエストによる依存関係追加
        updated_graph = self.add_new_dependencies(
            current_graph=self.agent_dependency_graph,
            new_requests=agent_requests
        )
        
        # 循環依存の検出
        cycles = list(nx.simple_cycles(updated_graph))
        
        if cycles:
            return {
                "deadlock_risk": True,
                "detected_cycles": cycles,
                "affected_agents": self.extract_affected_agents(cycles),
                "resolution_strategies": await self.generate_resolution_strategies(cycles)
            }
        
        return {"deadlock_risk": False}
    
    async def prevent_deadlock_formation(self, agent_execution_plan):
        """デッドロック形成の事前防止"""
        # 実行順序の強制（planner → analyzer → executor → monitor）
        enforced_order = self.enforce_strict_execution_order(agent_execution_plan)
        
        # 逆依存の禁止（後段→前段のIssue作成禁止）
        validated_plan = self.validate_no_reverse_dependencies(enforced_order)
        
        # タイムアウト・デッドライン設定
        timeout_plan = self.apply_execution_timeouts(validated_plan)
        
        return {
            "safe_execution_plan": timeout_plan,
            "deadlock_prevention_applied": True,
            "execution_constraints": self.get_applied_constraints(timeout_plan)
        }
    
    async def break_existing_deadlock(self, deadlocked_agents):
        """既存デッドロックの強制解除"""
        # 最も古い待機エージェントの特定
        oldest_wait = min(deadlocked_agents, key=lambda x: x.wait_start_time)
        
        # 強制タイムアウト・解除実行
        await self.force_timeout_agent(oldest_wait)
        
        # 解除後の依存関係クリーンアップ
        await self.cleanup_deadlock_dependencies(deadlocked_agents)
        
        # 影響を受けたエージェントの再起動
        restart_plan = await self.create_restart_plan(deadlocked_agents)
        
        return {
            "deadlock_broken": True,
            "forced_timeout_agent": oldest_wait.agent_id,
            "restart_plan": restart_plan,
            "recovery_actions": await self.generate_recovery_actions(deadlocked_agents)
        }
```

**防止根拠**:
- **順序強制**: 厳密な実行順序による循環依存防止
- **タイムアウト**: 無限待機・デッドロック状態の強制解除
- **早期検出**: 問題発生前の予防的検出・回避

### 4. エージェント実行状態・依存関係管理

#### エージェント状態追跡システム
```python
class AgentStateTracker:
    def __init__(self):
        self.state_transitions = {
            "idle": ["starting", "error"],
            "starting": ["running", "error"],
            "running": ["completed", "error", "timeout"],
            "completed": ["idle"],
            "error": ["idle", "starting"],  # リトライ可能
            "timeout": ["idle", "starting"]  # リスタート可能
        }
        
    async def track_agent_lifecycle(self, agent_id, competition_name):
        """エージェント生存期間の完全追跡"""
        lifecycle_record = {
            "agent_id": agent_id,
            "competition": competition_name,
            "start_time": datetime.utcnow(),
            "current_state": "idle",
            "state_history": [],
            "dependencies": [],
            "blocking_agents": [],
            "blocked_by": []
        }
        
        # 状態変更の監視・記録
        while not self.is_final_state(lifecycle_record["current_state"]):
            state_change = await self.wait_for_state_change(agent_id)
            
            # 状態遷移の妥当性検証
            if self.is_valid_transition(
                lifecycle_record["current_state"], 
                state_change.new_state
            ):
                # 状態更新・履歴記録
                lifecycle_record["state_history"].append({
                    "from_state": lifecycle_record["current_state"],
                    "to_state": state_change.new_state,
                    "timestamp": state_change.timestamp,
                    "trigger": state_change.trigger_event
                })
                lifecycle_record["current_state"] = state_change.new_state
                
                # 依存関係の動的更新
                await self.update_agent_dependencies(agent_id, state_change)
                
            else:
                # 無効な状態遷移・エラー処理
                await self.handle_invalid_state_transition(agent_id, state_change)
        
        return lifecycle_record
    
    async def manage_inter_agent_dependencies(self, agents):
        """エージェント間依存関係の動的管理"""
        dependency_matrix = self.build_dependency_matrix(agents)
        
        for agent in agents:
            # 依存関係チェック・実行可能性判定
            dependencies_met = await self.check_dependencies_satisfied(
                agent=agent,
                dependency_matrix=dependency_matrix
            )
            
            if dependencies_met:
                # 実行可能・起動許可
                await self.authorize_agent_execution(agent)
            else:
                # 依存関係未満足・待機状態継続
                await self.maintain_waiting_state(agent)
                
                # 依存先エージェントの進捗監視
                await self.monitor_dependency_progress(agent)
        
        # 依存関係解決・実行許可の連鎖反応処理
        await self.process_dependency_resolution_cascade(agents)
```

#### Issue安全連携フロー管理
```python
class SafeCollaborationFlowManager:
    async def orchestrate_safe_agent_handoff(self, source_agent, target_agent, context):
        """エージェント間の安全な引き継ぎ実行"""
        handoff_transaction = HandoffTransaction(
            source=source_agent,
            target=target_agent,
            context=context,
            transaction_id=self.generate_transaction_id()
        )
        
        try:
            # Phase 1: 引き継ぎ準備・検証
            await self.prepare_handoff(handoff_transaction)
            
            # Phase 2: 原子的データ転送
            await self.execute_atomic_data_transfer(handoff_transaction)
            
            # Phase 3: ターゲットエージェント起動
            await self.activate_target_agent(handoff_transaction)
            
            # Phase 4: ソースエージェント完了処理
            await self.finalize_source_agent(handoff_transaction)
            
            return {
                "handoff_success": True,
                "transaction_id": handoff_transaction.transaction_id,
                "completion_time": datetime.utcnow()
            }
            
        except Exception as e:
            # 引き継ぎ失敗・ロールバック実行
            await self.rollback_handoff_transaction(handoff_transaction)
            raise HandoffFailureError(f"Agent handoff failed: {str(e)}")
    
    async def enforce_competition_isolation(self, operation_request):
        """コンペ間操作の完全分離保証"""
        # 操作対象コンペの特定・検証
        target_competition = self.extract_competition_context(operation_request)
        
        # 他コンペIssueへの誤操作防止チェック
        isolation_violations = await self.check_isolation_violations(
            operation=operation_request,
            target_competition=target_competition
        )
        
        if isolation_violations:
            raise IsolationViolationError(
                f"Operation would violate competition isolation: {isolation_violations}"
            )
        
        # 安全な分離環境での操作実行
        async with self.competition_isolation_context(target_competition):
            return await self.execute_isolated_operation(operation_request)
```

### 5. プロンプト設計計画

#### Issue安全操作プロンプト構造
```yaml
# Issue安全操作時の標準プロンプト
safe_issue_operation_prompt: |
  # Issue安全連携システム操作指示
  
  ## 安全性原則
  あなたはGitHub Issue操作時に以下の安全性原則を厳守してください。
  
  ## Issue作成時の安全手順
  1. **重複チェック必須**: 同一comp+agentラベルの既存Issue確認
  2. **原子的作成**: 作成失敗時の部分状態残存防止
  3. **一意識別**: title+labelsによる確実な識別子設定
  4. **競合回避**: 同時作成時の重複防止ロック取得
  
  ### Issue作成コマンド例
  ```python
  # 安全なIssue作成（重複防止・原子性保証）
  issue_result = await safe_issue_creator.create_issue_atomically(
      title="[{competition_name}] {agent_type}: {task_description}",
      body=issue_content,
      labels=[
          f"agent:{agent_type}",
          f"comp:{competition_name}",
          "status:auto-processing",
          f"priority:{priority_level}"
      ]
  )
  
  if not issue_result["created"]:
      # 重複Issue存在・既存Issue使用
      existing_issue = issue_result["issue"]
      await handle_duplicate_issue_scenario(existing_issue)
  ```
  
  ## Issue更新時の安全手順
  1. **楽観的ロック**: ETag使用・競合検出時自動リトライ
  2. **部分更新**: 必要項目のみ更新・全体書き換え禁止
  3. **状態一貫性**: ラベル・ステータスの論理的一貫性確保
  4. **変更履歴**: 全更新操作の監査ログ記録
  
  ### Issue更新コマンド例
  ```python
  # 楽観的ロック付き安全更新
  update_result = await issue_updater.update_issue_with_optimistic_lock(
      issue_number=target_issue.number,
      updates={
          "labels": new_labels,
          "body": updated_body,
          "state": new_state
      }
  )
  ```
  
  ## エージェント起動時の安全連携
  1. **依存関係確認**: 前段エージェント完了状態の検証
  2. **環境分離**: 対象コンペディレクトリでの確実な実行
  3. **通知確認**: 起動通知の確実な送信・受信確認
  4. **タイムアウト**: 起動失敗時の自動エスカレーション
```

#### エラー処理・復旧プロンプト
```yaml
error_handling_recovery_prompt: |
  ## Issue操作エラー時の対応指針
  
  ### 競合エラー（Conflict Error）対応
  ```python
  try:
      result = await update_issue_operation(issue_data)
  except ConflictError as e:
      # 指数バックオフリトライ
      for attempt in range(max_retries):
          await asyncio.sleep(calculate_backoff_delay(attempt))
          try:
              # 最新状態取得・再試行
              fresh_issue = await fetch_latest_issue_state(issue_number)
              result = await retry_update_with_fresh_state(fresh_issue, updates)
              break
          except ConflictError:
              continue
      else:
          # 最大リトライ後失敗・人間エスカレーション
          await escalate_to_human_intervention(issue_number, "persistent_conflict")
  ```
  
  ### デッドロック検出時対応
  ```python
  deadlock_detection = await detect_potential_deadlock(current_agent_states)
  if deadlock_detection["deadlock_risk"]:
      # 自動デッドロック解除
      resolution = await break_deadlock_automatically(deadlock_detection["detected_cycles"])
      
      # 影響エージェントの安全リスタート
      for agent in resolution["affected_agents"]:
          await safe_agent_restart(agent, preserve_progress=True)
  ```
  
  ### 分離違反検出時対応
  ```yaml
  isolation_check:
    before_operation: |
      # 操作前コンペ分離チェック
      target_comp = extract_competition_from_labels(operation.labels)
      current_comp = get_current_working_competition()
      
      if target_comp != current_comp:
          raise IsolationViolationError(
              f"Cross-competition operation detected: {current_comp} -> {target_comp}"
          )
    
    recovery_action: |
      # 分離違反時の自動修正
      await switch_to_correct_competition_context(target_comp)
      await retry_operation_in_correct_context(operation)
  ```
```

### 6. 実行環境・コンペ分離システム

#### 完全分離実行保証
```python
class CompetitionIsolationEnforcer:
    async def ensure_execution_isolation(self, agent_type, competition_name):
        """完全な実行環境分離の保証"""
        isolation_context = {
            "working_directory": f"competitions/{competition_name}/",
            "virtual_environment": f"competitions/{competition_name}/.venv/",
            "issue_filter": f"comp:{competition_name}",
            "agent_scope": f"{agent_type}",
            "resource_namespace": f"{competition_name}_{agent_type}"
        }
        
        # 作業ディレクトリの強制設定・検証
        await self.enforce_working_directory(isolation_context["working_directory"])
        
        # 仮想環境の活性化・依存関係分離
        await self.activate_isolated_environment(isolation_context["virtual_environment"])
        
        # Issue操作フィルタの厳密適用
        await self.apply_issue_filter(isolation_context["issue_filter"])
        
        return isolation_context
    
    async def validate_operation_scope(self, operation, expected_scope):
        """操作スコープの厳密検証"""
        # Issue操作の対象コンペ検証
        target_competition = self.extract_competition_from_operation(operation)
        
        if target_competition != expected_scope["competition"]:
            raise ScopeViolationError(
                f"Operation targets wrong competition: expected {expected_scope['competition']}, "
                f"got {target_competition}"
            )
        
        # エージェント種別の検証
        target_agent = self.extract_agent_type_from_operation(operation)
        
        if target_agent != expected_scope["agent_type"]:
            raise ScopeViolationError(
                f"Operation from wrong agent: expected {expected_scope['agent_type']}, "
                f"got {target_agent}"
            )
        
        return True
```

### 7. 初期実装スコープ

#### Phase 1: 基本安全機能（1週間）
1. **原子的Issue操作**: 基本的な作成・更新の原子性保証
2. **重複防止**: 同一ラベル組合せの重複Issue防止
3. **基本楽観的ロック**: ETagによる更新競合回避
4. **エラーログ**: 操作失敗・競合の詳細ログ記録

#### Phase 2: 高度安全機能（2週間）
1. **デッドロック防止**: 循環依存検出・予防システム
2. **分散ロック**: Redis/ファイルロックによる排他制御
3. **バッチ原子性**: 複数操作の原子的実行・ロールバック
4. **依存関係管理**: エージェント間依存の動的追跡・制御

#### Phase 3: 完全自動化（1週間）
1. **自動復旧**: 障害・競合からの自動回復システム
2. **予測的制御**: 問題発生前の予防的制御・回避
3. **最適化**: 性能・効率最適化・レスポンス向上
4. **全エージェント統合**: 完全自動安全連携システム

### 8. テスト戦略

#### 競合状況シミュレーションテスト
- 複数エージェント同時実行での競合・デッドロック検証
- 高負荷状況での原子性・一貫性確認
- ネットワーク障害・タイムアウト耐性テスト

#### 安全性保証テスト
- 分離違反・誤操作の完全防止確認
- ロールバック・復旧機能の正確性検証
- 長期運用での状態一貫性・信頼性測定

#### パフォーマンステスト
- 安全機能オーバーヘッドの測定・最適化
- 大量Issue・エージェントでのスケーラビリティ確認
- GitHub API制限下での効率・安定性検証

## 成功指標

1. **安全性**: Issue操作競合・データ破損発生率 = 0%
2. **原子性**: バッチ操作の部分失敗率 < 0.1%
3. **可用性**: デッドロック・無限待機発生率 < 0.01%
4. **性能**: 安全機能による処理遅延 < 10%

## リスク対策

### 技術リスク
- **GitHub API制限**: レート制限・複数アカウント・キャッシュ戦略
- **分散システム障害**: 冗長化・自動フェイルオーバー・状態復元
- **競合状態複雑化**: 段階的機能追加・十分なテスト・保守的設計

### 運用リスク
- **過度な安全制御**: 性能バランス・必要最小限の制御・効率重視
- **システム複雑化**: シンプルな設計・明確な責任分離・文書化充実
- **学習コスト**: 段階的導入・明確なAPI・豊富な例示・エラーメッセージ

この実装計画により、READMEのIssue安全連携システム仕様を完全実現し、エージェント間の確実で安全な自動連携システムを構築します。