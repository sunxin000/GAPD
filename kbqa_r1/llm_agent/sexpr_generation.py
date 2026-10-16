"""
Enhanced LLM Generation Manager with S-Expression support
Replaces SPARQL-based generation with action-based reasoning and S-Expression execution
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import torch

from verl import DataProto

# Import S-Expression components
from ..sexpr import (ActionParser, FunctionBuilder, SExprExecutor,
                     SExprGenerator, SExprValidator)
from ..sexpr.action_parser import ActionResult, ActionType
from ..sexpr.relation_retrieval import RelationRetrieval
# Import SPARQL manager for backward compatibility and execution
from ..sparql.sparql_manager import SPARQLConfig, SPARQLExecutionManager
# Import GPU utilization manager
from .gpu_utilization_manager import (GPUUtilizationConfig,
                                      GPUUtilizationManager)
from .independent_gpu_manager import (IndependentGPUConfig,
                                      IndependentGPUMaintenanceManager)
from .sexpr_action_processor import SExprActionProcessor
from .sexpr_batch_utils import SExprBatchUtils
# Import new modular components
from .sexpr_config import SExprGenerationConfig
from .sexpr_logging import SExprLoggingManager
from .sexpr_multi_result_formatter import SExprMultiResultFormatter
from .sexpr_result_processor import SExprResultProcessor
from .sexpr_state_manager import SExprStateManager
from .sexpr_utils import SExprUtils
from .tensor_helper import TensorConfig, TensorHelper
from .timeout_detector import TimeoutDetector, TimeoutTracker

# Configure logger with proper level
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))  # Default to INFO for S-Expression logs


class SExprLLMGenerationManager:
    """
    Enhanced LLM Generation Manager with S-Expression support
    Handles action-based reasoning instead of direct SPARQL generation
    """
    
    # Class variable for tracking calls
    _call_counter = -1
    # Class variable for tracking debug saves (limit to 10 total)
    _debug_save_counter = 0
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: SExprGenerationConfig,
        is_validation: bool = False,
        sparql_config: dict = None,
        dataset: str = "WebQSP",
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # Initialize modular components
        self.state_manager = SExprStateManager()
        self.logging_manager = SExprLoggingManager(config)
        self.batch_utils = SExprBatchUtils(self.tensor_fn, config, actor_rollout_wg, tokenizer)
        
        # Initialize timeout detection and tracking
        self.timeout_detector = TimeoutDetector()
        self.timeout_tracker = TimeoutTracker()
        
        # Initialize S-Expression components
        if config.enable_sexpr_mode:
            self.action_parser = ActionParser()
            self.function_builder = FunctionBuilder()
            self.sexpr_generator = SExprGenerator()
            self.sexpr_validator = SExprValidator() if config.enable_sexpr_validation else None
            
            # Setup S-Expression executor with SPARQL backend
            sparql_config_obj = sparql_config if sparql_config else config.get_sparql_config()
            # Respect dataset type from environment if provided (e.g., DATASET_TYPE=grailqa)
            env_dataset = os.getenv("DATASET_TYPE")
            if env_dataset and isinstance(env_dataset, str) and env_dataset.strip():
                dataset = env_dataset.strip()
            if isinstance(sparql_config_obj, dict):
                sparql_config_obj = SPARQLConfig(**sparql_config_obj)
            self.sexpr_executor = SExprExecutor(sparql_config_obj, dataset_type=dataset)
            logger.info(f"[SEXPR] Using dataset_type='{str(dataset)}' for SPARQL converter selection")
            self.relation_retrieval = RelationRetrieval(sparql_config=sparql_config_obj, dataset=dataset)
            
            # Initialize action processor
            self.action_processor = SExprActionProcessor(self.relation_retrieval, self.state_manager)
            
            # Initialize result processor with relation_retrieval and tokenizer/model name for prompt formatting
            model_name = getattr(tokenizer, 'name_or_path', None)
            self.result_processor = SExprResultProcessor(config, self.relation_retrieval, model_name=model_name, tokenizer=tokenizer)
            # Wire action processor into result processor so per-action relation states are accessible
            self.result_processor.action_processor = self.action_processor
            
            # Initialize multi-result formatter
            self.multi_result_formatter = SExprMultiResultFormatter(self.result_processor)

        else:
            # Fallback to original SPARQL-based approach
            self.action_parser = None
            self.function_builder = None 
            self.sexpr_generator = None
            self.sexpr_validator = None
            self.sexpr_executor = None
            self.action_processor = None
            self.relation_retrieval = None
            
            # Initialize result processor without relation_retrieval
            model_name = getattr(tokenizer, 'name_or_path', None)
            self.result_processor = SExprResultProcessor(config, None, model_name=model_name, tokenizer=tokenizer)
            
            # Original SPARQL manager
            if sparql_config:
                sparql_config_obj = SPARQLConfig(
                    sparql_url=sparql_config.get('sparql_url'),
                    sparql_batch_size=sparql_config.get('sparql_batch_size', 128),
                    sparql_max_concurrent=sparql_config.get('sparql_max_concurrent', 16),
                    use_odbc=sparql_config.get('use_odbc', True),
                    use_aioodbc=sparql_config.get('use_aioodbc', False),
                    odbc_config=sparql_config.get('odbc_config', None)
                )
                self.sparql_manager = SPARQLExecutionManager(sparql_config_obj)
            else:
                self.sparql_manager = SPARQLExecutionManager(config.get_sparql_config())

        # 延迟初始化GPU利用率管理器 - 不在__init__中初始化，而是在第一次使用时初始化
        self._gpu_utilization_manager = None
        # 保存配置对象，但不立即创建GPUUtilizationConfig实例
        self._gpu_utilization_config_raw = config.gpu_utilization_config
        
        # 使用独立的GPU维护管理器（绕过Ray的GPU环境限制）
        self._independent_gpu_manager = None
        self._independent_gpu_config = IndependentGPUConfig(
            gpu_id=0,  # 兼容性参数，实际会检测所有GPU
            matrix_size=2048,
            interval=0.01,
            enable_gpu_utilization_maintenance=True,
            # 多GPU配置 - 针对8卡A100环境优化
            target_memory_percentage=0.80,  # 占用80%内存
            idle_gpu_threshold=20,  # 利用率<20%的GPU被认为是空闲的
            compute_matrix_size=4096,  # 4096x4096矩阵
            status_update_interval=5  # 每5秒更新状态
        )

    def _get_independent_gpu_manager(self):
        """
        获取独立的GPU维护管理器（绕过Ray的GPU环境限制）
        """
        if self._independent_gpu_manager is None:
            logger.info("[SEXPR] Creating independent GPU maintenance manager")
            self._independent_gpu_manager = IndependentGPUMaintenanceManager(self._independent_gpu_config)
        return self._independent_gpu_manager

    def _get_gpu_utilization_manager(self):
        """
        延迟初始化GPU利用率管理器，确保在Ray环境完全设置后再初始化
        """
        if self._gpu_utilization_manager is None:
            logger.info("[SEXPR] Initializing GPU utilization manager (delayed initialization)")
            # 延迟创建配置对象
            config = self._gpu_utilization_config_raw or GPUUtilizationConfig()
            self._gpu_utilization_manager = GPUUtilizationManager(config)
        return self._gpu_utilization_manager


    @property
    def gpu_utilization_manager(self):
        """属性访问器，确保延迟初始化"""
        # 直接返回独立的GPU管理器，它本身就是一个上下文管理器
        return self._get_independent_gpu_manager()

    def __del__(self):
        """Cleanup GPU resources when object is destroyed"""
        if hasattr(self, '_gpu_utilization_manager') and self._gpu_utilization_manager is not None:
            self._gpu_utilization_manager.shutdown()
        if hasattr(self, '_independent_gpu_manager') and self._independent_gpu_manager is not None:
            self._independent_gpu_manager.stop_maintenance()


    def _is_timeout_error(self, text: str) -> bool:
        """Wrapper for timeout detection"""
        return self.timeout_detector.is_timeout_error(text)
    
    def _record_timeout(self, sample_id: int, step: int):
        """Wrapper for timeout recording"""
        self.timeout_tracker.record_timeout(step, sample_id)
    
    def get_timeout_statistics(self) -> dict:
        """Get timeout statistics"""
        return self.timeout_tracker.get_statistics()
    
    def reset_timeout_statistics(self):
        """Reset timeout statistics"""
        self.timeout_tracker.reset()

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations (same as original)"""
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            logger.warning(f"Observation too long: {next_obs_ids.shape[1]} > {self.config.max_obs_length}")
            
            # 记录完整的observation字符串到日志文件
            self.logging_manager.log_long_observations(next_obs, next_obs_ids.shape[1])
            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_execution=True, turn: int = 0) -> Tuple[List[str], List[bool]]:
        """
        Execute predictions using S-Expression approach or fallback to SPARQL
        Main change: parse actions and convert to S-Expressions instead of direct SPARQL
        """
        batch_size = len(predictions)
        next_obs = [""] * batch_size
        dones = [False] * batch_size
        
        # Initialize timeout count for this step if not exists
        if turn not in self.timeout_tracker.per_step:
            self.timeout_tracker.per_step[turn] = 0

        # Controlled logging for execute_predictions
        logger.info(f"[SEXPR] Executing predictions | Batch: {len(predictions)} | S-Expression mode: {self.config.enable_sexpr_mode}")
        
        logger.debug(f"execute_predictions: predictions length={len(predictions)}, sexpr_mode={self.config.enable_sexpr_mode}")
        
        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool)
        
        # Define per-sample processing to enable parallel execution while keeping ordering
        def _process_single(i_pred):
            i, pred = i_pred
            local_next_obs = ""
            local_done = False
            local_timeout = False  # Track timeout for this sample
            
            if not active_mask[i]:
                logger.info(f"[SEXPR] Sample {i}: Skipped (inactive)")
                return (i, local_next_obs, True, False)

            answer_match = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
            if answer_match:
                logger.info(f"[SEXPR] Sample {i}: Completed (found answer)")
                logger.info(f"[SEXPR] Sample {i}: prediction: {pred}")
                # 清除样本状态，因为已经完成
                self.state_manager.clear_sample_state(i)
                return (i, local_next_obs, True, False)

            if not do_execution:
                logger.info(f"[SEXPR] Sample {i}: Skipped (no execution mode)")
                return (i, local_next_obs, False, False)

            if self.config.enable_sexpr_mode and self.action_parser:
                try:
                    observation = self._process_sexpr_prediction(pred, i, turn=turn)
                    observation = SExprUtils.ensure_leading_newline_info(observation)
                    local_next_obs = observation
                    
                    # Check if the observation contains timeout error
                    if self._is_timeout_error(observation):
                        local_timeout = True
                        local_done = True  # Stop this rollout
                        logger.warning(f"[TIMEOUT] Sample {i}: SPARQL timeout detected at step {turn}, stopping rollout")
                except Exception as e:
                    # Check if exception message contains timeout indicators
                    if self._is_timeout_error(str(e)):
                        local_timeout = True
                        local_done = True
                        local_next_obs = f"<information>\nSPARQL execution timeout: {str(e)}\n</information>"
                        logger.warning(f"[TIMEOUT] Sample {i}: SPARQL timeout exception at step {turn}, stopping rollout")
                    else:
                        raise  # Re-raise non-timeout exceptions
            else:
                # Original SPARQL mode
                observation = self._process_sparql_prediction(pred, i)
                observation = SExprUtils.ensure_leading_newline_info(observation)
                local_next_obs = observation
                
                # Check timeout in SPARQL mode too
                if self._is_timeout_error(observation):
                    local_timeout = True
                    local_done = True
                    logger.warning(f"[TIMEOUT] Sample {i}: SPARQL timeout detected at step {turn} (SPARQL mode)")
                    
            return (i, local_next_obs, local_done, local_timeout)

        # Parallel execution across predictions (threaded; mostly I/O-bound work)
        max_workers = None
        env_workers = os.getenv("PREDICTION_PARALLEL_WORKERS")
        if env_workers:
            max_workers = max(1, int(env_workers))
        if not max_workers:
            # Reasonable default cap to avoid oversubscription
            max_workers = min(len(predictions), 16)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, obs, done, timeout in executor.map(_process_single, enumerate(predictions)):
                if obs:
                    next_obs[i] = obs
                if done:
                    dones[i] = True
                if timeout:
                    # Update timeout tracking
                    self._record_timeout(i, turn)
                    # Mark as inactive in active_mask if available
                    if active_mask is not None:
                        active_mask[i] = False
        
        # Log timeout statistics for this step
        step_timeout_count = self.timeout_tracker.per_step.get(turn, 0)
        if step_timeout_count > 0:
            logger.warning(f"[TIMEOUT-STATS] Step {turn}: {step_timeout_count} timeouts occurred in this step")
            logger.warning(f"[TIMEOUT-STATS] Total timeouts so far: {self.timeout_tracker.total}")
                
        return next_obs, dones

    def _process_sexpr_prediction(self, prediction: str, index: int, function_state: List[str] = None, candidate_entities: List[Tuple[str, str]] = None, turn: int = 0) -> str:
        """
        Process prediction by extracting actions from <action></action> tags and executing them
        This is the correct approach for kbqa-r1 training format where LLM generates action sequences in tags
        """
        # Calculate if this is the final turn
        is_final_turn = (turn == self.config.max_turns - 1)
        
        # Clear action states for new turn (action states should not persist across turns)
        # Function states are kept persistent as they accumulate across turns
        self.action_processor.clear_sample_action_states(index)
        
        # Parse actions from prediction using <action></action> tags
        actions = self.action_parser.parse_actions_from_text(prediction)
        
        # Log all S-Expression processing samples
        if actions:
            logger.info(f"[SEXPR] Sample {index}: Parsed {len(actions)} actions from prediction")
            for i, action in enumerate(actions[:2]):  # Log first 2 actions only
                # Show full argument list to avoid truncation (e.g., COMPARE has 3 args)
                logger.info(f"[SEXPR]   Action {i+1}: {action.action_type.name} - {action.arguments if action.arguments else 'No args'}")
        
        if not actions:
            logger.info(f"[SEXPR] Sample {index}: No valid actions found in prediction")
            info_block = "<information>\nNo valid actions found. Please provide reasoning actions.\n</information>"
            return self.result_processor._add_guidance_prompt(info_block, is_final_turn)
        
        # Update action attempts for special actions present in this prediction
        present_special = set()
        for a in actions:
            if a.action_type in self.action_processor._special_action_names:
                present_special.add(self.action_processor._special_action_names[a.action_type])
        for name in present_special:
            self.action_processor._action_metrics[name]['attempts'] += 1

        # 获取持久化的function_state和实体信息
        persistent_function_state = self.state_manager.get_sample_function_state(index)
        sample_entities = self.state_manager.get_sample_entities(index)
        
        # 使用持久化的function_state，如果没有则使用传入的
        if persistent_function_state:
            function_state = persistent_function_state
            logger.info(f"[SEXPR] Sample {index}: Using persistent function_state with {len(function_state)} functions")
        else:
            function_state = function_state or []
            logger.info(f"[SEXPR] Sample {index}: Using provided function_state with {len(function_state)} functions")
        
        # 使用样本的实体信息，如果没有则使用传入的
        if sample_entities:
            candidate_entities = sample_entities
            logger.info(f"[SEXPR] Sample {index}: Using persistent entities with {len(candidate_entities)} entities")
        else:
            candidate_entities = candidate_entities or []
            logger.info(f"[SEXPR] Sample {index}: Using provided entities with {len(candidate_entities)} entities")
            # 如果提供了实体信息，保存到持久化存储中
            if candidate_entities:
                self.state_manager.set_sample_entities(index, candidate_entities)
        
        # Process actions with relation/entity retrieval
        # Record current function state length for potential rollback
        initial_function_state_length = len(self.state_manager.get_sample_function_state(index))
        
        processed_actions = self.action_processor.process_actions_with_retrieval(
            actions, function_state, candidate_entities, sample_id=index
        )
        
        if not processed_actions:
            info_block = "<information>\nFailed to process actions with retrieval.\n</information>"
            return self.result_processor._add_guidance_prompt(info_block, is_final_turn)
        
        # Special case: if all actions are FIND_RELATION, handle them independently regardless of validity
        if len(processed_actions) > 1 and self._all_actions_are_find_relation(processed_actions):
            logger.info(f"[SEXPR] Sample {index}: All actions are FIND_RELATION, processing each independently")
            return self._handle_find_relation_actions_independently(processed_actions, index, turn)
        
        # Check if any processed actions are invalid and return error information
        invalid_actions = [action for action in processed_actions if not action.is_valid]
        if invalid_actions:
            error_messages = []
            for action in invalid_actions:
                if hasattr(action, 'error_message') and action.error_message:
                    error_messages.append(f"Action {action.action_type.name}: {action.error_message}")
                else:
                    error_messages.append(f"Action {action.action_type.name}: Invalid action")
            
            error_info = "\n".join(error_messages)
            info_block = f"<information>\nAction processing failed:\n{error_info}\n</information>"
            return self.result_processor._add_guidance_prompt(info_block, is_final_turn)
        
        # Check if we have multiple actions that need independent execution
        if len(processed_actions) > 1:

            # Try normal execution first
            # Set executor error context so any SPARQL errors include input/pred
            try:
                input_text = self.state_manager.get_sample_prompt(index)
            except Exception:
                input_text = None
            self.sexpr_executor.set_error_context(input_text=input_text, prediction_text=prediction)
            normal_result = self._execute_action_list_normal(processed_actions, index, initial_function_state_length, is_final_turn=is_final_turn)
            
            # Check if normal execution returned empty results
            if SExprUtils.is_empty_result(normal_result):
                logger.info(f"[SEXPR] Sample {index}: Normal execution returned empty results, trying independent execution")
                return self._execute_actions_independently(processed_actions, index, candidate_entities, turn)
            else:
                return normal_result
        else:
            # Single action, use normal execution
            try:
                input_text = self.state_manager.get_sample_prompt(index)
            except Exception:
                input_text = None
            self.sexpr_executor.set_error_context(input_text=input_text, prediction_text=prediction)
            return self._execute_action_list_normal(processed_actions, index, initial_function_state_length, is_final_turn=is_final_turn)

    
    def _execute_action_list_normal(self, processed_actions: List[ActionResult], index: int, initial_function_state_length: int = None, is_final_turn: bool = False) -> str:
        """
        Execute action list using normal approach (all actions together)
        """
        # Build function sequence        
        # Generate S-Expression using persistent function_state
        # 参考KBQA-o1的functions_to_expression机制
        persistent_function_state = self.state_manager.get_sample_function_state(index)
        if persistent_function_state:
            # 使用持久化的function_state，避免重复添加已存在的function
            # function_calls 中的 function 已经在 _process_actions_with_retrieval 中被添加到 persistent_function_state
            all_function_strings = persistent_function_state
            # Infer target expression from the last assignment (KBQA-o1 style)
            # Preference: if the last assignment is STOP(expressionK), use K; else use LHS expressionK of the last assignment
            target_expr = None
            try:
                for stmt in reversed(all_function_strings):
                    if not isinstance(stmt, str):
                        continue
                    s = stmt.strip()
                    # Match STOP wrapper on RHS and extract inner expressionK
                    m_stop = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*STOP\(\s*(expression\d+)\s*\)\s*$", s)
                    if m_stop:
                        target_expr = m_stop.group(1)
                        break
                    # Otherwise, take the LHS expressionK of the last assignment
                    m_lhs = re.match(r"^(expression\d+)\s*=", s)
                    if m_lhs and target_expr is None:
                        target_expr = m_lhs.group(1)
                        # Do not break here; keep scanning in case a later STOP(...) exists above
                if not target_expr:
                    # Fallback to current pointer if parsing failed
                    target_expr = f"expression{self.state_manager.get_current_expression_id(index)}"
            except Exception:
                target_expr = f"expression{self.state_manager.get_current_expression_id(index)}"
            sexpr_result = self.sexpr_generator.generate_sexpr_from_strings(all_function_strings, target_expr)
            logger.info(f"[SEXPR] Sample {index}: Generated S-Expression using persistent state with {len(persistent_function_state)} functions")
        else:
            # function_calls = self.function_builder.build_function_sequence(processed_actions)
            # # Fallback to original method
            # if not function_calls:
            #     return "<information>\nFailed to build function sequence from actions.\n</information>"
            # sexpr_result = self.sexpr_generator.generate_sexpr_from_functions(function_calls)
            # logger.info(f"[SEXPR] Sample {index}: Generated S-Expression using function_calls only")
            raise ValueError("Failed to build function sequence from actions.")
        
        if not sexpr_result.is_valid:
            return f"<information>\nS-Expression generation failed: {sexpr_result.error_message}\n</information>"
        
        # Validate S-Expression if enabled
        if self.sexpr_validator:
            validation_result = self.sexpr_validator.validate(sexpr_result.sexpr)
            if not validation_result.is_valid:
                error_msg = "; ".join(validation_result.errors)
                return f"<information>\nS-Expression validation failed: {error_msg}\n</information>"
        
        # Execute S-Expression with enhanced logging for timeout detection
        logger.info(f"[SEXPR-EXEC] Sample {index}: About to execute S-Expression: {sexpr_result.sexpr}")
        
        try:
            # Get function_state for debugging context
            persistent_function_state = self.state_manager.get_sample_function_state(index)
            execution_result = self.sexpr_executor.execute_sexpr(sexpr_result.sexpr, function_state=persistent_function_state)
            
            # Log the generated SPARQL for debugging
            if execution_result.sparql:
                logger.info(f"[SEXPR-EXEC] Sample {index}: Generated SPARQL query: {execution_result.sparql[:200]}...")
                
        except RuntimeError as e:
            if "Dangerous timeout query detected" in str(e):
                # Add sample context to the timeout error
                enhanced_error = f"Sample {index} - {str(e)}"
                logger.error(f"[TIMEOUT-ERROR] {enhanced_error}")
                # Re-raise with sample context
                raise RuntimeError(enhanced_error) from e
            else:
                # Re-raise other RuntimeErrors as-is
                raise
        
        # Log execution results with sample tracking
        logger.info(f"[SEXPR] Sample {index}: S-Expression execution - Success: {execution_result.is_successful}")
        if execution_result.is_successful and execution_result.results:
            logger.info(f"[SEXPR] Sample {index}: Found {len(execution_result.results)} results")
        elif not execution_result.is_successful:
            logger.info(f"[SEXPR] Sample {index}: Execution error - {execution_result.error_message[:100]}...")

        if execution_result.is_successful:
            # Update successes for special actions if execution succeeded and has results
            mid_list = self.result_processor.extract_mid_list(execution_result.results)
            # Surface per-action relation states for UI (legacy removed)
            
            # Transfer per-action relation states for multiple Find_relation support
            if hasattr(self.action_processor, '_per_action_relation_states'):
                self.result_processor.set_per_action_relation_states(self.action_processor._per_action_relation_states)
            
            if len(mid_list) > 0:
                present_special = set()
                for a in processed_actions:
                    if a.action_type in self.action_processor._special_action_names:
                        present_special.add(self.action_processor._special_action_names[a.action_type])
                for name in present_special:
                    self.action_processor._action_metrics[name]['successes'] += 1
            
            # Build enriched info block
            # 原始实现仅在 "单个 FIND_RELATION" 时传入 action_index，导致
            # COMPARE / ORDER 等单动作场景虽然写入了 per-action relation state，
            # 但 result_processor 无法通过 (sample_id, action_index) 读取到这些状态，
            # 因而不会展示 selected_best_relation 与 top-5。
            #
            # 这里放宽策略：
            # - 只要本轮是「单个动作」，就把该动作的 action_index 传下去；
            # - 这样单个 COMPARE / ORDER / FIND_RELATION 等都能在 enriched info 中展示
            #   对应的 relation 选择及候选信息；
            # - 多动作链保持行为不变（不传 action_index），避免多步场景下误解绑定哪个 action。
            action_index = None
            if len(processed_actions) == 1:
                # action_index 一般由 process_actions_with_retrieval 赋值；
                # 若缺失则退回 step_number（兼容旧数据）；
                action = processed_actions[0]
                action_index = getattr(action, 'action_index', None) or getattr(action, 'step_number', 0)

            return self.result_processor.build_enriched_info_block(
                execution_result,
                sexpr_result,
                mid_list,
                action_index,
                index,
                is_final_turn=is_final_turn,
            )
        else:
            # Check if this is a timeout error
            if self._is_timeout_error(execution_result.error_message):
                logger.error(f"[TIMEOUT] Sample {index}: SPARQL execution timeout detected - {execution_result.error_message}")
                # Return a special timeout error message that will be caught upstream
                # This will trigger the done=True and penalty reward logic
                return f"<information>\n⏰ SPARQL EXECUTION TIMEOUT\nError: {execution_result.error_message}\n\nThis query took too long to execute and was terminated. This usually indicates an overly complex or poorly constrained query.\n</information>"
            
            # Execution failed - rollback function state changes from this turn
            if initial_function_state_length is not None:
                current_function_state_length = len(self.state_manager.get_sample_function_state(index))
                functions_added_this_turn = current_function_state_length - initial_function_state_length
                if functions_added_this_turn > 0:
                    logger.info(f"[SEXPR] Sample {index}: Execution failed, rolling back {functions_added_this_turn} function calls")
                    self.state_manager.rollback_sample_function_state(index, functions_added_this_turn)
            
            return f"<information>\nExecution failed: {execution_result.error_message}\n</information>"

    def _execute_actions_independently(self, processed_actions: List[ActionResult], index: int, candidate_entities: List[Tuple[str, str]], turn: int = 0) -> str:
        """
        Execute each action independently and return individual results
        This is used when normal execution returns empty results
        """
        logger.info(f"[SEXPR] Sample {index}: Executing {len(processed_actions)} actions independently")
        
        is_final_turn = (turn == self.config.max_turns - 1)
        
        independent_results = []
        
        # Skip the last action since normal execution already confirmed no results there
        actions_to_run = processed_actions[:-1] if len(processed_actions) > 0 else []
        
        for i, action in enumerate(actions_to_run):
            # Execute single action
            single_result = self._execute_single_action(action, index, candidate_entities, action_index=i)
            independent_results.append(single_result)
        
        # Combine all independent results
        combined_info = "\n\n".join(independent_results)
        info_block = f"<information>\nIndependent execution results (normal execution returned empty results):\n\n{combined_info}\n</information>"
        return self.result_processor._add_guidance_prompt(info_block, is_final_turn)
    
    def _all_actions_are_find_relation(self, actions: List[ActionResult]) -> bool:
        """Check if all actions in the list are FIND_RELATION actions"""
        if not actions:
            return False
        return all(action.action_type == ActionType.FIND_RELATION for action in actions)
    
    def _handle_find_relation_actions_independently(self, actions: List[ActionResult], index: int, turn: int = 0) -> str:
        """
        Handle FIND_RELATION actions independently, processing both valid and invalid actions.
        Each action gets its own information block.
        """
        results = []
        
        for action_idx, action in enumerate(actions):
            # Action index should already be set by process_actions_with_retrieval
            # Only set it if it's missing (fallback case)
            if not hasattr(action, 'action_index'):
                action.action_index = action_idx
            if action.is_valid:
                # Process valid action normally
                result = self._execute_action_list_normal([action], index, None)
                results.append(result)

            else:
                # Handle invalid action (e.g., threshold not met with candidates)
                if hasattr(action, 'error_message') and action.error_message:
                    # Format the error message as an information block
                    info_result = f"<information>\nAction FIND_RELATION: {action.error_message}\n</information>"
                    results.append(info_result)
                else:
                    # Fallback for actions without detailed error messages (should rarely happen)
                    error_result = "<information>\nAction FIND_RELATION: No candidate relations available (entity information not available)\n</information>"
                    results.append(error_result)
        
        # Format all results together
        is_final_turn = (turn == self.config.max_turns - 1)
        return self.multi_result_formatter.format_multi_result(results, is_final_turn)




    def _execute_single_action(self, action: ActionResult, index: int, candidate_entities: List[Tuple[str, str]], action_index: int) -> str:
        """
        Execute a single action and return its result
        Reuses already processed action to avoid duplicate processing
        """
        logger.info(f"[SEXPR] Sample {index}: Executing single action {action_index+1}: {action.action_type.name}")
        
        # The action is already processed, so we can reuse it directly
        # Create a temporary function state for this single action
        temp_function_state = []
        
        # For single action execution, we need to handle different action types appropriately
        # if action.action_type == ActionType.EXTRACT_ENTITY:
        #     # For Extract_entity, we need to create a START function
        #     if action.arguments and len(action.arguments) > 0:
        #         entity_id = action.arguments[0]
        #         function_string = f"expression1 = START('{entity_id}')"
        #         temp_function_state.append(function_string)
        #         target_expr = "expression1"
        #     else:
        #         return f"Action {action_index+1} ({action.action_type.name}): No entity ID provided"
                
        if action.action_type == ActionType.FIND_RELATION:
            # For Find_relation, we need both entity and relation
            if len(action.arguments) == 2:
                entity_id, relation_id = action.arguments[0], action.arguments[1]
                # Create both START and JOIN functions
                temp_function_state.append(f"expression1 = START('{entity_id}')")
                temp_function_state.append(f"expression1 = JOIN('{relation_id}', expression1)")
                target_expr = "expression1"
            else:
                return f"Action {action_index+1} ({action.action_type.name}): Invalid arguments - expected [entity | relation]"
                
        elif action.action_type == ActionType.COMPARE:
            # For Compare, we need operator, relation, and number
            if len(action.arguments) == 3:
                mode, relation, number = action.arguments
                # Detect if the number argument is a mathematical calculation expression
                if SExprUtils.detect_math_expression(number):
                    return f"Action {action_index+1} ({action.action_type.name}): Invalid mathematical expression '{number}'. Please use the original number directly without any calculations (e.g., use '6.75' instead of '(6.75)^2')"
                
                # Create START function for the number literal (this is the correct approach)
                temp_function_state.append(f"expression1 = START('{number}')")
                # Create CMP function using the standardized mode
                from ..sexpr.constants import COMPARISON_MODE_MAPPING
                standardized_mode = COMPARISON_MODE_MAPPING.get(mode, mode.lower())
                temp_function_state.append(f"expression2 = CMP('{standardized_mode}', '{relation}', expression1)")
                target_expr = "expression2"
            else:
                return f"Action {action_index+1} ({action.action_type.name}): Invalid arguments - expected [operator | relation | number]"
                
        # elif action.action_type == ActionType.MERGE:
        #     # For Merge, we need to handle expression references
        #     if len(action.arguments) == 2:
        #         expr1, expr2 = action.arguments[0], action.arguments[1]
        #         # For independent execution of MERGE, we need to create the expressions that are being merged
        #         # This is a simplified approach - in practice, these expressions should come from previous actions
        #         # For now, we'll create basic expressions and then merge them
        #         temp_function_state.append("expression1 = START('m.02mjmr')")  # Create expression1
        #         temp_function_state.append("expression2 = START('m.02mjmr')")  # Create expression2
        #         temp_function_state.append(f"expression3 = AND({expr1}, {expr2})")  # Merge them
        #         target_expr = "expression3" #TODO
                
        #         # Add information about what was being merged
        #         merge_info = f"Note: MERGE operation attempted to combine {expr1} and {expr2}. For independent execution, these expressions were created with dummy entities."
        #     else:
        #         return f"Action {action_index+1} ({action.action_type.name}): Invalid arguments - expected [expression1 | expression2]"
                
        # else:
        #     # Reuse already processed action directly without re-processing
        #     function_calls = self.function_builder.build_function_sequence([action])
            
        #     if not function_calls:
        #         return f"Action {action_index+1} ({action.action_type.name}): Failed to build function sequence"
            
        #     # Generate S-Expression for single action
        #     sexpr_result = self.sexpr_generator.generate_sexpr_from_functions(function_calls)
            
        #     if not sexpr_result.is_valid:
        #         return f"Action {action_index+1} ({action.action_type.name}): S-Expression generation failed - {sexpr_result.error_message}"
            
        #     # Execute S-Expression
        #     execution_result = self.sexpr_executor.execute_sexpr(sexpr_result.sexpr)
            
        #     if execution_result.is_successful:
        #         mid_list = self.result_processor.extract_mid_list(execution_result.results)
                
        #         # Build result string for this action
        #         if len(mid_list) > 0:
        #             result_info = f"Action {action_index+1} ({action.action_type.name}): Found {len(mid_list)} results"
        #             if len(mid_list) <= 10:  # Show first 10 results
        #                 result_info += f" - {', '.join(mid_list[:10])}"
        #             else:
        #                 result_info += f" - {', '.join(mid_list[:10])}... (and {len(mid_list)-10} more)"
                    
        #             # Add S-Expression info
        #             result_info += f"\nS-Expression: {sexpr_result.sexpr}"
                    
        #             return result_info
        #         else:
        #             return f"Action {action_index+1} ({action.action_type.name}): No results found\nS-Expression: {sexpr_result.sexpr}"
        #     else:
        #         return f"Action {action_index+1} ({action.action_type.name}): Execution failed - {execution_result.error_message}\nS-Expression: {sexpr_result.sexpr}"
        
        # For the special handling cases (EXTRACT_ENTITY, FIND_RELATION, MERGE)
        if temp_function_state:
            # Generate S-Expression from function strings
            sexpr_result = self.sexpr_generator.generate_sexpr_from_strings(temp_function_state, target_expr)
            
            if not sexpr_result.is_valid:
                return f"Action {action_index+1} ({action.action_type.name}): S-Expression generation failed - {sexpr_result.error_message}"
            
            # Execute S-Expression
            execution_result = self.sexpr_executor.execute_sexpr(sexpr_result.sexpr, function_state=temp_function_state)
            
            if execution_result.is_successful:
                mid_list = self.result_processor.extract_mid_list(execution_result.results)
                
                # Build result string for this action
                if len(mid_list) > 0:
                    max_display = 30
                    result_info = f"Action {action_index+1} ({action.action_type.name}): Found {len(mid_list)} results"
                    if len(mid_list) <= max_display:  # Show first 10 results
                        result_info += f" - {', '.join(mid_list[:max_display])}"
                    else:
                        result_info += f" - {', '.join(mid_list[:max_display])}... (and {len(mid_list)-max_display} more)"
                    
                    # Add S-Expression info
                    result_info += f"\nS-Expression: {sexpr_result.sexpr}"
                    
                    # Add merge info if available
                    # if action.action_type == ActionType.MERGE and 'merge_info' in locals():
                    #     result_info += f"\n{merge_info}"
                    
                    return result_info
                else:
                    result_info = f"Action {action_index+1} ({action.action_type.name}): No results found\nS-Expression: {sexpr_result.sexpr}"
                    
                    # Add merge info if available
                    # if action.action_type == ActionType.MERGE and 'merge_info' in locals():
                    #     result_info += f"\n{merge_info}"
                    
                    return result_info
            else:
                result_info = f"Action {action_index+1} ({action.action_type.name}): Execution failed - {execution_result.error_message}\nS-Expression: {sexpr_result.sexpr}"
                
                # Add merge info if available
                # if action.action_type == ActionType.MERGE and 'merge_info' in locals():
                #     result_info += f"\n{merge_info}"
                
                return result_info
        
        return f"Action {action_index+1} ({action.action_type.name}): Unsupported action type for independent execution (supported: FIND_RELATION, COMPARE)"

    # Removed bespoke single FIND_RELATION info executor in favor of normal executor reuse


    def _process_sparql_prediction(self, prediction: str, index: int) -> str:
        """
        Process prediction using original SPARQL approach (fallback)
        """
        sparql_match = re.search(r'<sparql>(.*?)</sparql>', prediction, re.DOTALL)
        if not sparql_match:
            return "<information>\nNo SPARQL query found.\n</information>"
        
        sparql_query = sparql_match.group(1).strip()
        
        sparql_results = self.sparql_manager.execute_batch([sparql_query])
        
        if "results" in sparql_results and sparql_results["results"]:
            result = sparql_results["results"][0]
            
            if isinstance(result, dict) and "error" in result:
                return f"<information>\nSPARQL Error: {result['error']}\n</information>"
            
            if isinstance(result, dict) and "results" in result:
                result_string = SPARQLExecutionManager.results_to_string(result['results'])
            else:
                result_string = SPARQLExecutionManager.results_to_string(result)
            
            truncated_result = self.result_processor.truncate_result(result_string)
            return f"<information>\n{truncated_result}\n</information>"
        else:
            return "<information>\nNo results found.\n</information>"
            


    # Include all other methods from original LLMGenerationManager
    # (same implementations for state management, GPU padding, etc.)
    
    



    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor):
        """
        Main LLM loop with S-Expression support
        Enhanced with action parsing and S-Expression execution
        """
        # Increment call counter
        SExprLLMGenerationManager._call_counter += 1
        
        # Reset per-call candidate length stats (for tensorboard)
        if self.action_processor:
            self.action_processor.reset_candidate_stats()
        
        dialogue_data = []
        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        # 初始化所有样本的持久化状态
        if self.config.enable_sexpr_mode:
            self.state_manager.initialize_batch_states(batch_size)
            # 解析并缓存每个样本在提示中的候选实体，供后续动作校验使用
            SExprUtils.initialize_candidate_entities_from_prompts(self.tokenizer, self.state_manager, initial_input_ids, batch_size)

        
        # Enhanced logging for testing (controlled frequency)
        logger.info(f"[SEXPR] Starting LLM loop #{self._call_counter} | Mode: {'S-Expression' if self.config.enable_sexpr_mode else 'SPARQL'} | Batch size: {batch_size}")
        if self.config.enable_sexpr_mode:
            logger.info(f"[SEXPR] Components: ActionParser={self.action_parser is not None}, "
                        f"SExprGenerator={self.sexpr_generator is not None}, "
                        f"SExprValidator={self.sexpr_validator is not None}, "
                        f"RelationRetrieval={self.relation_retrieval is not None}")
        
        logger.debug(f"Starting S-Expression LLM loop (call #{self._call_counter}), mode: {'sexpr' if self.config.enable_sexpr_mode else 'sparql'}")
        
        # Ensure batch size consistency
        if initial_input_ids.shape[0] != batch_size:
            logger.warning(f"Batch size mismatch: {initial_input_ids.shape[0]} != {batch_size}")
            if initial_input_ids.shape[0] > batch_size:
                initial_input_ids = initial_input_ids[:batch_size]
            else:
                pad_count = batch_size - initial_input_ids.shape[0]
                padding = initial_input_ids[0:1].repeat(pad_count, 1)
                initial_input_ids = torch.cat([initial_input_ids, padding], dim=0)
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        meta_info = {
            'done': torch.zeros(batch_size, dtype=torch.bool),
            'turn': 0,
        }

        # OPD: accumulate raw observation strings per turn for hindsight teacher prompt
        per_turn_observations = [[] for _ in range(batch_size)]

        # Main generation loop
        for turn in range(self.config.max_turns):
            meta_info['turn'] = turn
            
            if not active_mask.sum():
                # 即使提前结束，也要记录最后一轮的截断信息
                # 直接基于当前rollings计算截断信息，避免使用空tensor
                self.batch_utils.record_final_truncation_info(rollings, meta_info, batch_size)
                break
                
            logger.info(f"Turn {turn}: active samples: {active_mask.sum().item()}/{batch_size}")
            
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict(tensors={
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            # Generate responses
            gen_output = self.batch_utils.generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            
            # CRITICAL FIX: Extract rollout_log_probs from gen_output for this turn
            # rollout_log_probs come from vLLM generate_sequences() when calculate_log_probs=true
            # We need to expand them to full batch size (pad inactive samples with zeros)
            cur_rollout_log_probs = None
            if 'rollout_log_probs' in gen_output.batch:
                # Expand to full batch size (pad inactive samples with zeros)
                full_batch_rollout_log_probs = torch.zeros(
                    batch_size, gen_output.batch['rollout_log_probs'].shape[1],
                    dtype=gen_output.batch['rollout_log_probs'].dtype,
                    device=gen_output.batch['rollout_log_probs'].device
                )
                full_batch_rollout_log_probs[active_mask] = gen_output.batch['rollout_log_probs']
                cur_rollout_log_probs = full_batch_rollout_log_probs
                
                logger.info(f"[MISMATCH FIX] Turn {turn}: Extracted rollout_log_probs from gen_output, "
                           f"active shape: {gen_output.batch['rollout_log_probs'].shape}, "
                           f"expanded to full batch: {cur_rollout_log_probs.shape}")
            
            if 'responses' not in gen_output.batch or gen_output.batch['responses'].shape[0] == 0:
                logger.warning("Invalid response from generation")
                active_size = active_mask.sum().item()
                dummy_responses = torch.ones((active_size, 1), dtype=torch.long) * self.tokenizer.pad_token_id
                gen_output.batch['responses'] = dummy_responses
                
            # Process responses
            responses_ids, responses_str = SExprUtils.postprocess_responses(self.tokenizer, gen_output.batch['responses'], self.config.no_think_rl)
            if turn==1 and "" in responses_str and SExprLLMGenerationManager._debug_save_counter < 10:
                # 保存空响应的详细信息到文件（总共只保存10次）
                import json
                import os
                from datetime import datetime

                # 增加保存计数器
                SExprLLMGenerationManager._debug_save_counter += 1
                
                # 找到空字符串的索引
                empty_indices = [i for i, resp in enumerate(responses_str) if resp == ""]
                
                debug_info = {
                    "timestamp": datetime.now().isoformat(),
                    "turn": turn,
                    "call_counter": self._call_counter,
                    "debug_save_number": SExprLLMGenerationManager._debug_save_counter,
                    "empty_response_indices": empty_indices,
                    "total_responses": len(responses_str),
                    "active_mask": active_mask.tolist(),
                    "batch_size": batch_size,
                    "num_gpus": self.config.num_gpus,
                    "remainder": rollings_active.batch['input_ids'].shape[0] % self.config.num_gpus
                }
                
                # 为每个空响应收集详细信息（只保存前3个）
                empty_samples = []
                for idx in empty_indices[:3]:  # 每次只保存前3个空响应
                    if idx < len(rollings_active.batch['input_ids']):
                        sample_info = {
                            "sample_index": idx,
                            "input_ids": rollings_active.batch['input_ids'][idx].tolist(),
                            "input_text": self.tokenizer.decode(rollings_active.batch['input_ids'][idx], skip_special_tokens=False),
                            "attention_mask": rollings_active.batch['attention_mask'][idx].tolist(),
                            "position_ids": rollings_active.batch['position_ids'][idx].tolist(),
                            "raw_gen_output": gen_output.batch['responses'][idx].tolist() if 'responses' in gen_output.batch and idx < gen_output.batch['responses'].shape[0] else None,
                            "raw_gen_output_text": self.tokenizer.decode(gen_output.batch['responses'][idx], skip_special_tokens=False) if 'responses' in gen_output.batch and idx < gen_output.batch['responses'].shape[0] else None,
                            "postprocessed_response": responses_str[idx],
                            "postprocessed_ids": responses_ids[idx].tolist() if idx < len(responses_ids) else None
                        }
                        empty_samples.append(sample_info)
                
                debug_info["empty_samples"] = empty_samples
                
                # 根据用户需求关闭文件调试保存（仅当显式启用才写盘）
                if self.config.enable_logging and os.getenv("ENABLE_SEXPR_FILE_LOGS", "0").lower() in ("1","true","yes"):
                    debug_dir = "debug_logs"
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_file = os.path.join(debug_dir, f"empty_response_debug_{SExprLLMGenerationManager._debug_save_counter:02d}.json")
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        json.dump(debug_info, f, indent=2, ensure_ascii=False)
                    logger.warning(f"[{SExprLLMGenerationManager._debug_save_counter}/10] Found {len(empty_indices)} empty responses in turn 1. Saved {len(empty_samples)} samples to: {debug_file}")
                    print(f"[DEBUG {SExprLLMGenerationManager._debug_save_counter}/10] Found {len(empty_indices)} empty responses in turn 1. Saved {len(empty_samples)} samples to: {debug_file}")
                else:
                    logger.warning(f"[NO-FILE-LOG] Found {len(empty_indices)} empty responses in turn 1 (file logging disabled)")
                    print(f"[NO-FILE-LOG] Found {len(empty_indices)} empty responses in turn 1 (file logging disabled)")
            elif turn==1 and "" in responses_str:
                # 超过10次保存限制，只记录日志
                empty_count = len([i for i, resp in enumerate(responses_str) if resp == ""])
                logger.warning(f"[SKIP] Found {empty_count} empty responses in turn 1, but debug save limit (10) reached")
                print(f"[DEBUG SKIP] Found {empty_count} empty responses in turn 1, but debug save limit (10) reached")

            
            if len(responses_str) == 0 or all(not s for s in responses_str):
                logger.warning("Empty responses after postprocessing")
                active_mask = torch.zeros_like(active_mask)
                meta_info['done'] = ~active_mask
                continue
            
            # Apply padding to match batch size
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Execute predictions (S-Expression or SPARQL)
            next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask, turn=turn)

            # OPD: record raw observation strings for hindsight teacher prompt
            for i in range(batch_size):
                per_turn_observations[i].append(next_obs[i] if active_mask[i] else "")

            # Process observations
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update active mask
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            
            # Update states
            is_final_turn = (turn == self.config.max_turns - 1)
            rollings = self.batch_utils.update_rolling_state(rollings, responses_ids, next_obs_ids, is_final_turn)
            # CRITICAL FIX: Pass cur_rollout_log_probs to update_right_side
            original_right_side = self.batch_utils.update_right_side(
                original_right_side, 
                responses_ids, 
                cur_rollout_log_probs=cur_rollout_log_probs,
                next_obs_ids=next_obs_ids
            )
            
            # 只记录最后一轮的截断统计信息 (优化版本)
            if is_final_turn and hasattr(rollings, 'meta_info') and rollings.meta_info:
                truncation_info = rollings.meta_info
                
                # 只记录最后一轮的截断情况 (因为如果之前被截断，最后一轮肯定也被截断)
                meta_info['turn_final/truncation/clip_ratio'] = truncation_info.get('batch_truncation_ratio', 0.0)
                meta_info['turn_final/truncation/mean_effective_length'] = truncation_info.get('batch_mean_effective_length', 0.0)
                meta_info['turn_final/truncation/mean_truncation_ratio'] = truncation_info.get('batch_mean_truncation_ratio', 0.0)
                meta_info['turn_final/truncation/max_effective_length'] = truncation_info.get('max_effective_length', 0.0)
                
                # 添加与原有指标的对比
                meta_info['turn_final/truncation/max_prompt_length'] = truncation_info.get('max_prompt_length', 0)
                meta_info['turn_final/truncation/truncated_length'] = truncation_info.get('truncated_length', 0)
            
            # Collect dialogue data
            if self.config.enable_logging:
                for i, (resp_str, obs_str) in enumerate(zip(responses_str, next_obs)):
                    if i >= len(dialogue_data):
                        # Extract question and ID from gen_batch extra_info
                        question = ""
                        sample_id = i
                        if hasattr(gen_batch, 'non_tensor_batch') and gen_batch.non_tensor_batch:
                            extra_info = gen_batch.non_tensor_batch.get('extra_info', [])
                            if i < len(extra_info) and extra_info[i]:
                                question = extra_info[i].get('original_question', '')
                                sample_id = extra_info[i].get('id', i)
                        
                        dialogue_data.append({
                            "sample_id": sample_id,
                            "question": question,
                            "turns": []
                        })
                    
                    dialogue_data[i]["turns"].append({
                        "turn": turn,
                        "raw_response": resp_str,
                        "raw_observation": obs_str,
                        "mode": "sexpr" if self.config.enable_sexpr_mode else "sparql"
                    })
            
            meta_info['done'] = ~active_mask

        # 确保在循环结束后记录截断信息（如果还没有记录的话）
        if not any(k.startswith('turn_final/truncation/') for k in meta_info.keys()):
            # 如果循环正常结束但没有记录截断信息，现在记录
            # 直接基于当前rollings计算截断信息，避免使用空tensor
            self.batch_utils.record_final_truncation_info(rollings, meta_info, batch_size)

        # Final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self.batch_utils.generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = SExprUtils.postprocess_responses(self.tokenizer, gen_output.batch['responses'], self.config.no_think_rl)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # CRITICAL FIX: Extract rollout_log_probs for final turn (same as in main loop)
            cur_rollout_log_probs = None
            if 'rollout_log_probs' in gen_output.batch:
                # Expand to full batch size (pad inactive samples with zeros)
                full_batch_rollout_log_probs = torch.zeros(
                    batch_size,
                    gen_output.batch['rollout_log_probs'].shape[1],
                    dtype=gen_output.batch['rollout_log_probs'].dtype,
                    device=gen_output.batch['rollout_log_probs'].device
                )
                full_batch_rollout_log_probs[active_mask] = gen_output.batch['rollout_log_probs']
                cur_rollout_log_probs = full_batch_rollout_log_probs
                logger.info(f"[MISMATCH FIX] Final turn: Extracted rollout_log_probs, active shape: {gen_output.batch['rollout_log_probs'].shape}, expanded: {cur_rollout_log_probs.shape}")

            # Execute final predictions
            _, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask, do_execution=False, turn=turn)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())

            # CRITICAL FIX: Pass cur_rollout_log_probs to update_right_side (final turn)
            original_right_side = self.batch_utils.update_right_side(
                original_right_side,
                responses_ids,
                cur_rollout_log_probs=cur_rollout_log_probs
            )
        
            # Collect final turn dialogue data
            if self.config.enable_logging:
                for i, resp_str in enumerate(responses_str):
                    if i < len(dialogue_data):
                        dialogue_data[i]["turns"].append({
                            "turn": "final",
                            "raw_response": resp_str,
                            "raw_observation": "",
                            "mode": "sexpr" if self.config.enable_sexpr_mode else "sparql"
                        })
        
        logger.info(f"Active trajectory numbers: {active_num_list}")
        
        # Log dialogue data (deterministic: pass full batch to logger which will prioritize)
        if self.logging_manager.should_log(self._call_counter) and dialogue_data:
            all_indices = list(range(len(dialogue_data)))
            self.logging_manager.save_dialogue_log(dialogue_data, all_indices, self._call_counter)
            logger.info(f"Logged {len(dialogue_data)} S-Expression dialogues (call #{self._call_counter})")

        # Persist stats for threshold-not-met candidate counts
        if self.action_processor:
            candidate_stats = self.action_processor.get_candidate_stats()
            self.logging_manager.save_threshold_not_met_stats(
                candidate_stats['relation_threshold_not_met_counts'], 
                self._call_counter
            )
            # Save per-step relation similarity distributions (TOP-1 scores only)
            if 'per_step_relation_similarity' in candidate_stats:
                self.logging_manager.save_relation_similarity_distributions(
                    candidate_stats['per_step_relation_similarity'],
                    self._call_counter
                )
                # Expose histogram-friendly entries for tensorboard (distribution per step)
                # Also compute and expose mean values for trend monitoring
                all_top1_scores = []
                for step_key, scores in candidate_stats['per_step_relation_similarity'].items():
                    sk = int(step_key)
                    if scores:
                        # Histogram: distribution of top-1 scores for this step
                        meta_info[f'hist/sexpr/relation_similarity_top1/step_{sk}'] = list(scores)
                        # Mean: average top-1 score for this step
                        step_mean = sum(scores) / len(scores)
                        meta_info[f'sexpr/relation_similarity_top1/step_{sk}_mean'] = float(step_mean)
                        all_top1_scores.extend(scores)
                
                # Overall statistics across all steps
                if all_top1_scores:
                    meta_info['sexpr/relation_similarity_top1/overall_mean'] = float(sum(all_top1_scores) / len(all_top1_scores))
                    meta_info['sexpr/relation_similarity_top1/overall_min'] = float(min(all_top1_scores))
                    meta_info['sexpr/relation_similarity_top1/overall_max'] = float(max(all_top1_scores))
                    meta_info['hist/sexpr/relation_similarity_top1/all_steps'] = list(all_top1_scores)
                    # Debug log to verify metrics are being added
                    logger.info(f"[RELATION_SIMILARITY_TOP1] Added {len(all_top1_scores)} scores to meta_info: "
                               f"overall_mean={meta_info['sexpr/relation_similarity_top1/overall_mean']:.4f}, "
                               f"min={meta_info['sexpr/relation_similarity_top1/overall_min']:.4f}, "
                               f"max={meta_info['sexpr/relation_similarity_top1/overall_max']:.4f}")

            self.action_processor.clear_threshold_not_met_counts()
        
        # Attach tensorboard-friendly metrics into meta_info
        if self.action_processor:
            candidate_stats = self.action_processor.get_candidate_stats()
            overall_mean = (candidate_stats['cand_all_sum'] / candidate_stats['cand_all_count']) if candidate_stats['cand_all_count'] > 0 else 0.0
            notmet_mean = (candidate_stats['cand_notmet_sum'] / candidate_stats['cand_notmet_count']) if candidate_stats['cand_notmet_count'] > 0 else 0.0
            meta_info['sexpr/candidate_rel_len/overall_mean'] = float(overall_mean)
            meta_info['sexpr/candidate_rel_len/threshold_not_met_mean'] = float(notmet_mean)
            meta_info['sexpr/candidate_rel_len/threshold_not_met_ratio'] = float(candidate_stats['cand_notmet_count'] / candidate_stats['cand_all_count']) if candidate_stats['cand_all_count'] > 0 else 0.0
            
            # also attach special action metrics
            action_metrics = self.action_processor.get_action_metrics()
            for name, counters in action_metrics.items():
                attempts = counters.get('attempts', 0)
                successes = counters.get('successes', 0)
                rate = (successes / attempts) if attempts > 0 else 0.0
                meta_info[f'sexpr/actions/{name}/attempts'] = float(attempts)
                meta_info[f'sexpr/actions/{name}/successes'] = float(successes)
                meta_info[f'sexpr/actions/{name}/success_rate'] = float(rate)

        # OPD: store per-turn raw observation strings for hindsight teacher prompt
        meta_info['per_turn_observations'] = per_turn_observations

        return self.batch_utils.compose_final_output(original_left_side, original_right_side, meta_info)
