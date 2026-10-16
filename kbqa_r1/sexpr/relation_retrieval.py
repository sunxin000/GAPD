"""
Relation Retrieval Module for S-Expression based KBQA-R1
Integrates SimCSE model with dynamic relation retrieval and execution validation
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..sparql.sparql_manager import SPARQLConfig
from .constants import COMPARISON_MODE_READABLE_MAPPING
from .dynamic_relation_retrieval import DynamicRelationRetrieval
from .execution_validator import ExecutionValidator
from .simcse_tool import SimCSE, get_default_simcse_model

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Import enhanced components
SIMCSE_AVAILABLE = True


@dataclass
class RelationCandidate:
    """Represents a candidate relation with similarity score"""
    relation_name: str
    relation_id: str
    score: float
    is_reverse: bool = False

class RelationRetrieval:
    """
    Integrates KBQA-o1's relation retrieval mechanism with SimCSE similarity model
    Combines dynamic relation querying with similarity-based selection
    """
    
    def __init__(self, similarity_model=None, relation_config: dict = None, 
                 sparql_config: SPARQLConfig = None, dataset: str = "WebQSP"):
        """Initialize relation retrieval and all SimCSE backends.

        We intentionally allocate **separate** SimCSE instances for each index to
        avoid index pollution on a shared model instance (SimCSE.index is mutable).

        Instances:
        - simcse_rel:        global relation_list index
        - simcse_lit:        literal CMP index ("<mode> | relation")
        - simcse_name:       name_relation_list plain index
        - simcse_literal_plain: literal_relation_list plain index (non-CMP)
        """

        # Core SimCSE backends
        self.simcse_rel = None
        self.simcse_lit = None
        self.simcse_name = None
        self.simcse_literal_plain = None

        if SIMCSE_AVAILABLE:
            if similarity_model is None:
                # No external model provided: load four independent default instances
                self.simcse_rel = get_default_simcse_model()
                self.simcse_lit = get_default_simcse_model()
                self.simcse_name = get_default_simcse_model()
                self.simcse_literal_plain = get_default_simcse_model()
                logger.info("Successfully loaded default SimCSE models for relations, literals, name and literal_plain")

        #     else:
        #         # Use provided instance for the **global relation** index only.
        #         # All other indices get their own SimCSE instances to avoid index overwrite.
        #         self.simcse_rel = similarity_model
        #         model_name_or_path = getattr(getattr(similarity_model, 'model', None), 'config', None)
        #         model_name_or_path = getattr(model_name_or_path, '_name_or_path', None)
        #         if model_name_or_path:
        #             self.simcse_lit = SimCSE(model_name_or_path)
        #             self.simcse_name = SimCSE(model_name_or_path)
        #             self.simcse_literal_plain = SimCSE(model_name_or_path)
        #         else:
        #             # Fallback to default if we cannot infer the model path
        #             self.simcse_lit = get_default_simcse_model()
        #             self.simcse_name = get_default_simcse_model()
        #             self.simcse_literal_plain = get_default_simcse_model()

        # else:
        #     logger.warning("SimCSE unavailable; retrieval will use simple similarity only")
        
        self.config = relation_config or self._default_config()
        self.dataset = dataset
        
        # Initialize enhanced components
        self.execution_validator = ExecutionValidator(sparql_config, dataset)
        self.dynamic_retrieval = DynamicRelationRetrieval(sparql_config, dataset)
        
        # Load predefined relation lists (similar to KBQA-o1's limit.py)
        self._load_relation_lists()
        
        # Build relation index for efficient retrieval
        self._build_relation_index()
        
        # Build literal relation index for efficient COMPARE operations
        self._build_literal_relation_index()

        # Build additional indices with their dedicated SimCSE instances:
        # 1) name_relation_list 专用 index（裸关系，无比较前缀）
        # 2) literal_relation_list 的 plain index（裸关系，供非 CMP 场景使用）
        self._build_name_relation_index()
        self._build_literal_plain_index()

        # Store the most recent raw similarity scores for external inspection/logging
        self.last_similarity_scores = []  # type: ignore[var-annotated]
        # Track relations filtered by relation_list in the most recent dynamic retrieval
        self.last_filtered_by_relation_list = []  # type: ignore[var-annotated]
        # File path for recording relations filtered by relation_list (whitelist)
        self.filtered_relations_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'logs', 'filtered_relations.log')
        os.makedirs(os.path.dirname(self.filtered_relations_log_path), exist_ok=True)

    
    @property
    def sparql_manager(self):
        """Expose underlying SPARQL manager for external utilities (read-only)."""
        return getattr(self.dynamic_retrieval, 'sparql_manager', None)

    
    def _default_config(self) -> dict:
        """Default configuration for relation retrieval"""
        return {
            'relation_topk': 5,
            'relation_threshold': 0.3,
            'entity_topk': 1,
            'entity_threshold': 0.5,
            'cmp_relation_threshold': 0.3,  # Threshold for CMP operation relation selection
        }
    
    def _load_relation_lists(self):
        """Load predefined relation lists from local limit.py file"""
        # Import from local limit.py file
        from .limit import (join_ban_relation_list, literal_relation_list,
                            name_relation_list, relation_list, tc_time_list)

        # Load all relation lists from local limit.py
        self.name_relation_list = set(name_relation_list)
        self.tc_time_list = set(tc_time_list)
        self.join_ban_relation_list = set(join_ban_relation_list)
        self.literal_relation_list = set(literal_relation_list)
        self.relation_list = set(relation_list)
        
        logger.info("Successfully loaded relation lists from local limit.py")
        logger.info(f"  - relation_list: {len(self.relation_list)} relations")
        logger.info(f"  - join_ban_relation_list: {len(self.join_ban_relation_list)} relations")
        logger.info(f"  - literal_relation_list: {len(self.literal_relation_list)} relations")
        logger.info(f"  - name_relation_list: {len(self.name_relation_list)} relations")
        logger.info(f"  - tc_time_list: {len(self.tc_time_list)} time values")

    def _build_relation_index(self):
        """Build SimCSE index for efficient relation retrieval"""
        # 如果已经有 index，则不重复构建，避免在训练过程中每个 step 都重新加载 / build faiss
        if getattr(self, 'indexed_relations', None):
            logger.info("Relation index already built; skip rebuilding.")
            return

        if self.simcse_rel and hasattr(self.simcse_rel, 'build_index'):
            # Convert relation set to list for indexing
            relation_names = list(self.relation_list)
            
            if relation_names:
                logger.info(f"Building/Loading SimCSE index for {len(relation_names)} relations...")
                cache_dir = self._default_cache_dir()
                cache_key = f"relations-{self.dataset.lower()}"
                self.simcse_rel.build_index(
                    relation_names,
                    use_faiss=True,
                    cache_dir=cache_dir,
                    cache_key=cache_key
                )
                logger.info("Successfully built relation index")
                
                # Store relation mapping for retrieval
                self.indexed_relations = relation_names
            else:
                logger.warning("No relations found to index")
                self.indexed_relations = []
        else:
            logger.info("SimCSE model does not support indexing, will use direct similarity")
            self.indexed_relations = []
    
    def _build_literal_relation_index(self):
        """Build SimCSE index for literal relations used in COMPARE operations"""
        if getattr(self, 'indexed_literal_relations', None):
            logger.info("Literal CMP index already built; skip rebuilding.")
            return

        if self.simcse_lit and hasattr(self.simcse_lit, 'build_index'):
            # Convert literal relation set to list for indexing
            literal_relation_names = list(self.literal_relation_list)
            
            if literal_relation_names:
                logger.info(f"Building/Loading SimCSE index for {len(literal_relation_names)} literal relations...")
                
                # Create candidate strings with different comparison modes for indexing
                # This matches the format used in select_best_relation_for_cmp
                indexed_strings = []
                for mode, readable_mode in COMPARISON_MODE_READABLE_MAPPING.items():
                    for relation in literal_relation_names:
                        indexed_strings.append(f"{readable_mode} | {relation}")
                
                logger.info(f"Indexing/Loading {len(indexed_strings)} literal relation strings...")
                cache_dir = self._default_cache_dir()
                cache_key = f"literal-relations-{self.dataset.lower()}"
                self.simcse_lit.build_index(
                    indexed_strings,
                    use_faiss=True,
                    cache_dir=cache_dir,
                    cache_key=cache_key
                )
                logger.info("Successfully built literal relation index")
                
                # Store literal relation mapping for retrieval
                self.indexed_literal_relations = literal_relation_names
            else:
                logger.warning("No literal relations found to index")
                self.indexed_literal_relations = []
        else:
            logger.info("SimCSE model does not support indexing for literal relations, will use direct similarity")
            self.indexed_literal_relations = []

    def _build_name_relation_index(self):
        """Build SimCSE index for name_relation_list (plain relations, no comparison prefix).

        This index is intended for name-typed find_relation / order 场景，
        避免受全局 relation_list vocab 和 CMP literal_index 的干扰。
        """
        # 如果已经有 index，则不重复构建
        if getattr(self, 'indexed_name_relations', None):
            logger.info("Name relation index already built; skip rebuilding.")
            return

        # 使用专用的 simcse_name 实例，避免与其它 index 互相污染
        backend = getattr(self, "simcse_name", None)
        if backend and hasattr(backend, 'build_index'):
            name_relations = list(self.name_relation_list or [])
            if not name_relations:
                logger.warning("No name relations found to index")
                self.indexed_name_relations = []
                return

            logger.info(f"Building/Loading SimCSE index for {len(name_relations)} name relations...")
            cache_dir = self._default_cache_dir()
            cache_key = f"name-relations-{self.dataset.lower()}"
            try:
                backend.build_index(
                    name_relations,
                    use_faiss=True,
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                )
                self.indexed_name_relations = name_relations
                logger.info("Successfully built name relation index")
            except TypeError:
                # 兼容老版 SimCSE 接口（不支持 cache_dir / cache_key）。
                logger.info("SimCSE.build_index does not support cache parameters; building in-memory name index only")
                backend.build_index(name_relations, use_faiss=True)
                self.indexed_name_relations = name_relations
        else:
            logger.info("SimCSE model does not support indexing for name relations")
            self.indexed_name_relations = []

    def _build_literal_plain_index(self):
        """Build SimCSE index for literal_relation_list without comparison operator prefix.

        注意：这个 index 与用于 CMP 的 literal index 不同，后者在文本前拼接了
        "<比较模式> | "，只能用于 Compare；本函数构建的 plain index 只包含裸关系，
        供 Order / 非比较场景使用。
        """
        if getattr(self, 'indexed_literal_plain_relations', None):
            logger.info("Plain literal relation index already built; skip rebuilding.")
            return

        # 使用专用的 simcse_literal_plain 实例
        backend = getattr(self, "simcse_literal_plain", None)
        if backend and hasattr(backend, 'build_index'):
            literal_plain = list(self.literal_relation_list or [])
            if not literal_plain:
                logger.warning("No plain literal relations found to index")
                self.indexed_literal_plain_relations = []
                return

            logger.info(f"Building/Loading SimCSE index for {len(literal_plain)} plain literal relations...")
            cache_dir = self._default_cache_dir()
            cache_key = f"literal-plain-relations-{self.dataset.lower()}"
            try:
                backend.build_index(
                    literal_plain,
                    use_faiss=True,
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                )
                self.indexed_literal_plain_relations = literal_plain
                logger.info("Successfully built plain literal relation index")
            except TypeError:
                logger.info("SimCSE.build_index does not support cache parameters; building in-memory plain literal index only")
                backend.build_index(literal_plain, use_faiss=True)
                self.indexed_literal_plain_relations = literal_plain
        else:
            logger.info("SimCSE model does not support indexing for plain literal relations")
            self.indexed_literal_plain_relations = []

    def _default_cache_dir(self) -> str:
        """Default directory for SimCSE FAISS cache files."""
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'cache', 'simcse')
        os.makedirs(base, exist_ok=True)
        return base
    
    def get_candidate_entities(self) -> List[Tuple[str, str]]:
        """Get candidate entities (placeholder implementation)"""
        # This should be implemented to get actual candidate entities
        return []
    
    def select_best_relation_for_cmp(self, gen_argument: str, mode: str) -> List[RelationCandidate]:
        """
        Select best relation for CMP operation using literal_relation_list
        Uses pre-built index for efficient similarity search
        
        Args:
            gen_argument: LLM generated relation description
            mode: Comparison mode (e.g., 'LESS EQUAL', 'le', etc.)
            
        Returns:
            A RelationCandidate for the best matching relation, or None if no relation meets the threshold.
        """
        # if not self.literal_relation_list or (not self.simcse_lit and not self.simcse_rel):
        #     return gen_argument  # Fallback to original argument
        
        # Convert mode to human readable format for similarity matching
        readable_mode = COMPARISON_MODE_READABLE_MAPPING.get(mode, mode)
        
        # Create search query combining mode and relation
        search_query = f"{readable_mode} | {gen_argument}"

        # Use a deterministic ordering of literal relations for downstream consumers
        literal_relation_names = list(self.literal_relation_list)
        
        # Check if we have a pre-built index for literal relations
        # if (self.simcse_lit and hasattr(self.simcse_lit, 'index') and 
            # getattr(self.simcse_lit, 'index', None) and 
            # hasattr(self.simcse_lit, 'is_faiss_index') and 
            # self.simcse_lit.is_faiss_index):
        
        # Use efficient index-based search
        logger.info(f"Using pre-built index for CMP relation search: {gen_argument}")

        # Create candidate strings for the index (should match what was indexed)
        candidate_strings = [f"{readable_mode} | {relation}" for relation in literal_relation_names]
        
        # Search using the index
        threshold = self.config.get('cmp_relation_threshold', 0.5)
        search_results = self.simcse_lit.search(search_query, threshold=threshold, top_k=len(candidate_strings))
        
        if search_results:
            # Extract scores and find best match, ensuring alignment with literal_relation_list order
            best_score = 0.0
            best_relation = None
            score_map: Dict[str, float] = {}

            mode_prefix = f"{readable_mode} | "

            for result_text, score in search_results:
                text = result_text or ""
                rel_name = text
                if text.startswith(mode_prefix):
                    rel_name = text[len(mode_prefix):]
                elif " | " in text:
                    rel_name = text.split(" | ", 1)[1]
                rel_name = rel_name.strip()
                if not rel_name:
                    continue

                score_value = float(score)
                prev = score_map.get(rel_name)
                if prev is None or score_value > prev:
                    score_map[rel_name] = score_value
                if score_value > best_score:
                    best_score = score_value
                    best_relation = rel_name

            scores_list = [score_map.get(rel, 0.0) for rel in literal_relation_names]
            self.last_similarity_scores = scores_list
            
            if best_relation and best_score >= threshold:
                logger.info(f"Selected relation for CMP (index-based): {best_relation} (score: {best_score:.3f}) for query: {gen_argument}")
            # Return a RelationCandidate object to be consistent with select_best_relations
                candidate = RelationCandidate(
                    relation_name=best_relation,
                    relation_id=best_relation,
                    score=float(best_score),
                    is_reverse=False
                )
                return candidate

            else:
                logger.warning(f"No suitable literal relation found for '{gen_argument}' (best score: {best_score:.3f})")
                return None
        else:
            logger.warning(f"No search results found for '{gen_argument}' using index")
            return None
        # else:
        #     # Fallback to direct similarity computation (original method)
        #     logger.info(f"Using direct similarity computation for CMP relation search: {gen_argument}")
            
        #     # Create candidate strings for similarity matching
        #     candidate_strings = [f"{readable_mode} | {relation}" for relation in self.literal_relation_list]
            
        #     # Prefer literal simcse if available; else fallback to relation simcse
        #     backend = self.simcse_lit or self.simcse_rel
        #     scores = backend.similarity(search_query, candidate_strings)
        #     if hasattr(scores, '__len__'):
        #         scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        #         best_idx = scores.argmax()
        #         best_score = scores[best_idx]
        #     else:
        #         scores_list = [scores]
        #         best_idx = 0
        #         best_score = scores
            
        #     # Store similarity scores for UI ranking (similar to select_best_relations)
        #     self.last_similarity_scores = scores_list
            
        #     # Use a lower threshold for CMP operations since they're more specific
        #     threshold = self.config.get('cmp_relation_threshold', 0.5)
            
        #     if best_score >= threshold:
        #         best_relation = list(self.literal_relation_list)[best_idx]
        #         logger.info(f"Selected relation for CMP (direct): {best_relation} (score: {best_score:.3f}) for query: {gen_argument}")
        #         return best_relation
        #     else:
        #         logger.warning(f"No suitable literal relation found for '{gen_argument}' (best score: {best_score:.3f})")
        #         return None
            

    
    def select_best_entity(self, gen_argument: str, candidate_entities: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
        """
        Select best entity using similarity
        
        Args:
            gen_argument: LLM generated entity description
            candidate_entities: List of (entity_name, entity_id) tuples
            
        Returns:
            Best matching (entity_name, entity_id) or None
        """
        if not candidate_entities or not self.similarity_model:
            return None
        
        entity_names = [name for name, _ in candidate_entities]
        
        scores = self.similarity_model.similarity(gen_argument, entity_names)
        if hasattr(scores, '__len__'):
            best_idx = scores.argmax()
            best_score = scores[best_idx]
        else:
            best_idx = 0
            best_score = scores
        
        if best_score >= self.config['entity_threshold']:
            return candidate_entities[best_idx]
        
        return None
    
    def get_candidate_relations(self, function_state: List[str], entity_type: str = None,
                                allow_literal_relations: bool = False) -> List[Tuple[str, str]]:
        """
        Get candidate relations based on current function state
        Combines dynamic querying with predefined lists
        
        Args:
            function_state: Current function sequence
            entity_type: Optional entity type hint
            
        Returns:
            List of (relation_name, relation_id) tuples
        """
        if not function_state:
            # Return basic relations for initial state
            raise ValueError("Function state is empty")
        
        # Conditional debug breakpoint BEFORE dynamic calls, to step into get_next_relations/get_next_r_relations
        # Trigger when KBQA_DEBUG_BP=1 and the last step is not a START (expression context)
        # try:
        #     import os
        #     last_stmt = function_state[-1] if function_state else ''
        #     is_expression_ctx = ('START' not in last_stmt)
        #     if is_expression_ctx and os.getenv('KBQA_DEBUG_BP') == '1':
        #         logger.warning("[DEBUG] Breakpoint before dynamic retrieval (expression context)")
        #         tail = function_state[-5:] if function_state else []
        #         logger.warning(f"[DEBUG] function_state tail (last up to 5): {tail}")
        #          
        # except Exception:
        #     pass

        # Use dynamic retrieval to get current relations
        forward_result = self.dynamic_retrieval.get_next_relations(function_state)
        reverse_result = self.dynamic_retrieval.get_next_r_relations(function_state)

        candidate_relations = []
        candidate_r_relations = []
        
        if forward_result.is_successful:
            candidate_relations = forward_result.relations
        
        if reverse_result.is_successful:
            candidate_r_relations = reverse_result.relations
        
        # Analyze filtering and combine forward and reverse relations
        combined_relations = []

        allowed_relations = self.relation_list
        if allow_literal_relations:
            allowed_relations = allowed_relations.union(self.literal_relation_list)

        # Compute filtered sets for reverse relations
        # Note: we only record whitelist filtering to file per user's request
        # filtered_by_ban_r kept for clarity but not used
        # filtered_by_ban_r = [rel for rel in candidate_r_relations if rel in self.join_ban_relation_list]
        kept_r = [rel for rel in candidate_r_relations if (rel not in self.join_ban_relation_list and rel in allowed_relations)]
        filtered_by_whitelist_r = [rel for rel in candidate_r_relations if (rel not in self.join_ban_relation_list and rel not in allowed_relations)]

        # Compute filtered sets for forward relations
        # filtered_by_ban_f = [rel for rel in candidate_relations if rel in self.join_ban_relation_list]
        kept_f = [rel for rel in candidate_relations if (rel not in self.join_ban_relation_list and rel in allowed_relations)]
        filtered_by_whitelist_f = [rel for rel in candidate_relations if (rel not in self.join_ban_relation_list and rel not in allowed_relations)]

        # Record filtered by relation_list for external inspection/logging
        self.last_filtered_by_relation_list = list(filtered_by_whitelist_r) + list(filtered_by_whitelist_f)

        # Append only relations filtered by relation_list to a dedicated log file
        try:
            if self.last_filtered_by_relation_list:
                with open(self.filtered_relations_log_path, "a", encoding="utf-8") as f:
                    for rel in self.last_filtered_by_relation_list:
                        f.write(rel + "\n")
        except Exception:
            pass

        # Add reverse relations kept
        if self.dataset == 'grailqa':
            for rel in kept_f:
                combined_relations.append((rel, rel))
            for rel in kept_r:
                combined_relations.append((rel, f'(R {rel})'))
        else:
            for rel in kept_r:
                combined_relations.append((rel, f'(R {rel})'))
            for rel in kept_f:
                combined_relations.append((rel, rel))

            
        # Add forward relations kept

        
        # Apply dataset-specific filtering (from KBQA-o1 agent.py logic)
        combined_relations = self._apply_dataset_specific_filtering(combined_relations, function_state)
            
        return combined_relations
    
    def _apply_dataset_specific_filtering(self, candidate_relations: List[Tuple[str, str]], 
                                        function_state: List[str]) -> List[Tuple[str, str]]:
        """
        Apply dataset-specific filtering logic from KBQA-o1 agent.py
        Mainly handles type.object.name filtering for WebQSP and CWQ datasets
        """
        if self.dataset.upper() == "WEBQSP":
            return self._filter_webqsp_relations(candidate_relations, function_state)
        elif self.dataset.upper() == "CWQ":
            return self._filter_cwq_relations(candidate_relations, function_state)
        else:
            # For other datasets (GrailQA, GraphQ), no specific filtering
            return candidate_relations
    
    def _filter_webqsp_relations(self, candidate_relations: List[Tuple[str, str]], 
                               function_state: List[str]) -> List[Tuple[str, str]]:
        """WebQSP-specific relation filtering (from KBQA-o1 agent.py line 139-145)"""
        if 'START' in function_state[-1]:
            # Extract entity from START function
            import re
            ent = re.findall(r"expression.*? = START\(\'(.*?)\'\)", function_state[-1])[0]
            ent_type = self.dynamic_retrieval._detect_entity_type(ent)
            if ent_type != 'entity':
                # Filter out type.object.name for non-entity types
                return [(rel, rel_id) for rel, rel_id in candidate_relations 
                        if rel != 'type.object.name']
        else:
            # For non-START states, also filter out type.object.name in WebQSP
            return [(rel, rel_id) for rel, rel_id in candidate_relations 
                    if rel != 'type.object.name']
        
        return candidate_relations
    
    def _filter_cwq_relations(self, candidate_relations: List[Tuple[str, str]], 
                            function_state: List[str]) -> List[Tuple[str, str]]:
        """CWQ-specific relation filtering (from KBQA-o1 agent.py line 635-641)"""
        if 'START' in function_state[-1]:
            # Extract entity from START function
            try:
                import re
                ent = re.findall(r"expression.*? = START\(\'(.*?)\'\)", function_state[-1])[0]
                ent_type = self.dynamic_retrieval._detect_entity_type(ent)
                # CWQ logic: keep type.object.name only for @en name types
                if not (ent_type == 'name' and ent.endswith("@en")):
                    return [(rel, rel_id) for rel, rel_id in candidate_relations 
                            if rel != 'type.object.name']
            except (IndexError, AttributeError):
                pass
        else:
            # For non-START states, also filter out type.object.name in CWQ
            return [(rel, rel_id) for rel, rel_id in candidate_relations 
                    if rel != 'type.object.name']
        
        return candidate_relations
    
    def select_best_relations(self,
                             gen_argument: str,
                             candidate_relations: List[Tuple[str, str]],
                             source: Optional[str] = None) -> List[RelationCandidate]:
        """
        Select best relations using similarity-based ranking
        Implements KBQA-o1's similarity-based relation selection
        
        Args:
            gen_argument: LLM-generated relation description
            candidate_relations: List of candidate relations
            
        Returns:
            List of ranked RelationCandidate objects
        """
        if not candidate_relations:
            return []
        
        # Extract relation names for similarity calculation
        relation_names = [r for r, _ in candidate_relations]

        scores_list: List[float] = []

        # 默认使用全局 relation_list index
        backend = self.simcse_rel

        if source == "name":
            backend = getattr(self, 'simcse_name', None) or backend
        elif source == "literal_plain":
            backend = getattr(self, 'simcse_literal_plain', None) or backend

        use_index_search = bool(backend and hasattr(backend, 'search'))

        # 1) name / literal_plain：使用各自专用 index，基于 text -> score 的 map 对 candidate_relations 回填，避免顺序依赖
        if use_index_search and source in {"name", "literal_plain"}:
            search_results = backend.search(gen_argument, top_k=200, threshold=0.0)
            score_map = {text: float(score) for text, score in search_results}
            if source == 'name':
                for rel_name, _ in candidate_relations:
                    scores_list.append(score_map.get(rel_name, 0.0))
            else: 
                for rel_name, rel_id in candidate_relations:
                    scores_list.append(score_map.get(rel_name, 0.0)) #TODO

        # 2) 其它场景：沿用全局 relation_list index 逻辑
        if use_index_search and source not in {"name", "literal_plain"}:
            # --- Legacy Logic: Global Index Search ---
            # 使用 relation_list 上的全局 index，对所有候选名称做一次全局检索再回查。
            # if getattr(self.simcse_rel, "index", None) is None:
            #     logger.warning("Index not built for legacy search, building temporary index from candidates.")
            #     self.simcse_rel.build_index(relation_names, use_faiss=False)

            search_results = self.simcse_rel.search(gen_argument, top_k=200, threshold=0.0)
            score_map = {text: float(score) for text, score in search_results}

            for name in relation_names:
                scores_list.append(score_map.get(name, 0.0))

        # 3) 如果 index 路径都拿不到分数，兜底使用 direct similarity / simple similarity
        if not scores_list:
            similarities = backend.similarity(gen_argument, relation_names) if backend and hasattr(backend, 'similarity') else self._simple_similarity(gen_argument, relation_names)
            if isinstance(similarities, (int, float)):
                scores_list = [float(similarities)]
            else:
                scores_list = similarities.tolist() if hasattr(similarities, "tolist") else list(similarities)

        
        # Expose the raw scores (pre-threshold) for callers that want full distributions
        try:
            self.last_similarity_scores = list(scores_list)
        except Exception:
            # Ensure we always keep a list
            self.last_similarity_scores = [float(s) for s in scores_list]
        
        # Create relation candidates with scores
        relation_candidates = []
        for (rel_name, rel_id), score in zip(candidate_relations, scores_list):
            is_reverse = rel_id.startswith('(R ') and rel_id.endswith(')')
            
            candidate = RelationCandidate(
                relation_name=rel_name,
                relation_id=rel_id,
                score=score,
                is_reverse=is_reverse
            )
            relation_candidates.append(candidate)
        
        # Filter by threshold
        threshold = self.config.get('relation_threshold', 0.3)
        filtered_candidates = [
            c for c in relation_candidates 
            if c.score >= threshold
        ]
        
        # Sort by score (descending)
        filtered_candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Return top-k results
        topk = self.config.get('relation_topk', 6)
        return filtered_candidates[:topk]

    def rank_relations_no_threshold(self,
                                    gen_argument: str,
                                    candidate_relations: List[Tuple[str, str]],
                                    topk: int = 20) -> List[RelationCandidate]:
        """
        Rank candidate relations by similarity without applying a threshold.
        Returns up to topk candidates sorted by score (desc).
        """
        if not candidate_relations:
            return []

        relation_names = [r for r, _ in candidate_relations]

        # Calculate scores
        if self.simcse_rel:
            similarities = self.simcse_rel.similarity(gen_argument, relation_names)
            if isinstance(similarities, (int, float)):
                scores_list = [similarities]
            else:
                scores_list = similarities.tolist() if hasattr(similarities, 'tolist') else list(similarities)
        else:
            scores_list = self._simple_similarity(gen_argument, relation_names)

        ranked: List[RelationCandidate] = []
        for (rel_name, rel_id), score in zip(candidate_relations, scores_list):
            ranked.append(RelationCandidate(
                relation_name=rel_name,
                relation_id=rel_id,
                score=score,
                is_reverse=rel_id.startswith('(R ') and rel_id.endswith(')')
            ))

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:topk]
    
    def _simple_similarity(self, query: str, candidates: List[str]) -> List[float]:
        """
        Simple string similarity calculation (fallback when SimCSE is not available)
        """
        scores = []
        query_lower = query.lower()
        
        for candidate in candidates:
            candidate_clean = candidate.replace('(R ', '').replace(')', '').replace('_', ' ').replace('.', ' ')
            candidate_lower = candidate_clean.lower()
            
            # Simple inclusion-based scoring
            if query_lower in candidate_lower:
                score = 0.8
            elif any(word in candidate_lower for word in query_lower.split()):
                score = 0.5
            else:
                score = 0.1
            
            scores.append(score)
        
        return scores


# For backward compatibility
def get_candidate_relations(function_state: List[str], dataset: str = "WebQSP") -> List[Tuple[str, str]]:
    """Standalone function for getting candidate relations"""
    retrieval = RelationRetrieval(dataset=dataset)
    return retrieval.get_candidate_relations(function_state)


def select_best_relations(gen_argument: str, candidate_relations: List[Tuple[str, str]], dataset: str = "WebQSP") -> List[RelationCandidate]:
    """Standalone function for selecting best relations"""
    retrieval = RelationRetrieval(dataset=dataset)
    return retrieval.select_best_relations(gen_argument, candidate_relations)
