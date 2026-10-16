# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["SExprConfig", "ODBCConfig", "SparqlConfig", "KBQARewardConfig"]


@dataclass
class SExprConfig(BaseConfig):
    """Configuration for S-Expression generation and validation.
    
    Args:
        enable_sexpr_mode (bool): Enable S-Expression mode for KBQA.
        enable_action_reasoning (bool): Enable action reasoning in S-Expression generation.
        enable_relation_retrieval (bool): Enable relation retrieval during generation.
        validation_level (str): Validation strictness level: "NONE", "STANDARD", or "STRICT".
        max_function_calls (int): Maximum number of function calls allowed in S-Expression.
        enable_entity_linking (bool): Enable entity linking for S-Expression.
        enable_semantic_validation (bool): Enable semantic validation of S-Expression.
        use_complete_sparql_converter (bool): Use complete SPARQL converter.
    """
    
    enable_sexpr_mode: bool = False
    enable_action_reasoning: bool = False
    enable_relation_retrieval: bool = False
    validation_level: str = "STANDARD"
    max_function_calls: int = 10
    enable_entity_linking: bool = True
    enable_semantic_validation: bool = True
    use_complete_sparql_converter: bool = True


@dataclass
class ODBCConfig(BaseConfig):
    """Configuration for ODBC connection to Virtuoso knowledge base.
    
    Args:
        driver_path (str): Path or name of ODBC driver.
        host (str): Database host address.
        port (int): Database port number.
        uid (str): User ID for database connection.
        pwd (str): Password for database connection.
        pool_size (int): Initial connection pool size.
        max_pool_size (int): Maximum connection pool size.
        pool_timeout (int): Connection pool timeout in seconds.
        query_timeout (int): Query execution timeout in seconds.
        max_concurrent (int): Maximum concurrent connections.
        max_retries (int): Maximum number of retry attempts.
        retry_delay (float): Delay between retries in seconds.
    """
    
    driver_path: str = "Virtuoso"
    host: str = "localhost"
    port: int = 13001
    uid: str = "dba"
    pwd: str = "dba"
    pool_size: int = 4
    max_pool_size: int = 20
    pool_timeout: int = 30
    query_timeout: int = 600
    max_concurrent: int = 16
    max_retries: int = 1
    retry_delay: float = 1.0


@dataclass
class SparqlConfig(BaseConfig):
    """Configuration for SPARQL query execution.
    
    Args:
        url (str): SPARQL endpoint URL.
        batch_size (int): Batch size for SPARQL queries.
        max_concurrent (int): Maximum concurrent SPARQL queries.
    """
    
    url: str = "http://0.0.0.0:8000/execute"
    batch_size: int = 128
    max_concurrent: int = 16


@dataclass
class KBQARewardConfig(BaseConfig):
    """Configuration for KBQA-specific reward computation.
    
    Args:
        mid_f1_weight (float): Weight for mid-level F1 score in reward.
        structure_format_score (float): Weight for structure format score in reward.
    """
    
    mid_f1_weight: float = 1.0
    structure_format_score: float = 0.1
