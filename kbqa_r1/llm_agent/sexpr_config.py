"""
S-Expression Generation Configuration Module
Contains all configuration classes and related utilities
"""

from dataclasses import dataclass
from typing import Dict, Optional

from ..sparql.sparql_manager import SPARQLConfig
from .gpu_utilization_manager import GPUUtilizationConfig


@dataclass
class SExprGenerationConfig:
    """Configuration for S-Expression based generation"""
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    
    # S-Expression specific configurations
    enable_sexpr_mode: bool = True  # Enable S-Expression mode
    enable_action_validation: bool = True  # Validate actions before execution
    enable_sexpr_validation: bool = True  # Validate S-Expressions before conversion
    fallback_to_sparql: bool = False  # Fallback to SPARQL mode if S-Expression fails
    
    # GPU utilization management
    gpu_utilization_config: Optional[GPUUtilizationConfig] = None  # GPU utilization manager configuration
    
    # SPARQL execution configuration (for S-Expression backend)
    sparql_url: Optional[str] = None
    sparql_batch_size: int = 128
    sparql_max_concurrent: int = 16
    use_odbc: bool = True
    use_aioodbc: bool = False
    odbc_config: Optional[Dict] = None
    
    # Logging configuration
    log_dir: str = "logs"
    log_filename: Optional[str] = None
    log_interval: int = 10
    log_sample_size: int = 30
    enable_logging: bool = True
    
    # 实验信息参数
    experiment_name: Optional[str] = None  # 实验名称，用于创建子文件夹
    current_step: Optional[int] = None  # 当前训练步数，用于文件名
    
    def get_sparql_config(self) -> SPARQLConfig:
        """Get SPARQL configuration for S-Expression executor"""
        return SPARQLConfig(
            sparql_url=self.sparql_url,
            sparql_batch_size=self.sparql_batch_size,
            sparql_max_concurrent=self.sparql_max_concurrent,
            use_odbc=self.use_odbc,
            use_aioodbc=self.use_aioodbc,
            odbc_config=self.odbc_config
        )
