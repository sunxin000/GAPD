"""
ODBC Configuration for SPARQL execution with Virtuoso.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ODBCConfig:
    """Configuration for ODBC connection to Virtuoso."""
    
    # ODBC Connection parameters
    driver_path: str = "Virtuoso"
    host: str = "localhost"
    port: int = 13001  # Default Virtuoso ODBC port
    uid: str = "dba"
    pwd: str = "dba"
    
    # Connection pool settings
    # Increased to match typical ThreadPoolExecutor max_workers (16) in sexpr_generation.py
    # This prevents thread blocking when 16 samples are processed in parallel
    pool_size: int = 48  # Match max_workers to avoid connection pool bottleneck
    max_pool_size: int = 54
    pool_timeout: int = 30
    
    # Query execution settings
    query_timeout: int = 20  # Reduced from 600 to 20 seconds for faster failure
    max_concurrent: int = 32  # Match pool_size for optimal concurrent execution
    
    # ODBC-level timeout (passed to connection string)
    connection_timeout: int = 15  # ODBC connection-level timeout
    
    # Retry settings
    max_retries: int = 1
    retry_delay: float = 1.0  # Base retry delay for exponential backoff
    
    # Experiment tracking parameters
    experiment_name: Optional[str] = None  # Current experiment name for logging
    current_step: Optional[int] = None  # Current training step for logging
    
    @property
    def connection_string(self) -> str:
        """Generate ODBC connection string."""
        # Add timeout parameter to connection string for ODBC-level timeout
        return f'DRIVER={self.driver_path};Host={self.host}:{self.port};UID={self.uid};PWD={self.pwd};Timeout={self.connection_timeout}'
    
    @classmethod
    def from_env(cls) -> 'ODBCConfig':
        """Create configuration from environment variables."""
        return cls(
            driver_path=os.getenv('FREEBASE_ODBC_DRIVER', cls.driver_path),
            host=os.getenv('FREEBASE_ODBC_HOST', cls.host),
            port=int(os.getenv('FREEBASE_ODBC_PORT', str(cls.port))),
            uid=os.getenv('FREEBASE_ODBC_UID', cls.uid),
            pwd=os.getenv('FREEBASE_ODBC_PWD', cls.pwd),
            pool_size=int(os.getenv('FREEBASE_ODBC_POOL_SIZE', str(cls.pool_size))),
            max_pool_size=int(os.getenv('FREEBASE_ODBC_MAX_POOL_SIZE', str(cls.max_pool_size))),
            query_timeout=int(os.getenv('FREEBASE_ODBC_TIMEOUT', str(cls.query_timeout))),
            max_concurrent=int(os.getenv('FREEBASE_ODBC_MAX_CONCURRENT', str(cls.max_concurrent))),
            experiment_name=os.getenv('EXPERIMENT_NAME', cls.experiment_name),
        )

# Global configuration instance
DEFAULT_CONFIG = ODBCConfig.from_env() 