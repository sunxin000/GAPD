"""
Robust ODBC connection pool designed to prevent SIGSEGV crashes and handle high concurrency.
This implementation focuses on thread safety and graceful error handling.
"""
import threading
import time
import queue
import logging
import weakref
import gc
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RobustODBCConfig:
    """Enhanced ODBC configuration for robust connection handling."""
    driver_path: str = "utils/lib/virtodbc.so"
    host: str = "localhost"
    port: int = 1111
    uid: str = "dba"
    pwd: str = "dba"
    
    # Connection pool settings
    pool_size: int = 8  # Conservative pool size
    max_pool_size: int = 16
    pool_timeout: int = 30
    connection_timeout: int = 10
    
    # Query execution settings
    query_timeout: int = 20
    batch_size: int = 64  # Smaller batches for stability
    max_concurrent: int = 4  # Reduced concurrency
    
    # Retry and backoff settings
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    
    # Health check settings
    health_check_interval: int = 60  # seconds
    connection_max_age: int = 3600   # 1 hour max connection age
    
    @property
    def connection_string(self) -> str:
        """Generate ODBC connection string with optimized parameters."""
        return (f'DRIVER={self.driver_path};'
                f'Host={self.host}:{self.port};'
                f'UID={self.uid};'
                f'PWD={self.pwd};'
                f'ConnectTimeout={self.connection_timeout};'
                f'LoginTimeout={self.connection_timeout};'
                f'QueryTimeout={self.query_timeout}')

class ConnectionWrapper:
    """Wrapper for ODBC connection with metadata and health tracking."""
    
    def __init__(self, connection, config: RobustODBCConfig):
        self.connection = connection
        self.config = config
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.is_healthy = True
        self._lock = threading.RLock()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.last_used = time.time()
        if exc_type is not None:
            self.is_healthy = False
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query with proper error handling."""
        with self._lock:
            if not self.is_healthy:
                raise Exception("Connection is marked unhealthy")
            
            try:
                self.use_count += 1
                self.last_used = time.time()
                
                with self.connection.cursor() as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    # Process results
                    results = []
                    if rows:
                        columns = [desc[0] for desc in cursor.description]
                        for row in rows:
                            result = {}
                            for i, value in enumerate(row):
                                if value is not None:
                                    # Clean Freebase namespaces
                                    if isinstance(value, str) and value.startswith('http://rdf.freebase.com/ns/'):
                                        value = value.replace('http://rdf.freebase.com/ns/', '')
                                    result[columns[i]] = value
                            results.append(result)
                    
                    return results
                    
            except Exception as e:
                self.is_healthy = False
                logger.error(f"Query execution failed: {e}")
                raise
    
    def is_expired(self) -> bool:
        """Check if connection has expired."""
        age = time.time() - self.created_at
        return age > self.config.connection_max_age
    
    def close(self):
        """Safely close the connection."""
        try:
            if hasattr(self, 'connection') and self.connection:
                self.connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        finally:
            self.connection = None
            self.is_healthy = False

class RobustODBCPool:
    """Thread-safe, robust ODBC connection pool with advanced error handling."""
    
    def __init__(self, config: RobustODBCConfig):
        self.config = config
        self._pool = queue.Queue(maxsize=config.max_pool_size)
        self._active_connections = 0
        self._total_connections = 0
        self._lock = threading.RLock()
        self._shutdown = False
        self._stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'queries_executed': 0,
            'errors': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        # Background health monitoring
        self._health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self._health_thread.start()
        
        # Initialize minimum pool size
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the pool with minimum connections."""
        logger.info(f"Initializing ODBC pool with {self.config.pool_size} connections")
        
        for _ in range(min(self.config.pool_size, self.config.max_pool_size)):
            try:
                conn_wrapper = self._create_connection()
                self._pool.put(conn_wrapper, block=False)
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
                break
    
    def _create_connection(self) -> ConnectionWrapper:
        """Create a new ODBC connection with enhanced error handling."""
        if not PYODBC_AVAILABLE:
            raise Exception("pyodbc not available")
        
        try:
            # Configure pyodbc for better stability
            pyodbc.pooling = False  # Disable pyodbc's own pooling
            
            # Create connection with timeout
            connection = pyodbc.connect(
                self.config.connection_string,
                timeout=self.config.connection_timeout
            )
            
            # Configure connection for UTF-8 and stability
            connection.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
            connection.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
            connection.setencoding(encoding='utf8')
            
            # Set autocommit for better stability
            connection.autocommit = True
            
            with self._lock:
                self._total_connections += 1
                self._stats['connections_created'] += 1
            
            logger.debug(f"Created new ODBC connection (total: {self._total_connections})")
            
            return ConnectionWrapper(connection, self.config)
            
        except Exception as e:
            with self._lock:
                self._stats['errors'] += 1
            logger.error(f"Failed to create ODBC connection: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with robust error handling."""
        if self._shutdown:
            raise Exception("Connection pool is shutdown")
        
        conn_wrapper = None
        start_time = time.time()
        
        try:
            # Try to get connection from pool
            try:
                conn_wrapper = self._pool.get(timeout=self.config.pool_timeout)
                with self._lock:
                    self._stats['pool_hits'] += 1
                
                # Check if connection is still healthy and not expired
                if not conn_wrapper.is_healthy or conn_wrapper.is_expired():
                    conn_wrapper.close()
                    conn_wrapper = None
                    
            except queue.Empty:
                with self._lock:
                    self._stats['pool_misses'] += 1
            
            # Create new connection if needed
            if conn_wrapper is None:
                with self._lock:
                    if self._total_connections < self.config.max_pool_size:
                        conn_wrapper = self._create_connection()
                        self._active_connections += 1
                    else:
                        raise Exception("Connection pool exhausted and max connections reached")
            
            if conn_wrapper is None:
                raise Exception("Could not obtain connection from pool")
            
            yield conn_wrapper
            
        except Exception as e:
            # Handle connection errors
            if conn_wrapper:
                conn_wrapper.close()
                with self._lock:
                    self._active_connections = max(0, self._active_connections - 1)
                    self._total_connections = max(0, self._total_connections - 1)
                    self._stats['connections_closed'] += 1
                conn_wrapper = None
            raise e
        
        finally:
            # Return connection to pool or close it
            if conn_wrapper:
                try:
                    if conn_wrapper.is_healthy and not conn_wrapper.is_expired():
                        # Return healthy connection to pool
                        self._pool.put(conn_wrapper, block=False)
                    else:
                        # Close unhealthy or expired connection
                        conn_wrapper.close()
                        with self._lock:
                            self._active_connections = max(0, self._active_connections - 1)
                            self._total_connections = max(0, self._total_connections - 1)
                            self._stats['connections_closed'] += 1
                except queue.Full:
                    # Pool is full, close the connection
                    conn_wrapper.close()
                    with self._lock:
                        self._active_connections = max(0, self._active_connections - 1)
                        self._total_connections = max(0, self._total_connections - 1)
                        self._stats['connections_closed'] += 1
    
    def _health_monitor(self):
        """Background thread to monitor connection health."""
        while not self._shutdown:
            try:
                time.sleep(self.config.health_check_interval)
                
                if self._shutdown:
                    break
                
                # Clean up expired connections
                expired_connections = []
                temp_connections = []
                
                # Drain pool temporarily
                while True:
                    try:
                        conn_wrapper = self._pool.get_nowait()
                        if conn_wrapper.is_expired() or not conn_wrapper.is_healthy:
                            expired_connections.append(conn_wrapper)
                        else:
                            temp_connections.append(conn_wrapper)
                    except queue.Empty:
                        break
                
                # Close expired connections
                for conn_wrapper in expired_connections:
                    conn_wrapper.close()
                    with self._lock:
                        self._total_connections = max(0, self._total_connections - 1)
                        self._stats['connections_closed'] += 1
                
                # Return healthy connections to pool
                for conn_wrapper in temp_connections:
                    try:
                        self._pool.put(conn_wrapper, block=False)
                    except queue.Full:
                        conn_wrapper.close()
                        with self._lock:
                            self._total_connections = max(0, self._total_connections - 1)
                            self._stats['connections_closed'] += 1
                
                logger.debug(f"Pool health check: {len(expired_connections)} expired, "
                           f"{len(temp_connections)} healthy, "
                           f"{self._total_connections} total")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a single query with robust error handling."""
        if not query.strip():
            raise ValueError("Empty query")
        
        # Ensure query starts with SPARQL keyword
        if not query.strip().upper().startswith('SPARQL'):
            query = f"SPARQL {query}"
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                with self.get_connection() as conn_wrapper:
                    result = conn_wrapper.execute_query(query)
                    with self._lock:
                        self._stats['queries_executed'] += 1
                    return result
                    
            except Exception as e:
                last_error = e
                with self._lock:
                    self._stats['errors'] += 1
                
                error_msg = str(e)
                
                # Check for non-retryable errors
                if any(indicator in error_msg for indicator in [
                    "Log out of disk", "SR174", "GPF", "SIGSEGV",
                    "Connection pool exhausted"
                ]):
                    logger.error(f"Non-retryable error: {error_msg}")
                    break
                
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (self.config.backoff_multiplier ** attempt)
                    logger.warning(f"Query failed (attempt {attempt + 1}), retrying in {delay}s: {error_msg}")
                    time.sleep(delay)
                else:
                    logger.error(f"Query failed after {self.config.max_retries} attempts: {error_msg}")
        
        if last_error:
            raise last_error
        else:
            raise Exception("Unknown error during query execution")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                **self._stats,
                'active_connections': self._active_connections,
                'total_connections': self._total_connections,
                'pool_size': self._pool.qsize(),
                'max_pool_size': self.config.max_pool_size
            }
    
    def close_all(self):
        """Shutdown the pool and close all connections."""
        logger.info("Shutting down ODBC connection pool")
        self._shutdown = True
        
        # Close all connections in pool
        while True:
            try:
                conn_wrapper = self._pool.get_nowait()
                conn_wrapper.close()
                with self._lock:
                    self._stats['connections_closed'] += 1
            except queue.Empty:
                break
        
        # Reset counters
        with self._lock:
            self._active_connections = 0
            self._total_connections = 0
        
        # Force garbage collection
        gc.collect()
        
        logger.info("ODBC connection pool shutdown complete")

# Global robust pool instance
_global_robust_pool = None
_pool_lock = threading.Lock()

def get_robust_pool(config: Optional[RobustODBCConfig] = None) -> RobustODBCPool:
    """Get or create the global robust ODBC pool."""
    global _global_robust_pool
    
    with _pool_lock:
        if _global_robust_pool is None:
            if config is None:
                config = RobustODBCConfig()
            _global_robust_pool = RobustODBCPool(config)
        return _global_robust_pool

def execute_sparql_robust(queries: List[str], config: Optional[RobustODBCConfig] = None) -> Dict[str, Any]:
    """
    Execute SPARQL queries using a simple, robust approach.
    """
    if not PYODBC_AVAILABLE:
        raise Exception("pyodbc not available")
    
    if config is None:
        config = RobustODBCConfig()
    
    print(f"🚀 Executing {len(queries)} SPARQL queries using robust ODBC")
    start_time = time.time()
    
    results = []
    failed_queries = 0
    
    for i, query in enumerate(queries):
        if not query.strip():
            results.append({"query": query, "error": "Empty query"})
            failed_queries += 1
            continue
        
        # Ensure query starts with SPARQL keyword
        if not query.strip().upper().startswith('SPARQL'):
            query = f"SPARQL {query}"
        
        success = False
        last_error = None
        
        for attempt in range(config.max_retries):
            try:
                # Create a fresh connection for each query (simplest approach)
                conn = pyodbc.connect(config.connection_string, timeout=config.connection_timeout)
                conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
                conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
                conn.setencoding(encoding='utf8')
                conn.autocommit = True
                
                try:
                    with conn.cursor() as cursor:
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        
                        # Process results
                        query_results = []
                        if rows:
                            columns = [desc[0] for desc in cursor.description]
                            for row in rows:
                                result = {}
                                for j, value in enumerate(row):
                                    if value is not None:
                                        # Clean null characters that may come from database
                                        if isinstance(value, str):
                                            value = value.replace('\u0000', '').replace('\x00', '')
                                        if isinstance(value, str) and value.startswith('http://rdf.freebase.com/ns/'):
                                            value = value.replace('http://rdf.freebase.com/ns/', '')
                                        result[columns[j]] = value
                                query_results.append(result)
                        
                        results.append({
                            "query": query,
                            "results": query_results
                        })
                        success = True
                        break
                        
                finally:
                    conn.close()
                    
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Check for non-retryable errors
                if any(indicator in error_msg for indicator in [
                    "Log out of disk", "SR174", "GPF", "SIGSEGV"
                ]):
                    logger.error(f"Non-retryable error for query {i+1}: {error_msg}")
                    break
                
                if attempt < config.max_retries - 1:
                    delay = config.retry_delay * (config.backoff_multiplier ** attempt)
                    logger.warning(f"Query {i+1} failed (attempt {attempt + 1}), retrying in {delay}s: {error_msg}")
                    time.sleep(delay)
        
        if not success:
            failed_queries += 1
            error_msg = str(last_error) if last_error else "Unknown error"
            results.append({
                "query": query,
                "error": error_msg
            })
            logger.error(f"Query {i+1} failed after {config.max_retries} attempts: {error_msg}")
    
    elapsed_time = time.time() - start_time
    queries_per_second = len(queries) / elapsed_time if elapsed_time > 0 else 0
    success_rate = ((len(queries) - failed_queries) / len(queries)) * 100 if queries else 0
    
    print(f"✅ Completed {len(queries)} SPARQL queries in {elapsed_time:.2f} seconds")
    print(f"📊 Performance: {queries_per_second:.2f} queries/second")
    print(f"📈 Success rate: {success_rate:.1f}% ({failed_queries} failed)")
    
    return {"results": results} 