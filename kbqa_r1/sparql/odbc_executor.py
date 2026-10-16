"""
ODBC-based SPARQL executor for direct connection to Virtuoso.
This version is refactored for improved stability under high concurrency.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional

import pyodbc
from tqdm import tqdm

from .odbc_config import DEFAULT_CONFIG, ODBCConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ODBCConnectionPool:
    """
    A more robust and simplified thread-safe ODBC connection pool.
    It uses a fixed-size pool, which is generally more stable under high load.
    """
    
    def __init__(self, config: ODBCConfig):
        self.config = config
        # The queue itself is the pool. Its size is the single source of truth.
        self._pool = Queue(maxsize=config.pool_size) 
        self._lock = threading.Lock()
        
        # Statistics tracking
        self._total_get_count = 0
        self._total_wait_time = 0.0
        
        logger.info(f"🔧 Initializing ODBC connection pool: pool_size={config.pool_size}, max_concurrent={config.max_concurrent}")
        
        # Pre-populate the entire pool.
        for _ in range(config.pool_size):
            try:
                conn = self._create_connection()
                self._pool.put_nowait(conn)
            except Exception as e:
                logger.critical(f"Failed to create initial connection, the pool might not function: {e}")

    def _create_connection(self) -> pyodbc.Connection:
        """Create a new ODBC connection."""
        conn = pyodbc.connect(self.config.connection_string, autocommit=True)
        # Setting autocommit=True is often more stable for read-only query workloads.
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
        conn.setencoding(encoding='utf8')
        return conn

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool. This will block until a connection is available.
        """
        conn = None
        wait_start = time.time()
        try:
            # Log pool status for debugging
            pool_size_before = self._pool.qsize()
            
            # Block until a connection is available. This is safer than dynamic creation.
            conn = self._pool.get(block=True, timeout=self.config.pool_timeout)
            
            wait_time = time.time() - wait_start
            self._total_get_count += 1
            self._total_wait_time += wait_time
            
            if wait_time > 1.0:  # Log if waiting more than 1 second
                logger.warning(f"⏳ Thread {threading.current_thread().ident} waited {wait_time:.2f}s for connection (pool had {pool_size_before} available)")
            
            yield conn
            # If yield succeeds, the connection is assumed to be healthy.
            self._pool.put_nowait(conn) 
        except Empty:
            # This happens if the timeout is reached.
            logger.error("Could not get a connection from the pool within the timeout period.")
            raise Exception("Connection pool timeout")
        except Full:
            # This can happen if the connection logic has a bug, but it's good to handle.
            logger.warning("Connection pool is somehow full. A connection might be leaked.")
            # We don't re-raise, but the connection `conn` will be lost.
        except Exception:
            # Any other exception happens, we assume the connection is bad.
            # We don't return it to the pool. Instead, we close it and create a new one to replenish the pool.
            if conn:
                try:
                    conn.close()
                except Exception as close_e:
                    logger.warning(f"Error closing a bad connection: {close_e}")
            
            # Replenish the pool to maintain its size
            self._replenish_one_connection()
            raise # Re-raise the original exception

    def _replenish_one_connection(self):
        """Creates a new connection and adds it to the pool."""
        try:
            with self._lock: # Lock to prevent multiple threads from replenishing at once
                if self._pool.qsize() < self.config.pool_size:
                    new_conn = self._create_connection()
                    self._pool.put_nowait(new_conn)
                    logger.info("Connection pool was replenished with a new connection.")
        except Full:
            pass # Another thread might have replenished it already.
        except Exception as e:
            logger.error(f"Failed to replenish connection: {e}")
            
    def close_all(self):
        """Close all connections in the pool."""
        # Print statistics before closing
        if self._total_get_count > 0:
            avg_wait = self._total_wait_time / self._total_get_count
            logger.info(f"📊 Connection pool statistics:")
            logger.info(f"   Total connection requests: {self._total_get_count}")
            logger.info(f"   Total wait time: {self._total_wait_time:.2f}s")
            logger.info(f"   Average wait time: {avg_wait:.3f}s")
            if avg_wait > 0.1:
                logger.warning(f"   ⚠️  High average wait time detected! Consider increasing pool_size (current: {self.config.pool_size})")
        
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
            except Exception as e:
                logger.warning(f"Error closing a connection during pool shutdown: {e}")

class ODBCSPARQLExecutor:
    """ODBC-based SPARQL executor with the refactored, more stable connection pool."""
    
    def __init__(self, config: Optional[ODBCConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.pool = ODBCConnectionPool(self.config)
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent)
    
    def execute_single_query(self, query: str) -> Dict[str, Any]:
        """Execute a single SPARQL query with more stable resource management."""
        if not query.strip():
            return {"query": query, "results": [], "error": "Empty query"}
        
        if not query.strip().upper().startswith('SPARQL'):
            query = f"SPARQL {query}"
        
        try:
            with self.pool.get_connection() as conn:
                conn.timeout = 80
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()

                results = []
                if rows:
                    columns = [desc[0] for desc in cursor.description]
                    for row in rows:
                        result = {}
                        for i, value in enumerate(row):
                            if value is not None:
                                if isinstance(value, str):
                                    value = value.replace('\u0000', '').replace('\x00', '')
                                if isinstance(value, str) and value.startswith('http://rdf.freebase.com/ns/'):
                                    value = value.replace('http://rdf.freebase.com/ns/', '')
                                result[columns[i]] = value
                        results.append(result)

                cursor.close()
                return {"query": query, "results": results}

        except pyodbc.Error as db_err:
            error_msg = f"ODBC Error: {db_err}"
            logger.warning(f"Query execution failed: {error_msg}")
            return {"query": query, "results": [], "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected Error: {e}"
            logger.error(f"Query execution failed: {e}")
            return {"query": query, "results": [], "error": error_msg}
    
    def execute_batch(self, queries: List[str]) -> Dict[str, Any]:
        """Execute a batch of SPARQL queries concurrently."""
        if not queries:
            return {"results": []}

        logger.info(f"🚀 Executing {len(queries)} SPARQL queries using ODBC...")
        start_time = time.time()
        
        all_results = [None] * len(queries)
        query_map = {id(query): i for i, query in enumerate(queries)}

        with tqdm(total=len(queries), desc="🔍 SPARQL Processing", unit="q") as pbar:
            future_to_query = {self.executor.submit(self.execute_single_query, query): query for query in queries}
            
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    # Place result in the correct position to maintain order
                    idx = query_map[id(query)]
                    all_results[idx] = result
                except Exception as exc:
                    logger.error(f'Query generated an exception: {exc}')
                    idx = query_map[id(query)]
                    all_results[idx] = {"query": query, "results": [], "error": str(exc)}
                finally:
                    pbar.update(1)

        elapsed_time = time.time() - start_time
        failed_count = sum(1 for r in all_results if r and r.get("error"))
        success_rate = ((len(queries) - failed_count) / len(queries)) * 100 if queries else 0
        
        logger.info(f"✅ Completed in {elapsed_time:.2f}s | 📈 Rate: {len(queries) / elapsed_time:.1f} q/s | 🎯 Success: {success_rate:.1f}% ({failed_count} failed)")
        
        return {"results": all_results}
    
    def close(self):
        """Close the executor and connection pool."""
        self.executor.shutdown(wait=True)
        self.pool.close_all()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Singleton pattern management remains the same.
_global_executor = None
_executor_lock = threading.Lock()

def get_global_executor(config: Optional[ODBCConfig] = None) -> ODBCSPARQLExecutor:
    """Get or create the global ODBC SPARQL executor."""
    global _global_executor
    if _global_executor is None:
        with _executor_lock:
            if _global_executor is None:
                logger.info("Creating a new global ODBCSPARQLExecutor instance.")
                _global_executor = ODBCSPARQLExecutor(config)
    else:
        # Refresh experiment_name/current_step on the cached executor for logging
        if config is not None:
            try:
                if getattr(config, "experiment_name", None) is not None:
                    _global_executor.config.experiment_name = config.experiment_name
                if getattr(config, "current_step", None) is not None:
                    _global_executor.config.current_step = config.current_step
            except Exception:
                pass
    return _global_executor

def execute_sparql_odbc(queries: List[str], config: Optional[ODBCConfig] = None) -> Dict[str, Any]:
    executor = get_global_executor(config)
    return executor.execute_batch(queries)
