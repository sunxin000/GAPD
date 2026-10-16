"""
Async ODBC-based SPARQL executor using aioodbc for true asynchronous database operations.
This implementation provides better performance for concurrent SPARQL queries.
"""
import asyncio
import faulthandler
import json
import logging
import os
import signal
import threading
import time
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm

# try:
#     import aioodbc
#     import pyodbc  # Need pyodbc for encoding constants
#     AIOODBC_AVAILABLE = True
# except ImportError:
#     AIOODBC_AVAILABLE = False

# from .odbc_config import DEFAULT_CONFIG, ODBCConfig

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))  # Default to INFO for S-Expression logs


# def _format_query_for_log(query: str, max_length: int = 1000) -> str:
#     """Format query for logging, truncating if necessary."""
#     if len(query) <= max_length:
#         return query
#     return query[:max_length] + "... [TRUNCATED]"

# def _format_log_message(message: str, config: Optional['ODBCConfig'] = None) -> str:
#     """Format log message with experiment name and step if available."""
#     if config and config.experiment_name:
#         if config.current_step is not None:
#             return f"[SPARQL-ODBC] 🚀 EXPERIMENT: {config.experiment_name} - Step {config.current_step} - {message}"
#         else:
#             return f"[SPARQL-ODBC] 🚀 EXPERIMENT: {config.experiment_name} - {message}"
#     else:
#         return f"[SPARQL-ODBC] {message}"

# def _ensure_crash_dump_enabled() -> None:
#     """Enable faulthandler to capture fatal crashes (e.g., SIGSEGV from ODBC driver)."""
#     try:
#         os.makedirs("debug_logs", exist_ok=True)
#         crash_log_path = os.path.join("debug_logs", f"faulthandler_{os.getpid()}.log")
#         # Keep file handle open for the process lifetime
#         if not hasattr(_ensure_crash_dump_enabled, "_fh_file"):
#             _ensure_crash_dump_enabled._fh_file = open(crash_log_path, "a", encoding="utf-8")  # type: ignore[attr-defined]
#             faulthandler.enable(file=_ensure_crash_dump_enabled._fh_file, all_threads=True)  # type: ignore[arg-type]
#             # Register common fatal signals
#             for sig in (getattr(signal, "SIGSEGV", None), getattr(signal, "SIGBUS", None), getattr(signal, "SIGABRT", None)):
#                 if sig is not None:
#                     try:
#                         faulthandler.register(sig, file=_ensure_crash_dump_enabled._fh_file, all_threads=True)  # type: ignore[arg-type]
#                     except Exception:
#                         pass
#     except Exception:
#         # Never let diagnostics crash the program
#         pass

# def _persist_runtime_context(record: Dict[str, Any]) -> None:
#     """Persist a runtime context record both as append-only JSONL and as the latest snapshot.

#     This helps post-mortem analysis when the interpreter crashes due to native code (e.g., Virtuoso ODBC).
#     """
#     try:
#         os.makedirs("debug_logs", exist_ok=True)
#         record_copy = dict(record)
#         record_copy["timestamp"] = record_copy.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
#         # Append to rolling file
#         with open(os.path.join("debug_logs", "last_runtime_context.jsonl"), "a", encoding="utf-8") as f:
#             f.write(json.dumps(record_copy, ensure_ascii=False) + "\n")
#         # Write snapshot
#         with open(os.path.join("debug_logs", "last_runtime_context.json"), "w", encoding="utf-8") as f:
#             json.dump(record_copy, f, ensure_ascii=False, indent=2)
#     except Exception:
#         # Silent fail – diagnostics must not interfere with execution
#         pass

# class AsyncODBCSPARQLExecutor:
#     """
#     Async ODBC-based SPARQL executor using aioodbc for improved concurrency.
#     This executor provides true asynchronous database operations without blocking the event loop.
#     """
    
#     def __init__(self, config: Optional[ODBCConfig] = None):
#         if not AIOODBC_AVAILABLE:
#             raise ImportError("aioodbc is not available. Please install it with: pip install aioodbc")
        
#         self.config = config or DEFAULT_CONFIG
#         self.pool = None
#         self._pool_lock = None
#         _ensure_crash_dump_enabled()
        
#     async def _ensure_pool(self):
#         """Ensure the connection pool is initialized."""
#         if self.pool is None:
#             # Create lock for current event loop if not exists
#             if self._pool_lock is None:
#                 self._pool_lock = asyncio.Lock()
            
#             async with self._pool_lock:
#                 if self.pool is None:
#                     logger.info(_format_log_message("Creating aioodbc connection pool...", self.config))
#                     self.pool = await aioodbc.create_pool(
#                         dsn=self.config.connection_string,
#                         minsize=1,
#                         maxsize=self.config.pool_size,
#                         timeout=self.config.pool_timeout,
#                         echo=False,
#                         autocommit=True
#                     )
    
#     def _get_debug_context(self, provided_context: Optional[Dict[str, Any]] = None) -> str:
#         """Get debug context information for slow query logging."""
#         try:
#             context_parts = []
            
#             # Use provided context first if available
#             if provided_context:
#                 if 'input' in provided_context:
#                     context_parts.append(f"input={str(provided_context['input'])[:100]}...")
#                 if 'response' in provided_context:
#                     context_parts.append(f"response={str(provided_context['response'])[:100]}...")
#                 if 'sexpr_list' in provided_context:
#                     sexpr_list = provided_context['sexpr_list']
#                     if isinstance(sexpr_list, list):
#                         context_parts.append(f"sexpr_list={len(sexpr_list)} items")
#                         if sexpr_list:
#                             context_parts.append(f"last_sexpr={str(sexpr_list[-1])[:50]}...")
#                     else:
#                         context_parts.append(f"sexpr_list={str(sexpr_list)[:100]}...")
#                 if 'sample_id' in provided_context:
#                     context_parts.append(f"sample_id={provided_context['sample_id']}")
#                 if 'function_state' in provided_context:
#                     func_state = provided_context['function_state']
#                     if isinstance(func_state, list):
#                         context_parts.append(f"function_state={len(func_state)} functions")
#                     else:
#                         context_parts.append(f"function_state={str(func_state)[:50]}...")
            
#             # Try to get context from debug logs if no provided context
#             if not provided_context:
#                 debug_context = self._load_debug_context()
#                 if debug_context:
#                     context_parts.append(f"sexpr={debug_context.get('sexpr', 'N/A')[:100]}...")
#                     context_parts.append(f"function_state={len(debug_context.get('function_state', []))} functions")
                
#                 # Try to get current sample context from state manager
#                 sample_context = self._get_current_sample_context()
#                 if sample_context:
#                     context_parts.append(f"sample_id={sample_context.get('sample_id', 'N/A')}")
#                     context_parts.append(f"prompt={sample_context.get('prompt', 'N/A')[:50]}...")
#                     context_parts.append(f"prediction={sample_context.get('prediction', 'N/A')[:50]}...")
            
#             # Add experiment info if available
#             if self.config and hasattr(self.config, 'experiment_name') and self.config.experiment_name:
#                 context_parts.append(f"experiment={self.config.experiment_name}")
#             if self.config and hasattr(self.config, 'current_step') and self.config.current_step is not None:
#                 context_parts.append(f"step={self.config.current_step}")
            
#             return " | ".join(context_parts) if context_parts else "No context available"
            
#         except Exception as e:
#             return f"Context retrieval failed: {str(e)}"
    
#     def _load_debug_context(self) -> Optional[Dict[str, Any]]:
#         """Load the latest debug context from files."""
#         try:
#             debug_file = os.path.join("debug_logs", "last_sexpr_context.json")
#             if os.path.exists(debug_file):
#                 with open(debug_file, "r", encoding="utf-8") as f:
#                     return json.load(f)
#         except Exception:
#             pass
#         return None
    
#     def _get_current_sample_context(self) -> Optional[Dict[str, Any]]:
#         """Get current sample context from state manager if available."""
#         try:
#             # Try to read from runtime context files first
#             runtime_context = self._load_runtime_context()
#             if runtime_context:
#                 return {
#                     'sample_id': runtime_context.get('sample_id', 'N/A'),
#                     'prompt': runtime_context.get('prompt', 'N/A'),
#                     'prediction': runtime_context.get('prediction', 'N/A')
#                 }
            
#             # Fallback: try to access the global state manager if it exists
#             import sys
#             current_frame = sys._getframe()
            
#             # Look for state manager in the call stack
#             for i in range(10):  # Check up to 10 frames up
#                 frame = current_frame.f_back
#                 if frame is None:
#                     break
                    
#                 # Check if this frame has a state_manager attribute
#                 if 'self' in frame.f_locals:
#                     obj = frame.f_locals['self']
#                     if hasattr(obj, 'state_manager'):
#                         state_manager = obj.state_manager
#                         if hasattr(state_manager, '_sample_prompts'):
#                             # Find the most recent sample with a prompt
#                             for sample_id, prompt in state_manager._sample_prompts.items():
#                                 if prompt:
#                                     return {
#                                         'sample_id': sample_id,
#                                         'prompt': prompt,
#                                         'prediction': getattr(frame.f_locals.get('pred', ''), 'text', '') if 'pred' in frame.f_locals else ''
#                                     }
                
#                 current_frame = frame
#         except Exception:
#             pass
#         return None
    
#     def _load_runtime_context(self) -> Optional[Dict[str, Any]]:
#         """Load the latest runtime context from files."""
#         try:
#             runtime_file = os.path.join("debug_logs", "last_runtime_context.json")
#             if os.path.exists(runtime_file):
#                 with open(runtime_file, "r", encoding="utf-8") as f:
#                     return json.load(f)
#         except Exception:
#             pass
#         return None
    
#     async def _monitor_slow_query(self, query: str, query_completed: asyncio.Event, context: Optional[Dict[str, Any]] = None):
#         """Monitor query execution and log if it takes longer than 20 seconds."""
#         try:
#             await asyncio.wait_for(query_completed.wait(), timeout=20.0)
#         except asyncio.TimeoutError:
#             # Query is still running after 20 seconds - log it immediately with context
#             formatted_query = _format_query_for_log(query)
#             context_info = self._get_debug_context(context)
#             logger.warning(_format_log_message(f"🐌 Slow query detected (20s+) - Still executing | "
#                          f"Query: {formatted_query} | Context: {context_info}", self.config))

#     async def _execute_query_core(self, query: str, query_completed: asyncio.Event) -> Dict[str, Any]:
#         """Core query execution logic separated for monitoring."""
#         for attempt in range(self.config.max_retries):
#             try:
#                 # Use asyncio.wait_for to add query-level timeout control
#                 async with self.pool.acquire() as conn:
#                     # Configure encoding to prevent UTF-16 issues
#                     try:
#                         # Set encoding configuration similar to sync ODBC executors
#                         if hasattr(conn, '_conn') and hasattr(conn._conn, 'setdecoding'):
#                             conn._conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
#                             conn._conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
#                             conn._conn.setencoding(encoding='utf8')
#                         elif hasattr(conn, 'setdecoding'):
#                             conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
#                             conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
#                             conn.setencoding(encoding='utf8')
#                     except (AttributeError, TypeError) as e:
#                         # Log warning but continue - some versions may not support this
#                         logger.debug(_format_log_message(f"Could not set encoding on aioodbc connection: {e}", self.config))
                    
#                     # Persist context before issuing the native call
#                     _persist_runtime_context({
#                         "phase": "before_execute",
#                         "pid": os.getpid(),
#                         "thread": threading.current_thread().name,
#                         "attempt": attempt + 1,
#                         "max_retries": self.config.max_retries,
#                         "config": {
#                             "host": getattr(self.config, "host", None),
#                             "port": getattr(self.config, "port", None),
#                             "pool_size": getattr(self.config, "pool_size", None),
#                             "max_concurrent": getattr(self.config, "max_concurrent", None),
#                             "query_timeout": getattr(self.config, "query_timeout", None),
#                             "experiment_name": getattr(self.config, "experiment_name", None),
#                             "current_step": getattr(self.config, "current_step", None),
#                         },
#                         "query": query,
#                     })
#                     async with conn.cursor() as cursor:
#                         # Wrap the actual query execution with timeout and safer error handling
#                         # to prevent triggering SQLEndTran bugs in Virtuoso ODBC driver
#                         try:
#                             await asyncio.wait_for(
#                                 cursor.execute(query),
#                                 timeout=self.config.query_timeout
#                             )
#                             rows = await asyncio.wait_for(
#                                 cursor.fetchall(),
#                                 timeout=self.config.query_timeout
#                             )
#                         except (asyncio.TimeoutError, Exception):
#                             # On timeout or error, manually close cursor to avoid
#                             # triggering automatic transaction cleanup (SQLEndTran)
#                             # which can cause Virtuoso ODBC driver to segfault
#                             try:
#                                 await cursor.close()
#                             except:
#                                 pass  # Ignore cursor close errors
#                             raise  # Re-raise the original exception
                        
#                         results = []
#                         if rows:
#                             columns = [desc[0] for desc in cursor.description]
#                             for row in rows:
#                                 result = {}
#                                 for i, value in enumerate(row):
#                                     if value is not None:
#                                         if isinstance(value, str) and value.startswith('http://rdf.freebase.com/ns/'):
#                                             value = value.replace('http://rdf.freebase.com/ns/', '')
#                                         result[columns[i]] = value
#                                 results.append(result)
#                         # Persist context after successful fetch
#                         _persist_runtime_context({
#                             "phase": "after_execute",
#                             "pid": os.getpid(),
#                             "thread": threading.current_thread().name,
#                             "attempt": attempt + 1,
#                             "results_count": len(results),
#                             "query": query,
#                         })
#                         return {"query": query, "results": results}
                        
#             except asyncio.TimeoutError:
#                 # Only retry on timeout errors
#                 retry_delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
#                 logger.warning(_format_log_message(f"Query timeout on attempt {attempt + 1}/{self.config.max_retries}. "
#                              f"Retrying in {retry_delay:.1f}s...", self.config))
#                 if attempt < self.config.max_retries - 1:  # Don't sleep on last attempt
#                     await asyncio.sleep(retry_delay)
#                 continue
                
#             except Exception as e:
#                 # For all other errors, don't retry - return immediately
#                 error_msg = f"Query execution failed: {e}"
#                 logger.error(_format_log_message(f"Non-timeout error (no retry): {error_msg}", self.config))
#                 _persist_runtime_context({
#                     "phase": "exception",
#                     "pid": os.getpid(),
#                     "thread": threading.current_thread().name,
#                     "attempt": attempt + 1,
#                     "error": str(e),
#                     "query": query,
#                 })
#                 return {"query": query, "results": [], "error": error_msg}
        
#         # If we exhausted all retries due to timeouts
#         error_msg = f"Query timed out after {self.config.max_retries} attempts"
#         logger.error(_format_log_message(error_msg, self.config))
#         return {"query": query, "results": [], "error": error_msg}

#     async def execute_single_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """Execute a single SPARQL query asynchronously with real-time slow query monitoring."""
#         if not query.strip():
#             return {"query": query, "results": [], "error": "Empty query"}
        
#         if not query.strip().upper().startswith('SPARQL'):
#             query = f"SPARQL {query}"
        
#         await self._ensure_pool()
        
#         # Create event to signal when query completes
#         query_completed = asyncio.Event()
        
#         # Create monitoring task for 20-second detection with context
#         monitor_task = asyncio.create_task(
#             self._monitor_slow_query(query, query_completed, context)
#         )
        
#         # Create query execution task
#         query_task = asyncio.create_task(
#             self._execute_query_core(query, query_completed)
#         )
        
#         try:
#             # Wait for query to complete (monitoring runs in parallel)
#             result = await query_task
#             return result
#         finally:
#             # Signal that query is completed
#             query_completed.set()
#             # Cancel monitoring task if it's still running
#             if not monitor_task.done():
#                 monitor_task.cancel()
#                 try:
#                     await monitor_task
#                 except asyncio.CancelledError:
#                     pass
    
#     async def execute_batch(self, queries: List[str], contexts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
#         """Execute a batch of SPARQL queries asynchronously with optimized concurrency control."""
#         if not queries:
#             return {"results": []}
        
#         logger.info(_format_log_message(f"🚀 Executing {len(queries)} SPARQL queries using async aioodbc...", self.config))
#         start_time = time.time()
        
#         await self._ensure_pool()
        
#         # Use semaphore to control concurrency and prevent overwhelming the database
#         semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
#         async def execute_with_semaphore(query: str, index: int, context: Optional[Dict[str, Any]] = None) -> tuple[int, Dict[str, Any]]:
#             """Execute query with semaphore to control concurrency."""
#             async with semaphore:
#                 result = await self.execute_single_query(query, context)
#                 return index, result
        
#         # Create tasks for all queries with their contexts
#         tasks = [
#             execute_with_semaphore(query, i, contexts[i] if contexts and i < len(contexts) else None) 
#             for i, query in enumerate(queries)
#         ]
        
#         # Execute all tasks concurrently with progress bar
#         results = [None] * len(queries)
        
#         # Use tqdm for async progress tracking
#         with tqdm(total=len(queries), desc="🔍 Async SPARQL Processing", unit="q") as pbar:
#             for coro in asyncio.as_completed(tasks):
#                 try:
#                     index, result = await coro
#                     results[index] = result
#                     pbar.update(1)
#                 except Exception as exc:
#                     logger.error(f'Query generated an exception: {exc}')
#                     # Handle the case where we can't determine the index
#                     # This shouldn't happen with our current implementation
#                     pbar.update(1)
        
#         elapsed_time = time.time() - start_time
#         failed_count = sum(1 for r in results if r and r.get("error"))
#         timeout_retry_count = sum(1 for r in results if r and r.get("error") and "timed out after" in r.get("error", ""))
#         non_timeout_error_count = failed_count - timeout_retry_count
#         success_rate = ((len(queries) - failed_count) / len(queries)) * 100 if queries else 0
        
#         logger.info(_format_log_message(f"✅ Completed in {elapsed_time:.2f}s | 📈 Rate: {len(queries) / elapsed_time:.1f} q/s | "
#                    f"🎯 Success: {success_rate:.1f}% | "
#                    f"⏰ Timeout failures: {timeout_retry_count} | "
#                    f"❌ Other failures: {non_timeout_error_count}", self.config))
        
#         return {"results": results}
    
#     async def close(self):
#         """Close the connection pool."""
#         if self.pool:
#             self.pool.close()
#             await self.pool.wait_closed()
#             self.pool = None
    
#     async def __aenter__(self):
#         await self._ensure_pool()
#         return self
    
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         await self.close()

# # Global async executor management (per-event-loop cache)
# _global_async_executors = {}
# _async_executor_lock = None

# def _get_or_create_lock():
#     """Get or create an asyncio lock for the current event loop."""
#     global _async_executor_lock
#     try:
#         # Check if current event loop is the same as the lock's loop
#         current_loop = asyncio.get_running_loop()
#         if _async_executor_lock is None or _async_executor_lock._loop is not current_loop:
#             _async_executor_lock = asyncio.Lock()
#         return _async_executor_lock
#     except RuntimeError:
#         # No event loop running, create a new lock
#         if _async_executor_lock is None:
#             _async_executor_lock = asyncio.Lock()
#         return _async_executor_lock

# async def get_global_async_executor(config: Optional[ODBCConfig] = None) -> AsyncODBCSPARQLExecutor:
#     """Get or create an async ODBC SPARQL executor bound to the current event loop."""
#     global _global_async_executors
#     lock = _get_or_create_lock()
#     current_loop = asyncio.get_running_loop()
#     key = id(current_loop)
#     async with lock:
#         executor = _global_async_executors.get(key)
#         if executor is None:
#             logger.info(_format_log_message("Creating a new per-loop AsyncODBCSPARQLExecutor instance.", config))
#             executor = AsyncODBCSPARQLExecutor(config)
#             _global_async_executors[key] = executor
#         else:
#             # Refresh experiment_name/current_step on the cached executor for logging
#             if config is not None:
#                 try:
#                     if getattr(config, "experiment_name", None) is not None:
#                         executor.config.experiment_name = config.experiment_name
#                     if getattr(config, "current_step", None) is not None:
#                         executor.config.current_step = config.current_step
#                 except Exception:
#                     pass
#     return executor

# async def execute_sparql_aioodbc(queries: List[str], config: Optional[ODBCConfig] = None, contexts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
#     """Execute SPARQL queries using a global async aioodbc executor with a reused connection pool."""
#     # Reuse a global executor bound to the current event loop to avoid repeatedly creating/closing pools
#     executor = await get_global_async_executor(config)
#     # Ensure the pool is initialized once and reused
#     await executor._ensure_pool()
#     return await executor.execute_batch(queries, contexts)