"""
SPARQL Execution Manager - Handles all SPARQL query execution logic.
"""
# import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from tqdm import tqdm

# Import ODBC modules
try:
    from .odbc_config import ODBCConfig
    from .odbc_executor import execute_sparql_odbc
    ODBC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ODBC modules not available: {e}")
    ODBC_AVAILABLE = False

# Import async ODBC modules
try:
    from .aioodbc_executor import execute_sparql_aioodbc
    AIOODBC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: aioodbc modules not available: {e}")
    AIOODBC_AVAILABLE = False

# Validator disabled per requirement

@dataclass
class SPARQLConfig:
    """Configuration for SPARQL execution."""
    sparql_url: Optional[str] = None
    sparql_batch_size: int = 128  # For HTTP API batch requests
    sparql_max_concurrent: int = 16  # For HTTP API concurrent requests
    use_odbc: bool = True  # Use synchronous ODBC (more stable than aioodbc)
    use_aioodbc: bool = False  # Disabled: aioodbc has transaction management bugs causing segfaults
    odbc_config: Optional[dict] = None

class SPARQLExecutionManager:
    """Manager for executing SPARQL queries using HTTP API or ODBC connection."""
    
    def __init__(self, config: SPARQLConfig):
        self.config = config
        # Create a dedicated background event loop for async execution (aiohttp/aioodbc)
        # so that all threads submit to the same loop and share the same aioodbc pool
        self._bg_loop = None
        self._bg_thread = None
        self._bg_lock = threading.Lock()

    # def _ensure_background_loop(self):
    #     """Ensure a single long-lived background event loop is running."""
    #     if self._bg_loop is not None and self._bg_thread is not None and self._bg_thread.is_alive():
    #         return
    #     with self._bg_lock:
    #         if self._bg_loop is not None and self._bg_thread is not None and self._bg_thread.is_alive():
    #             return
    #         loop = asyncio.new_event_loop()
    #         def _runner():
    #             asyncio.set_event_loop(loop)
    #             loop.run_forever()
    #         thread = threading.Thread(target=_runner, name="sparql-bg-loop", daemon=True)
    #         thread.start()
    #         self._bg_loop = loop
    #         self._bg_thread = thread

    def _run_coro_in_background(self, coro):
        """Submit a coroutine to the background loop and wait for result synchronously."""
        self._ensure_background_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return future.result()
    
    def execute_batch(self, queries: List[str] = None, contexts: List[Dict[str, Any]] = None) -> Dict:
        """Execute a batch of SPARQL queries."""
        if not queries:
            return {"results": []}

        # Validation disabled: pass through all queries
        valid_queries = list(queries)
        all_results = [None] * len(queries)
        is_valid_mask = [True] * len(queries)
        
        # Choose execution method - prefer synchronous ODBC for stability
        if self.config.use_odbc or not self.config.sparql_url:
            # Use synchronous ODBC executor (most stable, no segfault issues)
            execution_results = self._batch_sparql_odbc(valid_queries)
        # elif self.config.use_aioodbc and AIOODBC_AVAILABLE:
        #     # aioodbc is deprecated due to Virtuoso ODBC driver transaction bugs
        #     print("⚠️  Warning: aioodbc may cause segfaults with Virtuoso. Consider using synchronous ODBC.")
        #     # Submit to the shared background loop to ensure single aioodbc pool reuse across threads
        #     execution_results = self._run_coro_in_background(self._batch_sparql_aioodbc(valid_queries, contexts))
        # else:
        #     execution_results = self._batch_sparql_http(valid_queries)

        # Merge results
        valid_idx = 0
        for i, is_valid in enumerate(is_valid_mask):
            if is_valid:
                if valid_idx < len(execution_results.get("results", [])):
                    all_results[i] = execution_results["results"][valid_idx]
                    valid_idx += 1
                else:
                    all_results[i] = {"query": queries[i], "error": "Missing result from execution."}
        
        return {"results": all_results}
    
    # async def execute_batch_async(self, queries: List[str] = None, contexts: List[Dict[str, Any]] = None) -> Dict:
    #     """Execute a batch of SPARQL queries asynchronously."""
    #     if not queries:
    #         return {"results": []}

    #     # Validation disabled: pass through all queries
    #     valid_queries = list(queries)
    #     all_results = [None] * len(queries)
    #     is_valid_mask = [True] * len(queries)
        
    #     # Choose execution method - prefer synchronous ODBC even in async context
    #     if self.config.use_odbc or not self.config.sparql_url:
    #         # Run sync ODBC in executor to not block the event loop
    #         loop = asyncio.get_running_loop()
    #         execution_results = await loop.run_in_executor(None, self._batch_sparql_odbc, valid_queries)
    #     elif self.config.use_aioodbc and AIOODBC_AVAILABLE:
    #         # aioodbc is deprecated due to Virtuoso ODBC driver transaction bugs
    #         print("⚠️  Warning: aioodbc may cause segfaults with Virtuoso. Consider using synchronous ODBC.")
    #         execution_results = await self._batch_sparql_aioodbc(valid_queries, contexts)
    #     elif not self.config.use_odbc and self.config.sparql_url:
    #         execution_results = await self._async_batch_sparql(valid_queries)
    #     else:
    #         # Final fallback to sync ODBC
    #         loop = asyncio.get_running_loop()
    #         execution_results = await loop.run_in_executor(None, self._batch_sparql_odbc, valid_queries)
        
    #     # Merge results
    #     valid_idx = 0
    #     for i, is_valid in enumerate(is_valid_mask):
    #         if is_valid:
    #             if valid_idx < len(execution_results.get("results", [])):
    #                 all_results[i] = execution_results["results"][valid_idx]
    #                 valid_idx += 1
    #             else:
    #                 all_results[i] = {"query": queries[i], "error": "Missing result from execution."}

    #     return {"results": all_results}
    
    # def _batch_sparql_http(self, queries: List[str]) -> Dict:
    #     """Execute a batch of SPARQL queries using the SPARQL execution server with concurrent processing."""
    #     if not self.config.sparql_url:
    #         error_msg = "SPARQL URL is not configured. Please set the 'sparql.url' parameter."
    #         print(f"Error: {error_msg}")
    #         return {"results": [{"query": q, "error": error_msg} for q in queries]}
        
    #     # 使用异步方法处理批量查询，通过共享背景事件循环以复用资源
    #     return self._run_coro_in_background(self._async_batch_sparql(queries))
    
    # async def _async_batch_sparql(self, queries: List[str]) -> Dict:
    #     """Asynchronously execute SPARQL queries in batches with concurrent processing."""
    #     if not queries:
    #         return {"results": []}
        
    #     print(f"🚀 Processing {len(queries)} SPARQL queries with batch_size={self.config.sparql_batch_size}, max_concurrent={self.config.sparql_max_concurrent}")
    #     start_time = time.time()
        
    #     # 将查询分批
    #     batches = []
    #     batch_size = self.config.sparql_batch_size
    #     for i in range(0, len(queries), batch_size):
    #         batch_queries = queries[i:i + batch_size]
    #         batches.append((i, batch_queries))
        
    #     print(f"📦 Split into {len(batches)} batches")
        
    #     # 创建总体进度条
    #     total_queries = len(queries)
    #     progress_bar = tqdm(
    #         total=total_queries,
    #         desc="🔍 SPARQL Processing",
    #         unit="queries",
    #         unit_scale=True,
    #         colour="green",
    #         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    #     )
        
    #     # 使用信号量控制并发度
    #     semaphore = asyncio.Semaphore(self.config.sparql_max_concurrent)
    #     completed_queries = 0
        
    #     async def process_batch_with_progress(batch_idx, batch_queries):
    #         nonlocal completed_queries
    #         async with semaphore:
    #             result = await self._async_sparql_batch(batch_idx, batch_queries, progress_bar)
    #             # 更新进度条
    #             completed_count = len(batch_queries)
    #             progress_bar.update(completed_count)
    #             completed_queries += completed_count
                
    #             # 更新进度条描述，显示实时统计
    #             elapsed_time = time.time() - start_time
    #             if elapsed_time > 0:
    #                 rate = completed_queries / elapsed_time
    #                 progress_bar.set_description(f"🔍 SPARQL Processing (📈{rate:.1f} q/s)")
                
    #             return result
        
    #     # 并发处理所有批次
    #     tasks = [process_batch_with_progress(batch_idx, batch_queries) for batch_idx, batch_queries in batches]
    #     batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
    #     # 关闭进度条
    #     progress_bar.close()
        
    #     # 合并结果，保持原始顺序
    #     all_results = []
    #     failed_batches = 0
    #     for i, batch_result in enumerate(batch_results):
    #         if isinstance(batch_result, Exception):
    #             failed_batches += 1
    #             print(f"❌ Batch {i+1} processing error: {batch_result}")
    #             # 为这个批次创建错误结果
    #             batch_size = self.config.sparql_batch_size
    #             error_results = [{"query": "", "error": str(batch_result)} for _ in range(batch_size)]
    #             all_results.extend(error_results)
    #         else:
    #             all_results.extend(batch_result.get("results", []))
        
    #     # 确保结果数量匹配查询数量
    #     if len(all_results) > len(queries):
    #         all_results = all_results[:len(queries)]
    #     elif len(all_results) < len(queries):
    #         # 补充缺失的结果
    #         missing_count = len(queries) - len(all_results)
    #         for i in range(missing_count):
    #             query_idx = len(all_results) + i
    #             query = queries[query_idx] if query_idx < len(queries) else ""
    #             all_results.append({"query": query, "error": "Missing result"})
        
    #     elapsed_time = time.time() - start_time
    #     queries_per_second = len(queries) / elapsed_time if elapsed_time > 0 else 0
        
    #     print(f"✅ Completed {len(queries)} SPARQL queries in {elapsed_time:.2f} seconds")
    #     print(f"📊 Performance: {queries_per_second:.2f} queries/second")
    #     print(f"📦 Batches processed: {len(batches)} total, {failed_batches} failed")
    #     print(f"🎯 Results collected: {len(all_results)}")
        
    #     if failed_batches > 0:
    #         success_rate = ((len(batches) - failed_batches) / len(batches)) * 100
    #         print(f"📈 Success rate: {success_rate:.1f}%")
        
    #     return {"results": all_results}
    
    # async def _async_sparql_batch(self, batch_idx: int, batch_queries: List[str], progress_bar=None) -> Dict:
    #     """Execute a single batch of SPARQL queries."""
    #     max_retries = 3
    #     retry_delay = 1.0
        
    #     # 为每个批次创建子进度条（可选）
    #     # batch_desc = f"📦 Batch {batch_idx+1}"
        
    #     for attempt in range(max_retries):
    #         try:
    #             timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
    #             async with aiohttp.ClientSession(timeout=timeout) as session:
    #                 async with session.post(
    #                     self.config.sparql_url,
    #                     json={"queries": batch_queries}
    #                 ) as response:
    #                     if response.status != 200:
    #                         error_msg = f"SPARQL server returned status {response.status}"
    #                         response_text = await response.text()
    #                         print(f"Batch {batch_idx} error: {error_msg}, Response: {response_text[:500]}...")
                            
    #                         if attempt < max_retries - 1:
    #                             print(f"Retrying batch {batch_idx} (attempt {attempt + 2}/{max_retries})")
    #                             await asyncio.sleep(retry_delay * (attempt + 1))
    #                             continue
                            
    #                         return {"results": [{"query": q, "error": error_msg} for q in batch_queries]}
                        
    #                     response_json = await response.json()
    #                     results = response_json.get('results', [])
                        
    #                     # 处理结果中的错误
    #                     error_count = 0
    #                     for i, result in enumerate(results):
    #                         if isinstance(result, dict) and 'error' in result:
    #                             error_count += 1
    #                             if progress_bar:
    #                                 progress_bar.write(f"⚠️ Batch {batch_idx+1}, Query {i+1} error: {result['error']}")
    #                         elif isinstance(result, str) and ('Error' in result or 'error' in result.lower()):
    #                             error_count += 1
    #                             if progress_bar:
    #                                 progress_bar.write(f"⚠️ Batch {batch_idx+1}, Query {i+1} error: {result}")
    #                             results[i] = {"query": batch_queries[i] if i < len(batch_queries) else "", "error": result, "results": []}
                        
    #                     # 显示批次完成信息
    #                     success_count = len(results) - error_count
    #                     if progress_bar:
    #                         if error_count > 0:
    #                             progress_bar.write(f"✅ Batch {batch_idx+1} completed: {success_count} success, {error_count} errors")
    #                         else:
    #                             progress_bar.write(f"✅ Batch {batch_idx+1} completed: {len(results)} queries successful")
                        
    #                     return {"results": results}
                        
    #         except asyncio.TimeoutError:
    #             error_msg = f"Timeout error for batch {batch_idx+1}"
    #             if progress_bar:
    #                 progress_bar.write(f"⏰ {error_msg}")
    #             if attempt < max_retries - 1:
    #                 if progress_bar:
    #                     progress_bar.write(f"🔄 Retrying batch {batch_idx+1} due to timeout (attempt {attempt + 2}/{max_retries})")
    #                 await asyncio.sleep(retry_delay * (attempt + 1))
    #                 continue
    #             return {"results": [{"query": q, "error": error_msg} for q in batch_queries]}
    #         except Exception as e:
    #             error_msg = f"Error processing batch {batch_idx+1}: {str(e)}"
    #             if progress_bar:
    #                 progress_bar.write(f"❌ {error_msg}")
    #             if attempt < max_retries - 1:
    #                 if progress_bar:
    #                     progress_bar.write(f"🔄 Retrying batch {batch_idx+1} due to error (attempt {attempt + 2}/{max_retries})")
    #                 await asyncio.sleep(retry_delay * (attempt + 1))
    #                 continue
    #             return {"results": [{"query": q, "error": error_msg} for q in batch_queries]}
        
    #     # 如果所有重试都失败了
    #     return {"results": [{"query": q, "error": "All retry attempts failed"} for q in batch_queries]}

    def _batch_sparql_odbc(self, queries: List[str]) -> Dict:
        """Execute a batch of SPARQL queries using ODBC direct connection."""
        if not ODBC_AVAILABLE:
            error_msg = "ODBC modules not available. Please install pyodbc and configure ODBC driver."
            print(f"Error: {error_msg}")
            return {"results": [{"query": q, "error": error_msg} for q in queries]}
        
        try:
            # 创建ODBC配置
            odbc_config = None
            if self.config.odbc_config:
                # 从配置字典创建ODBCConfig对象，过滤掉Hydra的_target_参数
                config_dict = {k: v for k, v in self.config.odbc_config.items() if k != '_target_'}
                odbc_config = ODBCConfig(**config_dict)
            
            print(f"🚀 Using ODBC direct connection for {len(queries)} SPARQL queries")
            
            # 使用ODBC执行器执行查询
            results = execute_sparql_odbc(queries, odbc_config)
            
            return results
            
        except Exception as e:
            error_msg = f"ODBC execution failed: {str(e)}"
            print(f"Error: {error_msg}")
            return {"results": [{"query": q, "error": error_msg} for q in queries]}

    async def _batch_sparql_aioodbc(self, queries: List[str], contexts: List[Dict[str, Any]] = None) -> Dict:
        """Execute a batch of SPARQL queries using async aioodbc.
        
        ⚠️  WARNING: This method is deprecated due to Virtuoso ODBC driver bugs.
        The driver's SQLEndTran implementation has memory management issues that cause
        segmentation faults when handling query timeouts or errors.
        
        Use _batch_sparql_odbc() instead for stable execution.
        """
        if not AIOODBC_AVAILABLE:
            error_msg = "aioodbc modules not available. Please install aioodbc: pip install aioodbc"
            print(f"Error: {error_msg}")
            return {"results": [{"query": q, "error": error_msg} for q in queries]}
        
        print("⚠️  WARNING: Using aioodbc which may cause segfaults. Consider using synchronous ODBC.")
        
        try:
            # 创建ODBC配置
            odbc_config = None
            if self.config.odbc_config:
                # 从配置字典创建ODBCConfig对象，过滤掉Hydra的_target_参数
                config_dict = {k: v for k, v in self.config.odbc_config.items() if k != '_target_'}
                odbc_config = ODBCConfig(**config_dict)
            
            print(f"🚀 Using async aioodbc for {len(queries)} SPARQL queries")
            
            # 使用统一的异步执行器，传递context信息
            results = await execute_sparql_aioodbc(queries, odbc_config, contexts)
            
            return results
            
        except Exception as e:
            error_msg = f"Async ODBC execution failed: {str(e)}"
            print(f"Error: {error_msg}")
            return {"results": [{"query": q, "error": error_msg} for q in queries]}

    @staticmethod
    def results_to_string(results) -> str:
        """Convert SPARQL results to a string format."""
        if not results:
            return "No results found." #TODO no results found的话应该拆分query， 然后重新执行query看哪一步出现了问题。
        
        # 检查是否是错误结果
        if isinstance(results, dict) and "error" in results:
            return f"Error: {results['error']}"
        
        # 检查结果列表
        if isinstance(results, list):
            if len(results) > 0:
                if isinstance(results[0], dict) and "error" in results[0]:
                    return f"Error: {results[0]['error']}"
                
                # 正常结果处理
                result_str = ""
                for i, result in enumerate(results):
                    if isinstance(result, dict):
                        result_str += f"Result {i+1}:\n"
                        for var, value in result.items():
                            # Clean null characters from the value
                            if isinstance(value, str):
                                value = value.replace('\u0000', '').replace('\x00', '')
                            result_str += f"  {var}: {value}\n"
                    else:
                        clean_result = str(result).replace('\u0000', '').replace('\x00', '')
                        result_str += f"Result {i+1}: {clean_result}\n"
                
                return result_str
        
        # 如果是字符串（可能包含错误消息）
        if isinstance(results, str):
            clean_results = results.replace('\u0000', '').replace('\x00', '')
            if "Error" in clean_results or "error" in clean_results.lower():
                return f"Error: {clean_results}"
            return clean_results
        
        # 其他情况，尝试转换为字符串
        clean_str = str(results).replace('\u0000', '').replace('\x00', '')
        return clean_str 