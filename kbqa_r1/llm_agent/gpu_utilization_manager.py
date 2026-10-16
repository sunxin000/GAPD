# -*- coding: utf-8 -*-
"""
GPU Utilization Manager (Ray-Compatible Version)

A robust manager to maintain GPU utilization during I/O-bound or CPU-bound
operations in distributed training environments (e.g., PyTorch DDP, Ray Actors).

This version is specifically designed to work with Ray workers and handles
CUDA environment issues that commonly occur in Ray distributed environments.
"""
import torch
import os
import logging
import time
import multiprocessing as mp
from multiprocessing import Process, Event
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

# --- 1. 配置日志 ---
# 建议在使用此模块的应用的主入口处配置日志
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))  # Default to INFO for S-Expression logs


# --- 2. GPU环境检测和初始化函数 ---
def _diagnose_cuda_environment():
    """
    详细诊断CUDA环境状态，提供详细的调试信息
    """
    diagnosis = {
        'torch_cuda_available': torch.cuda.is_available(),
        'torch_cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'torch_cuda_current_device': torch.cuda.current_device() if torch.cuda.is_available() else -1,
        'env_vars': {
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
            'LOCAL_RANK': os.environ.get('LOCAL_RANK', 'not set'),
            'RAY_LOCAL_RANK': os.environ.get('RAY_LOCAL_RANK', 'not set'),
            'WG_BACKEND': os.environ.get('WG_BACKEND', 'not set'),
            'RANK': os.environ.get('RANK', 'not set'),
            'WORLD_SIZE': os.environ.get('WORLD_SIZE', 'not set'),
        },
        'torch_distributed_initialized': torch.distributed.is_initialized() if hasattr(torch, 'distributed') else False,
        'is_ray_environment': 'RAY_LOCAL_RANK' in os.environ or 'WG_BACKEND' in os.environ,
    }
    
    if torch.cuda.is_available():
        try:
            diagnosis['torch_cuda_device_name'] = torch.cuda.get_device_name()
            diagnosis['torch_cuda_memory_allocated'] = torch.cuda.memory_allocated()
            diagnosis['torch_cuda_memory_reserved'] = torch.cuda.memory_reserved()
        except Exception as e:
            diagnosis['torch_cuda_device_name'] = f"Error: {e}"
            diagnosis['torch_cuda_memory_allocated'] = -1
            diagnosis['torch_cuda_memory_reserved'] = -1
    
    return diagnosis


def _ensure_cuda_environment():
    """
    Ensure CUDA environment is properly initialized, especially for Ray workers.
    This function handles common CUDA initialization issues in distributed environments.
    """
    # 首先进行详细诊断
    diagnosis = _diagnose_cuda_environment()
    logger.info(f"[CUDA-Diagnosis] Environment diagnosis: {diagnosis}")
    
    # 检查是否在Ray环境中
    is_ray_environment = diagnosis['is_ray_environment']
    
    if is_ray_environment:
        logger.info("[CUDA-Init] Detected Ray environment, attempting CUDA initialization...")
        
        # 尝试设置CUDA_VISIBLE_DEVICES如果未设置
        if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ['CUDA_VISIBLE_DEVICES'] == 'not set':
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
            logger.info(f"[CUDA-Init] Set CUDA_VISIBLE_DEVICES={local_rank} for Ray worker")
        
        # 尝试初始化CUDA上下文
        try:
            # 强制重新检查CUDA可用性
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"[CUDA-Init] CUDA available with {device_count} devices")
                
                # 设置当前设备
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                if device_count > 0:
                    torch.cuda.set_device(local_rank % device_count)
                    logger.info(f"[CUDA-Init] Set CUDA device to {local_rank % device_count}")
                    
                    # 验证设备设置是否成功
                    current_device = torch.cuda.current_device()
                    logger.info(f"[CUDA-Init] Current CUDA device confirmed: {current_device}")
                    
                    # 尝试一个简单的GPU操作来验证
                    try:
                        test_tensor = torch.randn(10, 10).cuda()
                        test_result = torch.mm(test_tensor, test_tensor)
                        logger.info(f"[CUDA-Init] GPU operation test successful: {test_result.sum().item():.2f}")
                        return True
                    except Exception as e:
                        logger.error(f"[CUDA-Init] GPU operation test failed: {e}")
                        return False
                else:
                    logger.warning("[CUDA-Init] CUDA available but device count is 0")
                    return False
            else:
                logger.warning("[CUDA-Init] CUDA not available after environment setup")
                
                # 尝试更激进的CUDA初始化
                logger.info("[CUDA-Init] Attempting aggressive CUDA initialization...")
                try:
                    # 尝试手动设置CUDA设备
                    if 'LOCAL_RANK' in os.environ:
                        local_rank = int(os.environ['LOCAL_RANK'])
                        # 尝试直接访问CUDA设备
                        device = torch.device(f'cuda:{local_rank}')
                        test_tensor = torch.randn(10, 10, device=device)
                        logger.info(f"[CUDA-Init] Aggressive initialization successful on device {local_rank}")
                        return True
                except Exception as e:
                    logger.error(f"[CUDA-Init] Aggressive initialization failed: {e}")
                
                return False
        except Exception as e:
            logger.error(f"[CUDA-Init] Error initializing CUDA: {e}")
            return False
    else:
        # 非Ray环境，直接检查CUDA可用性
        logger.info("[CUDA-Init] Non-Ray environment detected")
        return torch.cuda.is_available()


def _detect_cuda_from_distributed_training():
    """
    检测分布式训练框架是否已经初始化了CUDA环境
    """
    try:
        # 检查分布式训练是否已初始化
        if hasattr(torch, 'distributed') and torch.distributed.is_initialized():
            logger.info("[CUDA-Detect] Distributed training is initialized")
            
            # 检查是否有CUDA设备可用
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device_count = torch.cuda.device_count()
                logger.info(f"[CUDA-Detect] CUDA available: device {current_device}/{device_count}")
                return True, current_device
            else:
                logger.warning("[CUDA-Detect] Distributed training initialized but CUDA not available")
                return False, -1
        else:
            logger.info("[CUDA-Detect] Distributed training not initialized yet")
            return False, -1
            
    except Exception as e:
        logger.error(f"[CUDA-Detect] Error detecting CUDA from distributed training: {e}")
        return False, -1


def _wait_for_cuda_environment(max_wait_time=30):
    """
    等待CUDA环境被分布式训练框架初始化
    """
    logger.info(f"[CUDA-Wait] Waiting up to {max_wait_time}s for CUDA environment...")
    
    wait_interval = 1  # 每秒检查一次
    waited_time = 0
    
    while waited_time < max_wait_time:
        # 检查分布式训练是否初始化
        if hasattr(torch, 'distributed') and torch.distributed.is_initialized():
            logger.info("[CUDA-Wait] Distributed training initialized")
            
            # 检查CUDA是否可用
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device_count = torch.cuda.device_count()
                logger.info(f"[CUDA-Wait] CUDA environment ready: device {current_device}/{device_count}")
                return True, current_device
            else:
                logger.warning("[CUDA-Wait] Distributed training initialized but CUDA still not available")
        
        time.sleep(wait_interval)
        waited_time += wait_interval
        
        if waited_time % 5 == 0:  # 每5秒打印一次状态
            logger.info(f"[CUDA-Wait] Waited {waited_time}s... torch.distributed.is_initialized(): {torch.distributed.is_initialized() if hasattr(torch, 'distributed') else False}")
    
    logger.warning(f"[CUDA-Wait] Timeout after {max_wait_time}s, CUDA environment not ready")
    return False, -1


def _get_effective_gpu_device_id():
    """
    Get the effective GPU device ID considering Ray environment variables and distributed training.
    This function prioritizes the distributed training framework's device selection.
    """
    # 优先使用分布式训练框架的当前设备
    if hasattr(torch, 'distributed') and torch.distributed.is_initialized():
        try:
            current_device = torch.cuda.current_device()
            logger.info(f"[Device-Mapping] Using distributed training device: {current_device}")
            return current_device
        except Exception as e:
            logger.warning(f"[Device-Mapping] Failed to get distributed device: {e}")
    
    # 在Ray环境中，LOCAL_RANK通常对应GPU设备ID
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device_count = torch.cuda.device_count()
        if device_count > 0:
            effective_device_id = local_rank % device_count
            logger.info(f"[Device-Mapping] LOCAL_RANK={local_rank} -> GPU device {effective_device_id}")
            return effective_device_id
    
    # 回退到默认逻辑
    return int(os.environ.get('LOCAL_RANK', 0))


# --- 3. 独立的工作进程函数 ---
# 必须定义在顶层，以便 'spawn' 模式能够找到并序列化它
def gpu_maintenance_worker(config: 'GPUUtilizationConfig', gpu_device_id: int, shutdown_event: Event):
    """
    This function runs in a separate process.
    It initializes its own GPU environment and performs computations
    until the shutdown_event is set.
    """
    process_name = f"[GPU-Worker(PID:{os.getpid()})]"
    logger.info(f"{process_name} Process started, assigned to physical GPU mapped to local device {gpu_device_id}.")

    gpu_matrices: Optional[Dict[str, torch.Tensor]] = None

    try:
        # 在子进程中验证并初始化GPU环境
        if not _ensure_cuda_environment():
            logger.warning(f"{process_name} CUDA not available in this child process. Exiting.")
            return
        
        # 获取有效的设备ID
        effective_device_id = _get_effective_gpu_device_id()
        
        # device_count 在 spawn 的子进程中也可能因为父进程的环境变量而受限
        device_count = torch.cuda.device_count()
        if effective_device_id >= device_count:
            logger.warning(f"{process_name} Invalid device ID {effective_device_id}. Max is {device_count-1}. Exiting.")
            return

        device = torch.device(f'cuda:{effective_device_id}')
        torch.cuda.set_device(device)
        matrix_size = config.gpu_matrix_size

        gpu_matrices = {
            'A': torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32),
            'B': torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32),
        }
        # 预热
        _ = torch.mm(gpu_matrices['A'], gpu_matrices['B'])
        torch.cuda.synchronize(device)
        logger.info(f"{process_name} Initialized GPU workload on {device}.")

        # 循环执行GPU计算
        while not shutdown_event.is_set():
            with torch.no_grad():
                result = torch.mm(gpu_matrices['A'], gpu_matrices['B'])
                # 轻微更新矩阵以防止被编译器完全优化掉
                gpu_matrices['A'].add_(result, alpha=0.001).mul_(0.999)
            
            torch.cuda.synchronize(device)
            # 短暂休眠，让出CPU资源，避免空转消耗过多CPU
            time.sleep(0.01)

    except Exception as e:
        logger.error(f"{process_name} An error occurred: {e}", exc_info=True)
    finally:
        # 清理资源
        if gpu_matrices:
            del gpu_matrices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"{process_name} Process shutting down. Resources cleaned up.")


# --- 4. 配置类 ---
@dataclass
class GPUUtilizationConfig:
    """Configuration for GPU utilization management"""
    enable_gpu_utilization_maintenance: bool = True
    gpu_matrix_size: int = 4096  # 使用较大尺寸以确保在监控中能观察到显著的利用率
    force_cuda_initialization: bool = True  # 强制尝试初始化CUDA环境
    aggressive_cuda_init: bool = True  # 使用激进的CUDA初始化策略


# --- 5. 管理器类 ---
class GPUUtilizationManager:
    """
    Manages GPU utilization by launching a separate process for maintenance tasks,
    thus bypassing the main process's GIL and ensuring CUDA safety.

    Intended for distributed scenarios where one main process is responsible for one GPU.
    Enhanced to work with Ray distributed environments.
    """
    def __init__(self, config: GPUUtilizationConfig):
        self.config = config
        self.maintenance_process: Optional[Process] = None
        self.shutdown_event = mp.Event()
        self.is_maintenance_running = False
        
        # 延迟CUDA检查的标志
        self._cuda_checked = False
        self._cuda_available = False
        self._gpu_device_id = None

        # 不在初始化时检查CUDA，而是延迟到实际使用时
        logger.info("[Manager] GPUUtilizationManager initialized (CUDA check deferred)")
        
    def _ensure_cuda_available(self):
        """
        延迟检查CUDA可用性，确保在正确的时机进行CUDA环境检测
        """
        if not self._cuda_checked:
            logger.info("[Manager] Performing delayed CUDA availability check...")
            
            # 首先进行详细诊断
            diagnosis = _diagnose_cuda_environment()
            logger.info(f"[Manager] CUDA environment diagnosis: {diagnosis}")
            
            # 首先尝试检测分布式训练框架是否已经初始化了CUDA环境
            cuda_available, detected_device = _detect_cuda_from_distributed_training()
            
            if cuda_available:
                # 分布式训练框架已经初始化了CUDA环境
                logger.info(f"[Manager] CUDA environment detected from distributed training: device {detected_device}")
                self._cuda_available = True
                self._gpu_device_id = detected_device
                self._cuda_checked = True
                return True
            
            # 如果分布式训练还没有初始化，等待它初始化
            if hasattr(torch, 'distributed') and not torch.distributed.is_initialized():
                logger.info("[Manager] Waiting for distributed training to initialize CUDA environment...")
                cuda_available, detected_device = _wait_for_cuda_environment(max_wait_time=30)
                
                if cuda_available:
                    logger.info(f"[Manager] CUDA environment ready after waiting: device {detected_device}")
                    self._cuda_available = True
                    self._gpu_device_id = detected_device
                    self._cuda_checked = True
                    return True
            
            # 如果等待后仍然没有CUDA环境，尝试传统的检测方法
            self._cuda_available = torch.cuda.is_available()
            self._cuda_checked = True
            
            if self._cuda_available:
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    self._gpu_device_id = _get_effective_gpu_device_id()
                    logger.info(f"[Manager] CUDA available through traditional detection. Using GPU device {self._gpu_device_id}")
                else:
                    logger.warning("torch.cuda.device_count() is 0. GPU maintenance will be disabled.")
                    self._cuda_available = False
            else:
                logger.warning("Torch reports CUDA is not available. GPU maintenance will be disabled.")
                
                # 如果启用了激进初始化，尝试更多方法
                if self.config.aggressive_cuda_init:
                    logger.info("[Manager] Attempting aggressive CUDA initialization...")
                    try:
                        # 尝试直接设置CUDA设备
                        if 'LOCAL_RANK' in os.environ:
                            local_rank = int(os.environ['LOCAL_RANK'])
                            device = torch.device(f'cuda:{local_rank}')
                            test_tensor = torch.randn(10, 10, device=device)
                            logger.info(f"[Manager] Aggressive initialization successful on device {local_rank}")
                            self._cuda_available = True
                            self._gpu_device_id = local_rank
                    except Exception as e:
                        logger.error(f"[Manager] Aggressive initialization failed: {e}")
        
        return self._cuda_available
    
    @property
    def gpu_device_id(self):
        """获取GPU设备ID，如果还没有检查CUDA则先检查"""
        if not self._cuda_checked:
            self._ensure_cuda_available()
        return self._gpu_device_id if self._cuda_available else -1
        
    @contextmanager
    def maintain_utilization_context(self):
        """
        A context manager that starts a maintenance process on entry
        and stops it on exit.
        """
        if not self.config.enable_gpu_utilization_maintenance:
            # 如果被禁用，则什么也不做
            yield
            return
        
        # 延迟检查CUDA可用性
        if not self._ensure_cuda_available():
            logger.info("[Manager] CUDA not available, skipping GPU maintenance")
            yield
            return
        
        try:
            self._start_maintenance()
            yield
        finally:
            self._stop_maintenance()
            
    def _start_maintenance(self):
        """Creates and starts the GPU maintenance child process."""
        if self.is_maintenance_running:
            logger.warning("[Manager] Maintenance process is already running.")
            return

        logger.info(f"[Manager] Starting GPU maintenance process for local device {self._gpu_device_id}...")
        self.shutdown_event.clear()
        
        self.maintenance_process = Process(
            target=gpu_maintenance_worker,
            args=(self.config, self._gpu_device_id, self.shutdown_event),
            name=f"GPU-Worker-for-dev{self._gpu_device_id}",
            daemon=True
        )
        self.maintenance_process.start()
        self.is_maintenance_running = True
        logger.info(f"[Manager] Worker process (PID: {self.maintenance_process.pid}) has been started.")

    def _stop_maintenance(self):
        """Stops and cleans up the GPU maintenance child process."""
        if not self.is_maintenance_running or self.maintenance_process is None:
            return

        logger.info(f"[Manager] Stopping GPU maintenance process (PID: {self.maintenance_process.pid})...")
        
        self.shutdown_event.set()
        self.maintenance_process.join(timeout=5.0)
        
        if self.maintenance_process.is_alive():
            logger.error(f"[Manager] Worker process did not terminate gracefully. Forcing termination.")
            self.maintenance_process.terminate()
        else:
            logger.info(f"[Manager] Worker process has stopped gracefully.")
            
        self.maintenance_process.close()
        self.maintenance_process = None
        self.is_maintenance_running = False

    def shutdown(self):
        """Ensures the child process is stopped upon object destruction."""
        self._stop_maintenance()

    def __del__(self):
        self.shutdown()


# --- 6. 示例用法 ---
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 重要提示: 为了保证CUDA在子进程中的安全，必须在使用 multiprocessing
    # 启动任何进程之前，设置启动方式为 'spawn'。
    # 这行代码应该放在您应用的主入口脚本的顶层。
    # -------------------------------------------------------------------------
    try:
        mp.set_start_method('spawn', force=True)
        print("[Main] Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("[Main] Multiprocessing start method was already set.")

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("[Main] No CUDA GPUs available on this system. Exiting demo.")
        exit()

    print("="*60)
    print("DEMO: GPUUtilizationManager in a simulated single-process environment")
    print("This simulates a scenario like PyTorch DDP where LOCAL_RANK is set.")
    print("="*60)
    
    # 在单脚本演示中，手动设置 LOCAL_RANK 环境变量来模拟分布式环境
    # 您可以修改这个值来测试不同的GPU
    target_gpu_rank = "0"
    os.environ['LOCAL_RANK'] = target_gpu_rank
    print(f"[Main] Simulating a distributed worker by setting LOCAL_RANK='{target_gpu_rank}'.")
    
    config = GPUUtilizationConfig()
    gpu_manager = GPUUtilizationManager(config)

    def mock_long_io_operation():
        """模拟一个长时间的、会阻塞主进程的I/O或CPU操作。"""
        print(f"\n[Main] ---> Starting a 20-second mock I/O operation...")
        print(f"[Main] ---> The manager should now be keeping local GPU {gpu_manager.gpu_device_id} busy.")
        print(f"[Main] ---> Please run 'nvidia-smi' in your terminal to observe GPU utilization.")
        time.sleep(20) 
        print("[Main] ---> Mock I/O operation finished.\n")

    try:
        # 使用上下文管理器来自动管理后台进程的生命周期
        with gpu_manager.maintain_utilization_context():
            mock_long_io_operation()
            
        print("[Main] Exited context manager. The background process should have stopped.")
        print("[Main] Waiting for 5 seconds to observe GPU utilization returning to 0...")
        time.sleep(5)
        print("[Main] Demo finished successfully.")

    except Exception as e:
        print(f"[Main] An error occurred during the demo: {e}")
    finally:
        # 确保在退出前彻底关闭子进程
        gpu_manager.shutdown()