#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent GPU Manager

This module manages an independent GPU maintenance training process
that runs outside of Ray's GPU environment sandbox.
"""

import os
import time
import signal
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import subprocess as sp

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))  # Default to INFO for S-Expression logs


@dataclass
class IndependentGPUConfig:
    """独立GPU维护配置"""
    gpu_id: int = 0  # 兼容性参数，实际会检测所有GPU
    matrix_size: int = 2048
    interval: float = 0.01
    pid_file: str = '/tmp/gpu_maintenance.pid'
    script_path: str = 'scripts/gpu_maintenance_train.py'
    enable_gpu_utilization_maintenance: bool = True
    # 多GPU配置
    target_memory_percentage: float = 0.80  # 目标内存占用百分比
    idle_gpu_threshold: int = 20  # 空闲GPU利用率阈值
    compute_matrix_size: int = 4096  # 计算矩阵大小
    status_update_interval: int = 5  # 状态更新间隔


class IndependentGPUMaintenanceManager:
    """独立GPU维护管理器"""
    
    def __init__(self, config: IndependentGPUConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        
        # 确保脚本路径是绝对路径
        if not os.path.isabs(config.script_path):
            # 相对于项目根目录
            project_root = Path(__file__).parent.parent.parent
            self.script_path = str(project_root / config.script_path)
        else:
            self.script_path = config.script_path
        
        # 使用shell脚本启动器
        self.shell_script_path = str(Path(self.script_path).parent / "start_gpu_maintenance.sh")
        
        logger.info(f"[Independent-GPU] Initialized with script: {self.script_path}")
        logger.info(f"[Independent-GPU] Shell launcher: {self.shell_script_path}")
    
    def start_maintenance(self) -> bool:
        """启动GPU维护进程"""
        if self.is_running:
            logger.warning("[Independent-GPU] GPU maintenance process is already running")
            return True
        
        try:
            # 检查shell脚本是否存在
            if not os.path.exists(self.shell_script_path):
                logger.error(f"[Independent-GPU] Shell script not found: {self.shell_script_path}")
                return False
            
            # 确保shell脚本有执行权限
            os.chmod(self.shell_script_path, 0o755)
            
            # 构建shell命令 - 传递多GPU配置参数
            cmd = [
                '/bin/bash',
                self.shell_script_path,
                str(self.config.gpu_id),  # 兼容性参数
                str(self.config.matrix_size),  # 兼容性参数
                str(self.config.interval),  # 兼容性参数
                self.config.pid_file,
                f"{self.config.pid_file}.log",
                # 多GPU配置参数
                str(self.config.target_memory_percentage),
                str(self.config.idle_gpu_threshold),
                str(self.config.compute_matrix_size),
                str(self.config.status_update_interval)
            ]
            
            logger.info(f"[Independent-GPU] Starting GPU maintenance process with shell script: {' '.join(cmd)}")
            
            # 启动shell脚本
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            # 等待进程启动
            time.sleep(2)
            
            # 检查进程是否还在运行
            if self.process.poll() is None:
                self.is_running = True
                logger.info(f"[Independent-GPU] GPU maintenance process started (PID: {self.process.pid})")
                return True
            else:
                # 进程已经退出
                stdout, stderr = self.process.communicate()
                logger.error(f"[Independent-GPU] Process failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"[Independent-GPU] Failed to start GPU maintenance process: {e}")
            return False
    
    def stop_maintenance(self) -> bool:
        """停止GPU维护进程 - 快速版本"""
        if not self.is_running:
            logger.info("[Independent-GPU] GPU maintenance process is not running")
            return True
        
        try:
            logger.info("[Independent-GPU] Stopping GPU maintenance process...")
            
            # 立即设置状态为False，防止重复调用
            self.is_running = False
            
            # 快速终止策略：同时使用多种方法，不等待验证
            pids_to_kill = set()
            
            # 1. 从PID文件获取PID
            if os.path.exists(self.config.pid_file):
                try:
                    with open(self.config.pid_file, 'r') as f:
                        pid = int(f.read().strip())
                        pids_to_kill.add(pid)
                except Exception:
                    pass
            
            # 2. 从子进程获取PID
            if self.process:
                pids_to_kill.add(self.process.pid)
            
            # 3. 快速强制终止所有相关进程
            for pid in pids_to_kill:
                try:
                    # 直接使用SIGKILL，不等待
                    os.kill(pid, signal.SIGKILL)
                    logger.debug(f"[Independent-GPU] Killed PID {pid}")
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.debug(f"[Independent-GPU] Error killing PID {pid}: {e}")
                
                # 尝试杀进程组
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                    logger.debug(f"[Independent-GPU] Killed PGID {pgid}")
                except Exception:
                    pass
            
            # 4. 强制终止子进程（如果存在）
            if self.process:
                try:
                    self.process.kill()
                except Exception:
                    pass
            
            # 5. 全局清理（不等待）
            try:
                sp.run(["pkill", "-9", "-f", "gpu_maintenance_train.py"], 
                      stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=1)
            except Exception:
                pass
            
            # 6. 清理资源
            try:
                if os.path.exists(self.config.pid_file):
                    os.remove(self.config.pid_file)
            except Exception:
                pass
            
            # 清理子进程引用
            self.process = None
            
            # 7. 快速清理GPU内存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("[Independent-GPU] GPU cache cleared")
            except Exception:
                pass
            
            logger.info("[Independent-GPU] GPU maintenance process stopped")
            return True
            
        except Exception as e:
            logger.error(f"[Independent-GPU] Failed to stop GPU maintenance process: {e}")
            return False
    
    def is_maintenance_running(self) -> bool:
        """检查GPU维护进程是否在运行"""
        if not self.is_running:
            return False
        
        # 检查PID文件
        if os.path.exists(self.config.pid_file):
            try:
                with open(self.config.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # 检查进程是否存在
                os.kill(pid, 0)
                return True
            except (OSError, ValueError):
                pass
        
        # 检查子进程
        if self.process and self.process.poll() is None:
            return True
        
        # 进程已经结束
        self.is_running = False
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取GPU维护状态"""
        status = {
            'is_running': self.is_running,
            'process_pid': self.process.pid if self.process else None,
            'pid_file_exists': os.path.exists(self.config.pid_file),
            'config': {
                'gpu_id': self.config.gpu_id,
                'matrix_size': self.config.matrix_size,
                'interval': self.config.interval,
                'target_memory_percentage': self.config.target_memory_percentage,
                'idle_gpu_threshold': self.config.idle_gpu_threshold,
                'compute_matrix_size': self.config.compute_matrix_size,
                'status_update_interval': self.config.status_update_interval
            }
        }
        
        if os.path.exists(self.config.pid_file):
            try:
                with open(self.config.pid_file, 'r') as f:
                    status['pid_file_pid'] = int(f.read().strip())
            except Exception:
                status['pid_file_pid'] = None
        
        return status
    
    def __enter__(self):
        """上下文管理器入口"""
        if self.config.enable_gpu_utilization_maintenance:
            self.start_maintenance()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_maintenance()


# 便捷函数
def create_independent_gpu_manager(
    gpu_id: int = 0, 
    matrix_size: int = 2048, 
    interval: float = 0.01,
    target_memory_percentage: float = 0.80,
    idle_gpu_threshold: int = 20,
    compute_matrix_size: int = 4096,
    status_update_interval: int = 5
) -> IndependentGPUMaintenanceManager:
    """创建独立的GPU维护管理器"""
    config = IndependentGPUConfig(
        gpu_id=gpu_id,
        matrix_size=matrix_size,
        interval=interval,
        target_memory_percentage=target_memory_percentage,
        idle_gpu_threshold=idle_gpu_threshold,
        compute_matrix_size=compute_matrix_size,
        status_update_interval=status_update_interval
    )
    return IndependentGPUMaintenanceManager(config)
