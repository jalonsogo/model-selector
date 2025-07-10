"""
Hardware detection and system information gathering
"""

import asyncio
import json
import logging
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class CPUInfo:
    """CPU information"""
    name: str
    cores: int
    arch: str
    
    
@dataclass
class GPUInfo:
    """GPU information"""
    name: str
    vram: float
    vendor: str
    driver: Optional[str] = None
    note: Optional[str] = None


@dataclass
class MemoryInfo:
    """Memory information"""
    vram: float
    ram: float
    mode: str  # "cpu" or "gpu"
    note: Optional[str] = None


@dataclass
class SystemInfo:
    """Complete system information"""
    os: str
    arch: str
    cpu: CPUInfo
    ram: float
    gpus: List[GPUInfo] = field(default_factory=list)
    

class HardwareDetector:
    """Detects local hardware capabilities with improved error handling"""
    
    def __init__(self):
        self._system_info: Optional[SystemInfo] = None
        self._cache_file = Path.home() / ".model-selector" / "hardware_cache.json"
        self._cache_file.parent.mkdir(exist_ok=True)
        
    async def detect_system_info(self, use_cache: bool = True) -> SystemInfo:
        """Detects system information with caching"""
        if self._system_info and use_cache:
            return self._system_info
            
        # Try to load from cache
        if use_cache and self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Check if cache is recent (within 24 hours)
                    import time
                    if time.time() - cached_data.get('timestamp', 0) < 86400:
                        self._system_info = self._parse_cached_info(cached_data)
                        return self._system_info
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load hardware cache: {e}")
        
        # Detect hardware
        logger.info("Detecting hardware configuration...")
        
        # Run detection tasks concurrently
        cpu_task = asyncio.create_task(self._detect_cpu())
        ram_task = asyncio.create_task(self._detect_ram())
        gpu_task = asyncio.create_task(self._detect_gpus())
        
        cpu_info = await cpu_task
        ram = await ram_task
        gpus = await gpu_task
        
        self._system_info = SystemInfo(
            os=platform.system(),
            arch=platform.machine(),
            cpu=cpu_info,
            ram=ram,
            gpus=gpus
        )
        
        # Cache the results
        if use_cache:
            self._cache_hardware_info()
        
        return self._system_info
    
    def _parse_cached_info(self, cached_data: Dict) -> SystemInfo:
        """Parse cached hardware information"""
        return SystemInfo(
            os=cached_data['os'],
            arch=cached_data['arch'],
            cpu=CPUInfo(**cached_data['cpu']),
            ram=cached_data['ram'],
            gpus=[GPUInfo(**gpu) for gpu in cached_data['gpus']]
        )
    
    def _cache_hardware_info(self):
        """Cache hardware information to file"""
        try:
            import time
            cache_data = {
                'timestamp': time.time(),
                'os': self._system_info.os,
                'arch': self._system_info.arch,
                'cpu': {
                    'name': self._system_info.cpu.name,
                    'cores': self._system_info.cpu.cores,
                    'arch': self._system_info.cpu.arch
                },
                'ram': self._system_info.ram,
                'gpus': [
                    {
                        'name': gpu.name,
                        'vram': gpu.vram,
                        'vendor': gpu.vendor,
                        'driver': gpu.driver,
                        'note': gpu.note
                    }
                    for gpu in self._system_info.gpus
                ]
            }
            
            with open(self._cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to cache hardware info: {e}")
    
    async def _detect_cpu(self) -> CPUInfo:
        """Detects CPU information with platform-specific optimizations"""
        cpu_info = CPUInfo(
            name=platform.processor() or "Unknown",
            cores=self._get_cpu_cores(),
            arch=platform.machine()
        )
        
        # Platform-specific CPU detection
        try:
            if platform.system() == "Linux":
                cpu_info.name = await self._get_linux_cpu_name()
            elif platform.system() == "Darwin":
                cpu_info.name = await self._get_macos_cpu_name()
        except Exception as e:
            logger.warning(f"Failed to get detailed CPU info: {e}")
            
        return cpu_info
    
    def _get_cpu_cores(self) -> int:
        """Get CPU core count with fallback"""
        try:
            if HAS_PSUTIL:
                return psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
            return os.cpu_count() or 4
        except Exception:
            return 4
    
    async def _get_linux_cpu_name(self) -> str:
        """Get CPU name on Linux"""
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except FileNotFoundError:
            pass
        return "Unknown Linux CPU"
    
    async def _get_macos_cpu_name(self) -> str:
        """Get CPU name on macOS"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'sysctl', '-n', 'machdep.cpu.brand_string',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                return stdout.decode().strip()
        except Exception as e:
            logger.warning(f"Failed to get macOS CPU name: {e}")
        return "Unknown macOS CPU"
    
    async def _detect_ram(self) -> float:
        """Detects total system RAM in GB"""
        if HAS_PSUTIL:
            try:
                return round(psutil.virtual_memory().total / (1024**3), 1)
            except Exception as e:
                logger.warning(f"psutil RAM detection failed: {e}")
        
        # Fallback methods
        try:
            if platform.system() == "Darwin":
                return await self._get_macos_ram()
            elif platform.system() == "Linux":
                return await self._get_linux_ram()
        except Exception as e:
            logger.warning(f"Platform-specific RAM detection failed: {e}")
            
        return 8.0  # Default fallback
    
    async def _get_macos_ram(self) -> float:
        """Get RAM on macOS"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'sysctl', '-n', 'hw.memsize',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                bytes_ram = int(stdout.decode().strip())
                return round(bytes_ram / (1024**3), 1)
        except Exception as e:
            logger.warning(f"Failed to get macOS RAM: {e}")
        return 8.0
    
    async def _get_linux_ram(self) -> float:
        """Get RAM on Linux"""
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 1)
        except Exception as e:
            logger.warning(f"Failed to get Linux RAM: {e}")
        return 8.0
    
    async def _detect_gpus(self) -> List[GPUInfo]:
        """Detects available GPUs with parallel detection"""
        gpus = []
        
        # Run different GPU detection methods concurrently
        tasks = []
        
        if HAS_GPUTIL:
            tasks.append(asyncio.create_task(self._detect_nvidia_gputil()))
        
        if platform.system() != "Darwin":
            tasks.append(asyncio.create_task(self._detect_nvidia_smi()))
        
        if platform.system() == "Linux":
            tasks.append(asyncio.create_task(self._detect_amd_gpus()))
        
        if platform.system() == "Darwin":
            tasks.append(asyncio.create_task(self._detect_apple_gpus()))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in results:
            if isinstance(result, list):
                gpus.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"GPU detection failed: {result}")
        
        # Remove duplicates based on name
        unique_gpus = []
        seen_names = set()
        for gpu in gpus:
            if gpu.name not in seen_names:
                unique_gpus.append(gpu)
                seen_names.add(gpu.name)
        
        return unique_gpus
    
    async def _detect_nvidia_gputil(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using GPUtil"""
        try:
            nvidia_gpus = GPUtil.getGPUs()
            return [
                GPUInfo(
                    name=gpu.name,
                    vram=round(gpu.memoryTotal / 1024, 1),
                    driver=gpu.driver,
                    vendor="NVIDIA"
                )
                for gpu in nvidia_gpus
            ]
        except Exception as e:
            logger.warning(f"GPUtil detection failed: {e}")
            return []
    
    async def _detect_nvidia_smi(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-smi"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'nvidia-smi', '--query-gpu=name,memory.total',
                '--format=csv,noheader,nounits',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0:
                gpus = []
                for line in stdout.decode().strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        if len(parts) == 2:
                            gpus.append(GPUInfo(
                                name=parts[0],
                                vram=round(float(parts[1]) / 1024, 1),
                                vendor="NVIDIA"
                            ))
                return gpus
        except Exception as e:
            logger.warning(f"nvidia-smi detection failed: {e}")
        return []
    
    async def _detect_amd_gpus(self) -> List[GPUInfo]:
        """Detect AMD GPUs using rocm-smi"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'rocm-smi', '--showmeminfo', 'vram',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0:
                vram_match = re.search(r'Total\s+:\s+(\d+)\s+MB', stdout.decode())
                if vram_match:
                    vram_mb = float(vram_match.group(1))
                    return [GPUInfo(
                        name="AMD GPU",
                        vram=round(vram_mb / 1024, 1),
                        vendor="AMD"
                    )]
        except Exception as e:
            logger.warning(f"rocm-smi detection failed: {e}")
        return []
    
    async def _detect_apple_gpus(self) -> List[GPUInfo]:
        """Detect Apple GPUs (Apple Silicon)"""
        try:
            # Check for Apple Silicon
            proc = await asyncio.create_subprocess_exec(
                'sysctl', '-n', 'hw.optional.arm64',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0 and stdout.decode().strip() == '1':
                # Get chip name
                chip_proc = await asyncio.create_subprocess_exec(
                    'sysctl', '-n', 'machdep.cpu.brand_string',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                chip_stdout, _ = await chip_proc.communicate()
                
                chip_name = "Apple Silicon"
                if chip_proc.returncode == 0:
                    brand = chip_stdout.decode().strip()
                    for chip in ["M1", "M2", "M3", "M4"]:
                        if chip in brand:
                            chip_name = f"Apple {chip} GPU"
                            break
                
                # Get total RAM for unified memory estimate
                total_ram = await self._detect_ram()
                estimated_gpu_memory = round(total_ram * 0.75, 1)
                
                return [GPUInfo(
                    name=chip_name,
                    vram=estimated_gpu_memory,
                    vendor="Apple",
                    note="Unified memory (shared with CPU)"
                )]
            else:
                # Intel Mac - try to detect discrete GPU
                return await self._detect_intel_mac_gpu()
                
        except Exception as e:
            logger.warning(f"Apple GPU detection failed: {e}")
        return []
    
    async def _detect_intel_mac_gpu(self) -> List[GPUInfo]:
        """Detect discrete GPU on Intel Mac"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'system_profiler', 'SPDisplaysDataType', '-json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0:
                data = json.loads(stdout.decode())
                displays = data.get('SPDisplaysDataType', [])
                
                gpus = []
                for display in displays:
                    for key, value in display.items():
                        if isinstance(value, dict) and 'sppci_model' in value:
                            gpu_name = value.get('sppci_model', 'Unknown GPU')
                            vram_str = value.get('spdisplays_vram', '')
                            vram = 4.0  # Default estimate
                            
                            if vram_str:
                                vram_match = re.search(r'(\d+(?:\.\d+)?)\s*(GB|MB)', vram_str)
                                if vram_match:
                                    vram = float(vram_match.group(1))
                                    if vram_match.group(2) == 'MB':
                                        vram = vram / 1024
                            
                            if "Intel" not in gpu_name:  # Skip integrated Intel graphics
                                vendor = "AMD" if ("AMD" in gpu_name or "Radeon" in gpu_name) else "Unknown"
                                gpus.append(GPUInfo(
                                    name=gpu_name,
                                    vram=round(vram, 1),
                                    vendor=vendor
                                ))
                return gpus
                
        except Exception as e:
            logger.warning(f"Intel Mac GPU detection failed: {e}")
        return []
    
    def get_available_memory(self) -> MemoryInfo:
        """Calculates available memory for models"""
        if not self._system_info:
            raise RuntimeError("System info not detected. Call detect_system_info() first.")
        
        # Available RAM (leave 4GB for system)
        available_ram = max(0, self._system_info.ram - 4.0)
        
        # Available VRAM
        if self._system_info.gpus:
            # Use GPU with most VRAM
            max_vram = max(gpu.vram for gpu in self._system_info.gpus)
            available_vram = max_vram * 0.9  # Leave 10% free
            
            # Special handling for Apple Silicon unified memory
            if any(gpu.vendor == "Apple" for gpu in self._system_info.gpus):
                available_vram = min(available_vram, self._system_info.ram * 0.75)
                note = "Unified memory architecture"
            else:
                note = None
                
            return MemoryInfo(
                vram=available_vram,
                ram=available_ram,
                mode="gpu",
                note=note
            )
        else:
            return MemoryInfo(
                vram=0.0,
                ram=available_ram,
                mode="cpu"
            )