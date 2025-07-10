"""
Tests for hardware detection module
"""

import asyncio
import json
import platform
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from model_selector.hardware import (
    CPUInfo, GPUInfo, HardwareDetector, MemoryInfo, SystemInfo
)


class TestHardwareDetector:
    """Test hardware detection functionality"""
    
    @pytest.fixture
    def detector(self):
        return HardwareDetector()
    
    @pytest.fixture
    def mock_system_info(self):
        return SystemInfo(
            os="Darwin",
            arch="arm64",
            cpu=CPUInfo(name="Apple M1", cores=8, arch="arm64"),
            ram=16.0,
            gpus=[GPUInfo(name="Apple M1 GPU", vram=12.0, vendor="Apple")]
        )
    
    @pytest.mark.asyncio
    async def test_detect_system_info(self, detector):
        """Test basic system info detection"""
        with patch.object(detector, '_detect_cpu') as mock_cpu, \
             patch.object(detector, '_detect_ram') as mock_ram, \
             patch.object(detector, '_detect_gpus') as mock_gpus:
            
            mock_cpu.return_value = CPUInfo(name="Test CPU", cores=4, arch="x86_64")
            mock_ram.return_value = 8.0
            mock_gpus.return_value = []
            
            system_info = await detector.detect_system_info(use_cache=False)
            
            assert system_info.cpu.name == "Test CPU"
            assert system_info.cpu.cores == 4
            assert system_info.ram == 8.0
            assert system_info.gpus == []
    
    @pytest.mark.asyncio
    async def test_detect_cpu_linux(self, detector):
        """Test CPU detection on Linux"""
        with patch('platform.system', return_value='Linux'), \
             patch('builtins.open', mock_open(read_data='model name\t: Intel Core i7-9700K')):
            
            cpu_info = await detector._detect_cpu()
            assert "Intel Core i7-9700K" in cpu_info.name
    
    @pytest.mark.asyncio
    async def test_detect_cpu_macos(self, detector):
        """Test CPU detection on macOS"""
        with patch('platform.system', return_value='Darwin'), \
             patch('asyncio.create_subprocess_exec') as mock_subprocess:
            
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b'Apple M1 Pro\\n', b'')
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            cpu_info = await detector._detect_cpu()
            assert "Apple M1 Pro" in cpu_info.name
    
    @pytest.mark.asyncio
    async def test_detect_ram_with_psutil(self, detector):
        """Test RAM detection with psutil"""
        with patch('model_selector.hardware.HAS_PSUTIL', True), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.total = 16 * 1024**3  # 16GB
            ram = await detector._detect_ram()
            assert ram == 16.0
    
    @pytest.mark.asyncio
    async def test_detect_ram_macos_fallback(self, detector):
        """Test RAM detection on macOS without psutil"""
        with patch('model_selector.hardware.HAS_PSUTIL', False), \
             patch('platform.system', return_value='Darwin'), \
             patch('asyncio.create_subprocess_exec') as mock_subprocess:
            
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b'17179869184\\n', b'')  # 16GB
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            ram = await detector._detect_ram()
            assert ram == 16.0
    
    @pytest.mark.asyncio
    async def test_detect_nvidia_gpus(self, detector):
        """Test NVIDIA GPU detection"""
        with patch('model_selector.hardware.HAS_GPUTIL', True), \
             patch('GPUtil.getGPUs') as mock_gpus:
            
            mock_gpu = MagicMock()
            mock_gpu.name = "NVIDIA GeForce RTX 3080"
            mock_gpu.memoryTotal = 10240  # 10GB
            mock_gpu.driver = "460.89"
            mock_gpus.return_value = [mock_gpu]
            
            gpus = await detector._detect_nvidia_gputil()
            assert len(gpus) == 1
            assert gpus[0].name == "NVIDIA GeForce RTX 3080"
            assert gpus[0].vram == 10.0
            assert gpus[0].vendor == "NVIDIA"
    
    @pytest.mark.asyncio
    async def test_detect_apple_silicon_gpu(self, detector):
        """Test Apple Silicon GPU detection"""
        with patch('platform.system', return_value='Darwin'), \
             patch('asyncio.create_subprocess_exec') as mock_subprocess:
            
            # Mock Apple Silicon detection
            mock_process1 = AsyncMock()
            mock_process1.communicate.return_value = (b'1\\n', b'')
            mock_process1.returncode = 0
            
            # Mock CPU brand detection
            mock_process2 = AsyncMock()
            mock_process2.communicate.return_value = (b'Apple M1 Pro\\n', b'')
            mock_process2.returncode = 0
            
            mock_subprocess.side_effect = [mock_process1, mock_process2]
            
            # Mock RAM detection
            detector._detect_ram = AsyncMock(return_value=16.0)
            
            gpus = await detector._detect_apple_gpus()
            assert len(gpus) == 1
            assert "Apple M1 Pro GPU" in gpus[0].name
            assert gpus[0].vendor == "Apple"
    
    def test_get_available_memory_gpu_mode(self, detector, mock_system_info):
        """Test memory calculation in GPU mode"""
        detector._system_info = mock_system_info
        
        memory_info = detector.get_available_memory()
        
        assert memory_info.mode == "gpu"
        assert memory_info.vram == 12.0 * 0.75  # Apple Silicon unified memory
        assert memory_info.ram == 16.0 - 4.0  # Total - reserved
        assert memory_info.note == "Unified memory architecture"
    
    def test_get_available_memory_cpu_mode(self, detector):
        """Test memory calculation in CPU mode"""
        system_info = SystemInfo(
            os="Linux",
            arch="x86_64",
            cpu=CPUInfo(name="Intel CPU", cores=4, arch="x86_64"),
            ram=8.0,
            gpus=[]
        )
        detector._system_info = system_info
        
        memory_info = detector.get_available_memory()
        
        assert memory_info.mode == "cpu"
        assert memory_info.vram == 0.0
        assert memory_info.ram == 4.0  # 8GB - 4GB reserved
    
    def test_cache_functionality(self, detector, tmp_path):
        """Test hardware info caching"""
        # Mock cache file location
        cache_file = tmp_path / "hardware_cache.json"
        detector._cache_file = cache_file
        
        # Mock system info
        detector._system_info = SystemInfo(
            os="Linux",
            arch="x86_64",
            cpu=CPUInfo(name="Test CPU", cores=4, arch="x86_64"),
            ram=8.0,
            gpus=[]
        )
        
        # Test caching
        detector._cache_hardware_info()
        assert cache_file.exists()
        
        # Test loading from cache
        with patch('time.time', return_value=1000):
            cached_data = json.loads(cache_file.read_text())
            cached_data['timestamp'] = 999  # Recent timestamp
            cache_file.write_text(json.dumps(cached_data))
            
            parsed_info = detector._parse_cached_info(cached_data)
            assert parsed_info.cpu.name == "Test CPU"
            assert parsed_info.ram == 8.0


class TestDataClasses:
    """Test data classes"""
    
    def test_cpu_info_creation(self):
        """Test CPUInfo creation"""
        cpu = CPUInfo(name="Intel i7", cores=8, arch="x86_64")
        assert cpu.name == "Intel i7"
        assert cpu.cores == 8
        assert cpu.arch == "x86_64"
    
    def test_gpu_info_creation(self):
        """Test GPUInfo creation"""
        gpu = GPUInfo(name="RTX 3080", vram=10.0, vendor="NVIDIA", driver="460.89")
        assert gpu.name == "RTX 3080"
        assert gpu.vram == 10.0
        assert gpu.vendor == "NVIDIA"
        assert gpu.driver == "460.89"
    
    def test_memory_info_creation(self):
        """Test MemoryInfo creation"""
        memory = MemoryInfo(vram=8.0, ram=16.0, mode="gpu", note="Test note")
        assert memory.vram == 8.0
        assert memory.ram == 16.0
        assert memory.mode == "gpu"
        assert memory.note == "Test note"
    
    def test_system_info_creation(self):
        """Test SystemInfo creation"""
        cpu = CPUInfo(name="Test CPU", cores=4, arch="x86_64")
        gpu = GPUInfo(name="Test GPU", vram=8.0, vendor="NVIDIA")
        
        system = SystemInfo(
            os="Linux",
            arch="x86_64",
            cpu=cpu,
            ram=16.0,
            gpus=[gpu]
        )
        
        assert system.os == "Linux"
        assert system.arch == "x86_64"
        assert system.cpu.name == "Test CPU"
        assert system.ram == 16.0
        assert len(system.gpus) == 1
        assert system.gpus[0].name == "Test GPU"


if __name__ == "__main__":
    pytest.main([__file__])