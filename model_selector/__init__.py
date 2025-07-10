"""
AI Model Selection Assistant
Detects local hardware and recommends models with optimal quantization
"""

from .hardware import HardwareDetector, SystemInfo, GPUInfo, MemoryInfo
from .models import ModelManager, ModelData, ModelVariant
from .selector import ModelSelector

__version__ = "1.0.0"
__all__ = [
    "HardwareDetector",
    "SystemInfo", 
    "GPUInfo",
    "MemoryInfo",
    "ModelManager",
    "ModelData",
    "ModelVariant",
    "ModelSelector",
]