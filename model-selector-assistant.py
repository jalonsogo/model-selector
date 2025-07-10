#!/usr/bin/env python3
"""
Interactive AI Model Selection Assistant
Detects local hardware and recommends models with optimal quantization
"""

import json
import platform
import subprocess
import os
from typing import Dict, List, Tuple, Optional
import re

# Try to import optional libraries
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not installed. Install with: pip install psutil")

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Expanded database with different quantizations
MODELS_DB = {
    # SmolLM2 - Ultra lightweight
    "smollm2": {
        "base_name": "SmolLM2",
        "variants": {
            "135M-Q2_K": {"params": "135M", "quant": "Q2_K", "vram": 0.34, "ram": 0.5},
            "135M-Q4_0": {"params": "135M", "quant": "Q4_0", "vram": 0.35, "ram": 0.6},
            "135M-Q4_K_M": {"params": "135M", "quant": "Q4_K_M", "vram": 0.36, "ram": 0.6},
            "135M-F16": {"params": "135M", "quant": "F16", "vram": 0.51, "ram": 0.8},
            "360M-Q4_0": {"params": "360M", "quant": "Q4_0", "vram": 0.59, "ram": 0.9},
            "360M-Q4_K_M": {"params": "360M", "quant": "Q4_K_M", "vram": 0.63, "ram": 1.0},
            "360M-F16": {"params": "360M", "quant": "F16", "vram": 1.06, "ram": 1.5}
        },
        "context": 8000,
        "scores": {"knowledge": 2, "math": 1, "code": 2, "reasoning": 2, "language": 3, "multilingual": 1},
        "tools": True,
        "languages": ["English"],
        "docker": "ai/smollm2"
    },
    
    # Gemma3 - Multimodal
    "gemma3": {
        "base_name": "Gemma3",
        "variants": {
            "1B-Q4_K_M": {"params": "1B", "quant": "Q4_K_M", "vram": 1.40, "ram": 2.0},
            "1B-F16": {"params": "1B", "quant": "F16", "vram": 2.51, "ram": 3.5},
            "4B-Q4_0": {"params": "4B", "quant": "Q4_0", "vram": 3.32, "ram": 4.5},
            "4B-Q4_K_M": {"params": "4B", "quant": "Q4_K_M", "vram": 3.43, "ram": 4.8},
            "4B-F16": {"params": "4B", "quant": "F16", "vram": 8.35, "ram": 10.0}
        },
        "context": 131000,
        "scores": {"knowledge": 3, "math": 2, "code": 2, "reasoning": 3, "language": 3, "multilingual": 4},
        "tools": True,
        "languages": ["140+ languages"],
        "docker": "ai/gemma3",
        "features": ["multimodal"]
    },
    
    # Llama 3.2
    "llama3.2": {
        "base_name": "Llama 3.2",
        "variants": {
            "1B-Q4_0": {"params": "1B", "quant": "Q4_0", "vram": 1.35, "ram": 2.0},
            "1B-Q8_0": {"params": "1B", "quant": "Q8_0", "vram": 1.87, "ram": 2.5},
            "1B-F16": {"params": "1B", "quant": "F16", "vram": 2.95, "ram": 4.0},
            "3B-Q4_0": {"params": "3B", "quant": "Q4_0", "vram": 2.68, "ram": 3.5},
            "3B-Q4_K_M": {"params": "3B", "quant": "Q4_K_M", "vram": 2.77, "ram": 3.8},
            "3B-F16": {"params": "3B", "quant": "F16", "vram": 6.89, "ram": 8.5}
        },
        "context": 131000,
        "scores": {"knowledge": 3, "math": 3, "code": 3, "reasoning": 3, "language": 4, "multilingual": 3},
        "tools": True,
        "languages": ["8 main languages"],
        "docker": "ai/llama3.2"
    },
    
    # Mistral 7B
    "mistral": {
        "base_name": "Mistral",
        "variants": {
            "7B-Q4_0": {"params": "7B", "quant": "Q4_0", "vram": 4.61, "ram": 6.0},
            "7B-Q4_K_M": {"params": "7B", "quant": "Q4_K_M", "vram": 4.85, "ram": 6.5},
            "7B-F16": {"params": "7B", "quant": "F16", "vram": 14.10, "ram": 16.0}
        },
        "context": 33000,
        "scores": {"knowledge": 3, "math": 3, "code": 2, "reasoning": 4, "language": 3, "multilingual": 2},
        "tools": False,
        "languages": ["Primarily English"],
        "docker": "ai/mistral"
    },
    
    # Qwen2.5
    "qwen2.5": {
        "base_name": "Qwen2.5",
        "variants": {
            "0.5B-F16": {"params": "0.5B", "quant": "F16", "vram": 1.38, "ram": 2.0},
            "1.5B-F16": {"params": "1.5B", "quant": "F16", "vram": 3.39, "ram": 4.5},
            "3B-Q4_K_M": {"params": "3B", "quant": "Q4_K_M", "vram": 2.37, "ram": 3.5},
            "3B-F16": {"params": "3B", "quant": "F16", "vram": 6.33, "ram": 8.0},
            "7B-Q4_0": {"params": "7B", "quant": "Q4_0", "vram": 4.60, "ram": 6.0},
            "7B-Q4_K_M": {"params": "7B", "quant": "Q4_K_M", "vram": 4.83, "ram": 6.5},
            "7B-F16": {"params": "7B", "quant": "F16", "vram": 13.93, "ram": 16.0}
        },
        "context": 33000,
        "scores": {"knowledge": 4, "math": 5, "code": 5, "reasoning": 4, "language": 4, "multilingual": 4},
        "tools": True,
        "languages": ["29 languages"],
        "docker": "ai/qwen2.5"
    },
    
    # Llama 3.1
    "llama3.1": {
        "base_name": "Llama 3.1",
        "variants": {
            "8B-Q4_K_M": {"params": "8B", "quant": "Q4_K_M", "vram": 5.33, "ram": 7.0},
            "8B-F16": {"params": "8B", "quant": "F16", "vram": 15.01, "ram": 18.0}
        },
        "context": 131000,
        "scores": {"knowledge": 3, "math": 4, "code": 4, "reasoning": 4, "language": 4, "multilingual": 3},
        "tools": True,
        "languages": ["8 main languages"],
        "docker": "ai/llama3.1"
    },
    
    # Qwen3
    "qwen3": {
        "base_name": "Qwen3",
        "variants": {
            "0.6B-Q4_0": {"params": "0.6B", "quant": "Q4_0", "vram": 1.22, "ram": 2.0},
            "0.6B-Q4_K_M": {"params": "0.6B", "quant": "Q4_K_M", "vram": 1.23, "ram": 2.0},
            "0.6B-F16": {"params": "0.6B", "quant": "F16", "vram": 1.98, "ram": 3.0},
            "8B-Q4_0": {"params": "8B", "quant": "Q4_0", "vram": 5.26, "ram": 7.0},
            "8B-Q4_K_M": {"params": "8B", "quant": "Q4_K_M", "vram": 5.49, "ram": 7.5},
            "8B-F16": {"params": "8B", "quant": "F16", "vram": 15.24, "ram": 18.0},
            "14B-Q6_K": {"params": "14B", "quant": "Q6_K", "vram": 11.96, "ram": 14.0}
        },
        "context": 41000,
        "scores": {"knowledge": 5, "math": 5, "code": 4, "reasoning": 5, "language": 4, "multilingual": 5},
        "tools": True,
        "languages": ["119 languages"],
        "docker": "ai/qwen3",
        "features": ["thinking_modes"]
    },
    
    # Mistral-Nemo
    "mistral-nemo": {
        "base_name": "Mistral-Nemo",
        "variants": {
            "12B-Q4_K_M": {"params": "12B", "quant": "Q4_K_M", "vram": 7.78, "ram": 10.0}
        },
        "context": 131000,
        "scores": {"knowledge": 4, "math": 4, "code": 4, "reasoning": 4, "language": 5, "multilingual": 4},
        "tools": True,
        "languages": ["9 languages"],
        "docker": "ai/mistral-nemo"
    },
    
    # Phi4
    "phi4": {
        "base_name": "Phi4",
        "variants": {
            "14B-Q4_0": {"params": "14B", "quant": "Q4_0", "vram": 9.16, "ram": 11.0},
            "14B-Q4_K_M": {"params": "14B", "quant": "Q4_K_M", "vram": 9.78, "ram": 12.0},
            "14B-F16": {"params": "14B", "quant": "F16", "vram": 27.97, "ram": 32.0}
        },
        "context": 16000,
        "scores": {"knowledge": 4, "math": 5, "code": 4, "reasoning": 3, "language": 3, "multilingual": 3},
        "tools": False,
        "languages": ["8 languages"],
        "docker": "ai/phi4"
    },
    
    # DeepSeek R1 Distill
    "deepseek-r1-distill-llama": {
        "base_name": "DeepSeek-R1-Distill-Llama",
        "variants": {
            "8B-Q4_0": {"params": "8B", "quant": "Q4_0", "vram": 5.09, "ram": 6.5},
            "8B-Q4_K_M": {"params": "8B", "quant": "Q4_K_M", "vram": 5.33, "ram": 7.0},
            "8B-F16": {"params": "8B", "quant": "F16", "vram": 15.01, "ram": 18.0},
            "70B-Q4_0": {"params": "70B", "quant": "Q4_0", "vram": 38.73, "ram": 45.0},
            "70B-Q4_K_M": {"params": "70B", "quant": "Q4_K_M", "vram": 41.11, "ram": 48.0}
        },
        "context": 131000,
        "scores": {"knowledge": 5, "math": 5, "code": 5, "reasoning": 5, "language": 5, "multilingual": 4},
        "tools": True,
        "languages": ["English, Chinese"],
        "docker": "ai/deepseek-r1-distill-llama",
        "features": ["reasoning_optimized"]
    },
    
    # QwQ
    "qwq": {
        "base_name": "QwQ",
        "variants": {
            "32B-Q4_0": {"params": "32B", "quant": "Q4_0", "vram": 18.60, "ram": 22.0},
            "32B-Q4_K_M": {"params": "32B", "quant": "Q4_K_M", "vram": 19.72, "ram": 24.0},
            "32B-F16": {"params": "32B", "quant": "F16", "vram": 61.23, "ram": 70.0}
        },
        "context": 41000,
        "scores": {"knowledge": 4, "math": 5, "code": 5, "reasoning": 5, "language": 4, "multilingual": 4},
        "tools": True,
        "languages": ["29+ languages"],
        "docker": "ai/qwq",
        "features": ["agent_capabilities"]
    },
    
    # Embedding models
    "mxbai-embed": {
        "base_name": "mxbai-embed-large",
        "variants": {
            "335M-F16": {"params": "335M", "quant": "F16", "vram": 0.63, "ram": 1.0}
        },
        "context": 512,
        "scores": {"retrieval": 3, "classification": 4, "clustering": 3, "sts": 5},
        "languages": ["English"],
        "docker": "ai/mxbai-embed-large",
        "type": "embedding"
    }
}

class HardwareDetector:
    """Detects local hardware capabilities"""
    
    def __init__(self):
        self.system_info = {
            "os": platform.system(),
            "arch": platform.machine(),
            "cpu": self.detect_cpu(),
            "ram": self.detect_ram(),
            "gpus": self.detect_gpus()
        }
        
    def detect_cpu(self) -> Dict:
        """Detects CPU information"""
        cpu_info = {
            "name": platform.processor() or "Unknown",
            "cores": os.cpu_count() or 4,
            "arch": platform.machine()
        }
        
        # Try to get more CPU details
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_info["name"] = line.split(":")[1].strip()
                            break
            except:
                pass
        elif platform.system() == "Darwin":
            # macOS specific CPU detection
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    cpu_info["name"] = result.stdout.strip()
            except:
                pass
                
        return cpu_info
        
    def detect_ram(self) -> float:
        """Detects total system RAM in GB"""
        if HAS_PSUTIL:
            return round(psutil.virtual_memory().total / (1024**3), 1)
        else:
            # Fallback methods
            if platform.system() == "Darwin":
                # macOS specific
                try:
                    result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        bytes_ram = int(result.stdout.strip())
                        return round(bytes_ram / (1024**3), 1)
                except:
                    pass
            elif platform.system() == "Linux":
                try:
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if "MemTotal" in line:
                                kb = int(line.split()[1])
                                return round(kb / (1024**2), 1)
                except:
                    pass
            return 8.0  # Default value
            
    def detect_gpus(self) -> List[Dict]:
        """Detects available GPUs"""
        gpus = []
        
        # Method 1: GPUtil (NVIDIA)
        if HAS_GPUTIL:
            try:
                nvidia_gpus = GPUtil.getGPUs()
                for gpu in nvidia_gpus:
                    gpus.append({
                        "name": gpu.name,
                        "vram": round(gpu.memoryTotal / 1024, 1),  # MB to GB
                        "driver": gpu.driver,
                        "vendor": "NVIDIA"
                    })
            except:
                pass
                
        # Method 2: nvidia-smi directly
        if not gpus and platform.system() != "Darwin":  # Not on macOS
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split(', ')
                            if len(parts) == 2:
                                gpus.append({
                                    "name": parts[0],
                                    "vram": round(float(parts[1]) / 1024, 1),  # MB to GB
                                    "vendor": "NVIDIA"
                                })
            except:
                pass
                
        # Method 3: AMD GPUs (Linux)
        if platform.system() == "Linux":
            try:
                result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse rocm-smi output
                    vram_match = re.search(r'Total\s+:\s+(\d+)\s+MB', result.stdout)
                    if vram_match:
                        vram_mb = float(vram_match.group(1))
                        gpus.append({
                            "name": "AMD GPU",
                            "vram": round(vram_mb / 1024, 1),
                            "vendor": "AMD"
                        })
            except:
                pass
                
        # Method 4: macOS Metal Performance Shaders
        if platform.system() == "Darwin":
            try:
                # Check for Apple Silicon
                result = subprocess.run(['sysctl', '-n', 'hw.optional.arm64'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip() == '1':
                    # Apple Silicon detected
                    # Get total memory (shared between CPU and GPU)
                    total_ram = self.system_info.get("ram", 8.0)
                    
                    # Estimate available GPU memory (typically can use up to 75% on Apple Silicon)
                    estimated_gpu_memory = round(total_ram * 0.75, 1)
                    
                    # Get chip name
                    chip_result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                               capture_output=True, text=True)
                    chip_name = "Apple Silicon"
                    if chip_result.returncode == 0:
                        brand = chip_result.stdout.strip()
                        if "M1" in brand:
                            chip_name = "Apple M1 GPU"
                        elif "M2" in brand:
                            chip_name = "Apple M2 GPU"
                        elif "M3" in brand:
                            chip_name = "Apple M3 GPU"
                        elif "M4" in brand:
                            chip_name = "Apple M4 GPU"
                    
                    gpus.append({
                        "name": chip_name,
                        "vram": estimated_gpu_memory,
                        "vendor": "Apple",
                        "note": "Unified memory (shared with CPU)"
                    })
                else:
                    # Intel Mac - check for discrete GPU
                    result = subprocess.run(['system_profiler', 'SPDisplaysDataType', '-json'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        try:
                            import json
                            data = json.loads(result.stdout)
                            displays = data.get('SPDisplaysDataType', [])
                            
                            for display in displays:
                                # Look for discrete GPUs
                                for key, value in display.items():
                                    if isinstance(value, dict) and 'sppci_model' in value:
                                        gpu_name = value.get('sppci_model', 'Unknown GPU')
                                        # Try to find VRAM info
                                        vram_str = value.get('spdisplays_vram', '')
                                        vram = 4.0  # Default estimate
                                        
                                        if vram_str:
                                            # Parse VRAM string (e.g., "1536 MB")
                                            vram_match = re.search(r'(\d+(?:\.\d+)?)\s*(GB|MB)', vram_str)
                                            if vram_match:
                                                vram = float(vram_match.group(1))
                                                if vram_match.group(2) == 'MB':
                                                    vram = vram / 1024
                                        
                                        if "Intel" not in gpu_name:  # Skip integrated Intel graphics
                                            gpus.append({
                                                "name": gpu_name,
                                                "vram": round(vram, 1),
                                                "vendor": "AMD" if "AMD" in gpu_name or "Radeon" in gpu_name else "Unknown"
                                            })
                        except:
                            pass
            except:
                pass
                
        return gpus
        
    def get_available_memory(self) -> Dict[str, float]:
        """Calculates available memory for models"""
        available = {
            "vram": 0.0,
            "ram": 0.0,
            "mode": "cpu"  # cpu or gpu
        }
        
        # Available RAM (leave 4GB for system)
        total_ram = self.system_info["ram"]
        available["ram"] = max(0, total_ram - 4.0)
        
        # Available VRAM
        if self.system_info["gpus"]:
            # Use GPU with most VRAM
            max_vram = max(gpu["vram"] for gpu in self.system_info["gpus"])
            available["vram"] = max_vram * 0.9  # Leave 10% free
            available["mode"] = "gpu"
            
            # Special handling for Apple Silicon unified memory
            if any(gpu.get("vendor") == "Apple" for gpu in self.system_info["gpus"]):
                # On Apple Silicon, we can use more of the total RAM for GPU tasks
                available["vram"] = min(available["vram"], total_ram * 0.75)
                available["note"] = "Unified memory architecture"
            
        return available

class ModelSelector:
    def __init__(self):
        self.hardware = HardwareDetector()
        self.user_requirements = {}
        
    def clear_screen(self):
        """Clears the screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self, text: str):
        """Prints a formatted header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70 + "\n")
        
    def display_hardware_info(self):
        """Displays detected hardware information"""
        self.print_header("🖥️  DETECTED HARDWARE")
        
        info = self.hardware.system_info
        print(f"📊 Operating System: {info['os']} ({info['arch']})")
        print(f"🧠 CPU: {info['cpu']['name']}")
        print(f"   • Cores: {info['cpu']['cores']}")
        print(f"💾 Total RAM: {info['ram']} GB")
        
        if info['gpus']:
            print(f"\n🎮 Detected GPU(s):")
            for i, gpu in enumerate(info['gpus'], 1):
                print(f"   {i}. {gpu['name']}")
                print(f"      • VRAM: {gpu['vram']} GB")
                print(f"      • Vendor: {gpu['vendor']}")
                if 'note' in gpu:
                    print(f"      • Note: {gpu['note']}")
        else:
            print(f"\n⚠️  No GPUs detected - Models will run on CPU")
            
        available = self.hardware.get_available_memory()
        print(f"\n💡 Available Memory for Models:")
        print(f"   • RAM: {available['ram']:.1f} GB")
        if available['mode'] == 'gpu':
            print(f"   • VRAM: {available['vram']:.1f} GB")
            if 'note' in available:
                print(f"   • {available['note']}")
        print(f"   • Mode: {available['mode'].upper()}")
        
        input("\nPress Enter to continue...")
        
    def get_quantization_recommendation(self, model_family: str, available_memory: float, 
                                      mode: str) -> Optional[str]:
        """Recommends the best quantization based on available memory"""
        if model_family not in MODELS_DB:
            return None
            
        variants = MODELS_DB[model_family]["variants"]
        suitable = []
        
        for variant_name, specs in variants.items():
            memory_req = specs["vram"] if mode == "gpu" else specs["ram"]
            if memory_req <= available_memory:
                # Prioritize quantizations by quality
                quant_priority = {
                    "F16": 100,      # Maximum quality
                    "Q8_0": 90,      # Very high quality
                    "Q6_K": 80,      # High quality
                    "Q4_K_M": 70,    # Good quality (recommended)
                    "Q4_0": 60,      # Acceptable quality
                    "Q2_K": 30       # Low quality
                }
                priority = quant_priority.get(specs["quant"], 50)
                suitable.append((variant_name, priority, memory_req))
                
        if not suitable:
            return None
            
        # Sort by priority (quality) descending
        suitable.sort(key=lambda x: x[1], reverse=True)
        return suitable[0][0]  # Return highest quality variant that fits
        
    def ask_use_case(self) -> str:
        """Asks about the main use case"""
        self.print_header("STEP 1: Main Use Case")
        print("What will you primarily use the model for?")
        print("\n1. RAG (Retrieval-Augmented Generation)")
        print("2. Programming assistant")
        print("3. Conversational chatbot")
        print("4. Document analysis and processing")
        print("5. Translation and multilingual tasks")
        print("6. Education and tutoring (math/science)")
        print("7. Creative writing")
        print("8. General use/multiple purposes")
        
        while True:
            try:
                choice = input("\nSelect an option (1-8): ").strip()
                use_cases = {
                    "1": "rag", "2": "code", "3": "chat", "4": "analysis",
                    "5": "multilingual", "6": "education", "7": "creative", "8": "general"
                }
                if choice in use_cases:
                    return use_cases[choice]
                else:
                    print("Please select a valid option (1-8)")
            except:
                print("Invalid input. Please try again.")
                
    def ask_context_importance(self) -> int:
        """Asks about the importance of long context"""
        self.print_header("STEP 2: Context Length")
        print("How important is it to process very long texts?")
        print("\n1. Not important (short texts are fine)")
        print("2. Somewhat important (medium documents)")
        print("3. Very important (need to process long documents)")
        print("4. Critical (need maximum context possible)")
        
        while True:
            try:
                choice = input("\nSelect an option (1-4): ").strip()
                if choice in ["1", "2", "3", "4"]:
                    return int(choice)
                else:
                    print("Please select a valid option (1-4)")
            except:
                print("Invalid input. Please try again.")
                
    def ask_languages(self) -> List[str]:
        """Asks about required languages"""
        self.print_header("STEP 3: Required Languages")
        print("Which languages do you need the model to support?")
        print("\n1. English only")
        print("2. English and Spanish")
        print("3. Main European languages")
        print("4. I need support for many languages (20+)")
        print("5. I need support for Asian languages")
        
        while True:
            try:
                choice = input("\nSelect an option (1-5): ").strip()
                languages_map = {
                    "1": ["english"],
                    "2": ["english", "spanish"],
                    "3": ["european"],
                    "4": ["many"],
                    "5": ["asian"]
                }
                if choice in languages_map:
                    return languages_map[choice]
                else:
                    print("Please select a valid option (1-5)")
            except:
                print("Invalid input. Please try again.")
                
    def ask_tool_calling(self) -> bool:
        """Asks about tool calling needs"""
        self.print_header("STEP 4: Tool Integration")
        print("Do you need the model to call external functions/APIs?")
        print("\n1. Yes, it's important")
        print("2. Not necessary")
        print("3. I don't know what this is")
        
        while True:
            try:
                choice = input("\nSelect an option (1-3): ").strip()
                if choice == "1":
                    return True
                elif choice in ["2", "3"]:
                    return False
                else:
                    print("Please select a valid option (1-3)")
            except:
                print("Invalid input. Please try again.")
                
    def ask_override_hardware(self) -> bool:
        """Asks if user wants manual configuration"""
        print("\nDo you want to use the detected hardware configuration?")
        print("1. Yes, use detected hardware")
        print("2. No, configure manually")
        
        while True:
            choice = input("\nSelect an option (1-2): ").strip()
            if choice == "1":
                return False
            elif choice == "2":
                return True
            else:
                print("Please select a valid option (1-2)")
                
    def calculate_model_score(self, model_data: Dict, use_case: str, 
                            context_importance: int, languages: List[str], 
                            needs_tools: bool) -> float:
        """Calculates model score based on requirements"""
        score = 0.0
        
        # Scoring by use case
        use_case_weights = {
            "rag": {"knowledge": 0.3, "language": 0.3, "reasoning": 0.2, "multilingual": 0.2},
            "code": {"code": 0.5, "reasoning": 0.3, "language": 0.2},
            "chat": {"language": 0.4, "reasoning": 0.3, "knowledge": 0.3},
            "analysis": {"reasoning": 0.4, "knowledge": 0.3, "language": 0.3},
            "multilingual": {"multilingual": 0.5, "language": 0.3, "knowledge": 0.2},
            "education": {"math": 0.4, "reasoning": 0.3, "language": 0.3},
            "creative": {"language": 0.5, "reasoning": 0.3, "knowledge": 0.2},
            "general": {"knowledge": 0.25, "reasoning": 0.25, "language": 0.25, "code": 0.25}
        }
        
        if model_data.get("type") != "embedding":
            weights = use_case_weights.get(use_case, use_case_weights["general"])
            scores = model_data["scores"]
            
            for metric, weight in weights.items():
                if metric in scores:
                    score += scores[metric] * weight * 20
                    
            # Bonus for long context
            if context_importance >= 3 and model_data["context"] >= 100000:
                score += 20
            elif context_importance >= 2 and model_data["context"] >= 40000:
                score += 10
                
            # Bonus for multilingual
            if "many" in languages and scores.get("multilingual", 0) >= 4:
                score += 15
            elif "european" in languages and scores.get("multilingual", 0) >= 3:
                score += 10
            elif "spanish" in languages and scores.get("multilingual", 0) >= 2:
                score += 5
                
            # Penalty if needs tools and doesn't have them
            if needs_tools and not model_data.get("tools", False):
                score -= 20
                
        return score
        
    def recommend_models(self, use_case: str, context_importance: int,
                        languages: List[str], needs_tools: bool) -> List[Tuple[str, Dict, float, str]]:
        """Recommends best models based on requirements and hardware"""
        available = self.hardware.get_available_memory()
        suitable_models = []
        
        # For RAG, we also need an embedding model
        if use_case == "rag":
            embed_vram = MODELS_DB["mxbai-embed"]["variants"]["335M-F16"]["vram"]
            embed_ram = MODELS_DB["mxbai-embed"]["variants"]["335M-F16"]["ram"]
            
            if available["mode"] == "gpu":
                remaining_memory = available["vram"] - embed_vram - 0.5
            else:
                remaining_memory = available["ram"] - embed_ram - 1.0
        else:
            if available["mode"] == "gpu":
                remaining_memory = available["vram"] - 0.5
            else:
                remaining_memory = available["ram"] - 1.0
                
        # Evaluate each model family
        for model_family, model_data in MODELS_DB.items():
            if model_data.get("type") == "embedding":
                continue
                
            # Find best quantization for this model
            best_variant = self.get_quantization_recommendation(
                model_family, remaining_memory, available["mode"]
            )
            
            if best_variant:
                variant_data = model_data["variants"][best_variant]
                score = self.calculate_model_score(
                    model_data, use_case, context_importance, languages, needs_tools
                )
                
                suitable_models.append((
                    model_family,
                    model_data,
                    score,
                    best_variant
                ))
                
        # Sort by score
        suitable_models.sort(key=lambda x: x[2], reverse=True)
        
        return suitable_models[:5]  # Top 5 models
        
    def display_recommendations(self, recommendations: List[Tuple[str, Dict, float, str]], 
                              use_case: str):
        """Displays recommendations clearly"""
        self.print_header("🎯 PERSONALIZED RECOMMENDATIONS")
        
        available = self.hardware.get_available_memory()
        mode = available["mode"].upper()
        
        if not recommendations:
            print("❌ No models found that meet your requirements.")
            print(f"   Your hardware has significant limitations.")
            print(f"   Consider using cloud services or upgrading your hardware.")
            return
            
        if use_case == "rag":
            print("🔍 For RAG you'll need two components:")
            print("\n📊 EMBEDDING MODEL:")
            print(f"   • mxbai-embed-large (0.63 GB)")
            print("   • Optimized for semantic search")
            print(f"\n🤖 GENERATIVE MODELS (sorted by recommendation) - {mode} Mode:\n")
        else:
            print(f"🤖 RECOMMENDED MODELS (sorted by score) - {mode} Mode:\n")
            
        for i, (model_family, model_data, score, variant) in enumerate(recommendations, 1):
            variant_data = model_data["variants"][variant]
            params = variant_data["params"]
            quant = variant_data["quant"]
            
            print(f"{i}. {model_data['base_name']} {params} ({quant})")
            
            if mode == "GPU":
                print(f"   • VRAM required: {variant_data['vram']:.2f} GB")
            else:
                print(f"   • RAM required: {variant_data['ram']:.2f} GB")
                print(f"   • ⚠️  Will run on CPU - may be slower")
                
            print(f"   • Context: {model_data['context']:,} tokens")
            print(f"   • Score: {'⭐' * int(score/20)}")
            print(f"   • Languages: {', '.join(model_data['languages'])}")
            
            # Special features
            if "features" in model_data:
                features = model_data.get("features", [])
                if features:
                    print(f"   • Features: {', '.join(features)}")
                    
            # Docker command with specific variant
            docker_cmd = f"docker model pull {model_data['docker']}:{params}-{quant}"
            print(f"   • Docker: {docker_cmd}")
            print()
            
    def run(self):
        """Runs the complete flow"""
        self.clear_screen()
        print("🤖 INTELLIGENT AI MODEL SELECTION ASSISTANT")
        print("   With automatic hardware detection")
        
        # Show detected hardware
        self.display_hardware_info()
        
        # Ask if manual override wanted
        manual_override = self.ask_override_hardware()
        
        if manual_override:
            # Manual configuration (legacy code)
            print("\n⚠️  Manual configuration not implemented in this version")
            print("   Using detected hardware...")
            
        # Collect requirements
        self.clear_screen()
        use_case = self.ask_use_case()
        context_importance = self.ask_context_importance()
        languages = self.ask_languages()
        needs_tools = self.ask_tool_calling()
        
        # Save requirements
        self.user_requirements = {
            "hardware": self.hardware.system_info,
            "available_memory": self.hardware.get_available_memory(),
            "use_case": use_case,
            "context_importance": context_importance,
            "languages": languages,
            "needs_tools": needs_tools
        }
        
        # Get recommendations
        recommendations = self.recommend_models(
            use_case, context_importance, languages, needs_tools
        )
        
        # Show results
        self.clear_screen()
        self.display_recommendations(recommendations, use_case)
        
        # Summary
        self.display_summary(recommendations)
        
    def display_summary(self, recommendations):
        """Shows final summary"""
        print("\n" + "-"*70)
        print("📋 YOUR CONFIGURATION SUMMARY:")
        
        hw = self.hardware.system_info
        available = self.hardware.get_available_memory()
        
        print(f"   • CPU: {hw['cpu']['cores']} cores")
        print(f"   • RAM: {hw['ram']} GB total ({available['ram']:.1f} GB available)")
        
        if hw['gpus']:
            gpu = hw['gpus'][0]  # First GPU
            print(f"   • GPU: {gpu['name']} ({gpu['vram']} GB VRAM)")
            print(f"   • Mode: GPU ({available['vram']:.1f} GB VRAM available)")
        else:
            print(f"   • GPU: Not detected")
            print(f"   • Mode: CPU (may be slower)")
            
        print(f"\n   • Use case: {self.user_requirements['use_case']}")
        print(f"   • Context importance: {'⭐' * self.user_requirements['context_importance']}")
        print(f"   • Tool calling: {'Yes' if self.user_requirements['needs_tools'] else 'No'}")
        
        # Save results
        self.save_results(recommendations)
        
    def save_results(self, recommendations):
        """Saves results to file"""
        print("\n" + "-"*70)
        save = input("\nWould you like to save these recommendations? (y/n): ").strip().lower()
        
        if save == 'y':
            results = {
                "hardware": self.hardware.system_info,
                "requirements": self.user_requirements,
                "recommendations": [
                    {
                        "model": f"{model['base_name']} {model['variants'][variant]['params']}",
                        "quantization": model['variants'][variant]['quant'],
                        "memory_required": {
                            "vram": model['variants'][variant]['vram'],
                            "ram": model['variants'][variant]['ram']
                        },
                        "context": model["context"],
                        "score": score,
                        "docker_command": f"docker model pull {model['docker']}:{model['variants'][variant]['params']}-{model['variants'][variant]['quant']}"
                    }
                    for _, model, score, variant in recommendations
                ]
            }
            
            filename = f"model_recommendations_{platform.node()}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Recommendations saved to '{filename}'")

if __name__ == "__main__":
    print("🔍 Detecting hardware...")
    print("   This may take a few seconds...\n")
    
    selector = ModelSelector()
    selector.run()