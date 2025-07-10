# AI Model Selection Assistant

A Python-based intelligent assistant that detects your local hardware capabilities and recommends optimal AI models with appropriate quantization settings. Built with a modern, modular architecture featuring async hardware detection, rich CLI interface, and comprehensive caching.

## 🚀 Features

- **🔍 Smart Hardware Detection**: Automatically detects CPU, RAM, and GPU capabilities across Linux, macOS, and Windows
- **🎯 Intelligent Recommendations**: Scores models based on your specific use case and hardware constraints
- **⚡ Async Performance**: Parallel hardware detection for faster startup times
- **💾 Caching**: Hardware detection results cached for subsequent runs
- **🎨 Beautiful CLI**: Rich terminal interface with progress bars, tables, and formatted output
- **📊 Multiple Export Formats**: Save recommendations as JSON, CSV, or YAML
- **🔧 Configuration Persistence**: Saves your preferences between sessions
- **🧪 Comprehensive Testing**: Full test suite with hardware mocking

## 📦 Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
pip install -r requirements-dev.txt
```

### Optional Dependencies
For enhanced functionality, install optional packages:
```bash
pip install psutil GPUtil PyYAML
```

## 🖥️ Usage

### Interactive Mode (Recommended)
Run the full interactive experience:
```bash
python main.py interactive
```

### Hardware Detection Only
Check your system capabilities:
```bash
python main.py hardware
```

### List Available Models
Browse the model database:
```bash
python main.py models
python main.py models --model-id qwen2.5  # Show specific model details
```

### Direct Recommendations
Get recommendations with specific parameters:
```bash
python main.py recommend --use-case code --context-importance 3 --tools
```

### Cache Management
Clear hardware detection cache:
```bash
python main.py clear-cache
```

## 🏗️ Architecture

### Modular Design
- **`model_selector/hardware.py`**: Async hardware detection with caching
- **`model_selector/models.py`**: Model management and recommendation engine
- **`model_selector/selector.py`**: Rich CLI interface with user interaction
- **`main.py`**: CLI entry point with Click framework

### Configuration Files
- **`models.json`**: External model database (easily updatable)
- **`~/.model-selector/config.json`**: User preferences persistence
- **`~/.model-selector/hardware_cache.json`**: Hardware detection cache

## 🎯 Use Cases

The assistant optimizes recommendations for various scenarios:

1. **RAG (Retrieval-Augmented Generation)**: Includes embedding model requirements
2. **Programming Assistant**: Prioritizes code generation capabilities
3. **Conversational Chatbot**: Focuses on language and reasoning skills
4. **Document Analysis**: Emphasizes reasoning and knowledge capabilities
5. **Multilingual Tasks**: Considers language support breadth
6. **Education/Tutoring**: Highlights math and reasoning abilities
7. **Creative Writing**: Prioritizes language generation quality
8. **General Use**: Balanced scoring across all capabilities

## 🔧 Development

### Running Tests
```bash
pytest                                    # Run all tests
pytest tests/test_hardware.py -v        # Test hardware detection
pytest tests/test_models.py -v          # Test model management
pytest --cov=model_selector             # With coverage
```

### Code Quality
```bash
black model_selector/                    # Format code
flake8 model_selector/                   # Lint code
mypy model_selector/                     # Type checking
```

### Adding New Models
Update the `models.json` file with new model specifications:
```json
{
  "models": {
    "new_model": {
      "base_name": "New Model",
      "variants": {
        "7B-Q4_K_M": {"params": "7B", "quant": "Q4_K_M", "vram": 4.5, "ram": 6.0}
      },
      "context": 8000,
      "scores": {"knowledge": 4, "math": 3, "code": 4, "reasoning": 4, "language": 4},
      "tools": true,
      "languages": ["English"],
      "docker": "ai/new-model"
    }
  }
}
```

## 🛠️ Hardware Support

### CPU Detection
- **Linux**: Reads `/proc/cpuinfo` for detailed CPU information
- **macOS**: Uses `sysctl` for CPU brand and core count
- **Windows**: Platform-specific detection with fallbacks

### GPU Detection
- **NVIDIA**: Uses `nvidia-smi` and optional `GPUtil` library
- **AMD**: Uses `rocm-smi` on Linux systems
- **Apple Silicon**: Detects M1/M2/M3/M4 chips with unified memory
- **Intel Mac**: Detects discrete GPUs via `system_profiler`

### Memory Analysis
- **Primary**: Uses `psutil` for accurate memory detection
- **Fallback**: Platform-specific methods for systems without psutil
- **Smart Allocation**: Reserves memory for system processes

## 📊 Model Database

The application includes a comprehensive database of popular AI models:

- **SmolLM2**: Ultra-lightweight models (135M-360M parameters)
- **Gemma3**: Multimodal capabilities (1B-4B parameters)
- **Llama 3.2**: Meta's latest models (1B-3B parameters)
- **Qwen2.5**: Multilingual excellence (0.5B-7B parameters)
- **Qwen3**: Latest with thinking modes (0.6B-14B parameters)
- **Mistral**: Reasoning-focused models (7B-12B parameters)
- **Phi4**: Microsoft's efficient models (14B parameters)
- **DeepSeek-R1**: Reasoning-optimized models (8B-70B parameters)
- **QwQ**: Agent-capable models (32B parameters)

Each model includes:
- Multiple quantization options (F16, Q8_0, Q6_K, Q4_K_M, Q4_0, Q2_K)
- Memory requirements for both VRAM and RAM
- Capability scores across different domains
- Context length specifications
- Tool calling support
- Language support information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hardware detection inspired by various system monitoring tools
- Model specifications gathered from official documentation
- CLI interface built with the excellent Rich library
- Async programming patterns for improved performance