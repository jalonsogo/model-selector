# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based AI model selection assistant that detects local hardware capabilities and recommends optimal AI models with appropriate quantization. The tool has been completely refactored into a modern, modular architecture with enhanced CLI, caching, and comprehensive testing.

## New Architecture

The application is now organized into separate modules:

### Core Modules

1. **`model_selector/hardware.py`**: Hardware detection with async support
   - `HardwareDetector`: Async hardware detection with caching
   - `SystemInfo`, `CPUInfo`, `GPUInfo`, `MemoryInfo`: Data classes for hardware info
   - Cross-platform support (Linux, macOS, Windows)
   - Parallel GPU detection methods
   - Hardware info caching to `~/.model-selector/hardware_cache.json`

2. **`model_selector/models.py`**: Model management and recommendations
   - `ModelManager`: Loads models from JSON, handles recommendations
   - `ModelData`, `ModelVariant`: Data classes for model specifications
   - Scoring algorithm based on use case and requirements
   - Export functionality (JSON, CSV, YAML)
   - Model database updates

3. **`model_selector/selector.py`**: Enhanced CLI interface
   - `ModelSelector`: Rich CLI with progress bars and formatted output
   - User configuration persistence
   - Interactive questionnaire with saved preferences
   - Beautiful tables and panels using Rich library

4. **`main.py`**: CLI entry point with Click framework
   - Multiple commands: `interactive`, `hardware`, `models`, `recommend`
   - Verbose logging and error handling
   - Cache management
   - Batch processing support

### Configuration

- **`models.json`**: External model database (no longer hardcoded)
- **`~/.model-selector/config.json`**: User preferences persistence
- **`~/.model-selector/hardware_cache.json`**: Hardware detection cache

## Running the Application

### Interactive Mode (Recommended)
```bash
python main.py interactive
```

### Hardware Detection Only
```bash
python main.py hardware
```

### List Available Models
```bash
python main.py models
python main.py models --model-id qwen2.5  # Show specific model
```

### Get Direct Recommendations
```bash
python main.py recommend --use-case code --context-importance 3 --tools
```

### Clear Cache
```bash
python main.py clear-cache
```

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
pip install -r requirements-dev.txt
```

## Dependencies

### Core Dependencies
- `click`: Modern CLI framework
- `rich`: Beautiful terminal output with progress bars and tables

### Optional Dependencies
- `psutil`: Enhanced system memory detection
- `GPUtil`: NVIDIA GPU detection
- `PyYAML`: YAML export support

## Development

### Running Tests
```bash
pytest
```

### Code Quality
```bash
black model_selector/  # Format code
flake8 model_selector/  # Lint code
mypy model_selector/   # Type checking
```

### Test Coverage
```bash
pytest --cov=model_selector --cov-report=html
```

## Key Improvements

1. **Modular Architecture**: Separated concerns into distinct modules
2. **Async Hardware Detection**: Parallel detection methods for better performance
3. **Rich CLI**: Beautiful interface with progress bars and formatted output
4. **Caching**: Hardware detection results cached for faster startup
5. **Configuration Persistence**: User preferences saved between sessions
6. **Comprehensive Testing**: Unit tests with mocking for hardware detection
7. **Error Handling**: Specific exception handling with graceful degradation
8. **Export Formats**: JSON, CSV, and YAML export options
9. **Type Safety**: Complete type annotations throughout
10. **Logging**: Structured logging with Rich formatter

## Configuration Files

- **User config**: `~/.model-selector/config.json`
- **Hardware cache**: `~/.model-selector/hardware_cache.json`
- **Models database**: `models.json` (can be overridden with `--models-file`)

## Testing

The test suite includes:
- Hardware detection mocking for different platforms
- Model recommendation scoring validation
- Configuration persistence testing
- Error handling verification
- Data class validation

Run specific test files:
```bash
pytest tests/test_hardware.py -v
pytest tests/test_models.py -v
```