"""
Tests for model management module
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from model_selector.hardware import MemoryInfo
from model_selector.models import ModelData, ModelManager, ModelVariant


class TestModelManager:
    """Test model management functionality"""
    
    @pytest.fixture
    def sample_models_data(self):
        """Sample model data for testing"""
        return {
            "models": {
                "test_model": {
                    "base_name": "Test Model",
                    "variants": {
                        "1B-Q4_K_M": {"params": "1B", "quant": "Q4_K_M", "vram": 1.0, "ram": 1.5},
                        "1B-F16": {"params": "1B", "quant": "F16", "vram": 2.0, "ram": 2.5}
                    },
                    "context": 8000,
                    "scores": {"knowledge": 3, "math": 2, "code": 3, "reasoning": 3, "language": 4},
                    "tools": True,
                    "languages": ["English"],
                    "docker": "ai/test-model",
                    "features": ["test_feature"]
                },
                "embed_model": {
                    "base_name": "Test Embed",
                    "variants": {
                        "335M-F16": {"params": "335M", "quant": "F16", "vram": 0.5, "ram": 0.8}
                    },
                    "context": 512,
                    "scores": {"retrieval": 4, "classification": 3},
                    "tools": False,
                    "languages": ["English"],
                    "docker": "ai/test-embed",
                    "type": "embedding"
                }
            }
        }
    
    @pytest.fixture
    def models_file(self, sample_models_data):
        """Create temporary models file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_models_data, f)
            temp_path = Path(f.name)
        
        yield temp_path
        temp_path.unlink()
    
    @pytest.fixture
    def manager(self, models_file):
        """Create ModelManager instance"""
        return ModelManager(models_file)
    
    def test_load_models(self, manager):
        """Test model loading from JSON"""
        assert len(manager.models) == 2
        assert "test_model" in manager.models
        assert "embed_model" in manager.models
        
        test_model = manager.models["test_model"]
        assert test_model.base_name == "Test Model"
        assert test_model.context == 8000
        assert test_model.tools is True
        assert test_model.model_type == "generative"
        assert len(test_model.variants) == 2
        
        embed_model = manager.models["embed_model"]
        assert embed_model.model_type == "embedding"
    
    def test_get_model(self, manager):
        """Test getting model by ID"""
        model = manager.get_model("test_model")
        assert model is not None
        assert model.base_name == "Test Model"
        
        # Test non-existent model
        assert manager.get_model("nonexistent") is None
    
    def test_get_models_by_type(self, manager):
        """Test filtering models by type"""
        generative_models = manager.get_models_by_type("generative")
        assert len(generative_models) == 1
        assert "test_model" in generative_models
        
        embedding_models = manager.get_models_by_type("embedding")
        assert len(embedding_models) == 1
        assert "embed_model" in embedding_models
    
    def test_get_embedding_models(self, manager):
        """Test getting embedding models"""
        embedding_models = manager.get_embedding_models()
        assert len(embedding_models) == 1
        assert "embed_model" in embedding_models
    
    def test_get_generative_models(self, manager):
        """Test getting generative models"""
        generative_models = manager.get_generative_models()
        assert len(generative_models) == 1
        assert "test_model" in generative_models
    
    def test_get_quantization_recommendation(self, manager):
        """Test quantization recommendation"""
        # Test with enough memory for F16
        variant = manager.get_quantization_recommendation("test_model", 3.0, "gpu")
        assert variant == "1B-F16"  # Should prefer highest quality
        
        # Test with limited memory
        variant = manager.get_quantization_recommendation("test_model", 1.5, "gpu")
        assert variant == "1B-Q4_K_M"  # Should fall back to lower quality
        
        # Test with insufficient memory
        variant = manager.get_quantization_recommendation("test_model", 0.5, "gpu")
        assert variant is None
    
    def test_calculate_model_score(self, manager):
        """Test model scoring"""
        model = manager.get_model("test_model")
        
        # Test code use case
        score = manager.calculate_model_score(model, "code", 2, ["english"], True)
        assert score > 0
        
        # Test penalty for missing tools
        score_without_tools = manager.calculate_model_score(model, "code", 2, ["english"], False)
        assert score_without_tools > score  # No penalty since model has tools
        
        # Test context bonus
        score_high_context = manager.calculate_model_score(model, "code", 4, ["english"], True)
        # Should be same as model context is low
        
        # Test embedding model (should return 0)
        embed_model = manager.get_model("embed_model")
        score = manager.calculate_model_score(embed_model, "code", 2, ["english"], True)
        assert score == 0.0
    
    def test_recommend_models(self, manager):
        """Test model recommendation"""
        memory_info = MemoryInfo(vram=4.0, ram=6.0, mode="gpu")
        
        recommendations = manager.recommend_models(
            memory_info, "code", 2, ["english"], True, top_n=5
        )
        
        assert len(recommendations) == 1  # Only one generative model
        model_id, model_data, score, variant = recommendations[0]
        assert model_id == "test_model"
        assert score > 0
        assert variant in ["1B-Q4_K_M", "1B-F16"]
    
    def test_get_docker_command(self, manager):
        """Test Docker command generation"""
        cmd = manager.get_docker_command("test_model", "1B-Q4_K_M")
        assert cmd == "docker run -it --rm ai/test-model:1B-Q4_K_M"
        
        # Test invalid model
        cmd = manager.get_docker_command("nonexistent", "1B-Q4_K_M")
        assert cmd == ""
    
    def test_get_model_memory_requirements(self, manager):
        """Test memory requirements retrieval"""
        vram, ram = manager.get_model_memory_requirements("test_model", "1B-Q4_K_M")
        assert vram == 1.0
        assert ram == 1.5
        
        # Test invalid model
        result = manager.get_model_memory_requirements("nonexistent", "1B-Q4_K_M")
        assert result is None
    
    def test_filter_models_by_memory(self, manager):
        """Test filtering models by memory"""
        memory_info = MemoryInfo(vram=1.5, ram=2.0, mode="gpu")
        
        suitable_models = manager.filter_models_by_memory(memory_info, "generative")
        assert "test_model" in suitable_models
        assert "1B-Q4_K_M" in suitable_models["test_model"]
        assert "1B-F16" not in suitable_models["test_model"]  # Too much VRAM
    
    def test_export_recommendations_json(self, manager):
        """Test exporting recommendations as JSON"""
        model = manager.get_model("test_model")
        recommendations = [("test_model", model, 85.0, "1B-Q4_K_M")]
        
        json_export = manager.export_recommendations(recommendations, "json")
        data = json.loads(json_export)
        
        assert len(data) == 1
        assert data[0]["model_id"] == "test_model"
        assert data[0]["score"] == 85.0
        assert data[0]["quantization"] == "Q4_K_M"
    
    def test_export_recommendations_csv(self, manager):
        """Test exporting recommendations as CSV"""
        model = manager.get_model("test_model")
        recommendations = [("test_model", model, 85.0, "1B-Q4_K_M")]
        
        csv_export = manager.export_recommendations(recommendations, "csv")
        assert "model_id,name,quantization" in csv_export
        assert "test_model" in csv_export
        assert "Q4_K_M" in csv_export
    
    def test_export_recommendations_yaml(self, manager):
        """Test exporting recommendations as YAML"""
        model = manager.get_model("test_model")
        recommendations = [("test_model", model, 85.0, "1B-Q4_K_M")]
        
        # Test with PyYAML available
        with patch('yaml.dump') as mock_dump:
            mock_dump.return_value = "yaml_output"
            yaml_export = manager.export_recommendations(recommendations, "yaml")
            assert yaml_export == "yaml_output"
        
        # Test without PyYAML (should fallback to JSON)
        with patch('model_selector.models.yaml', None):
            yaml_export = manager.export_recommendations(recommendations, "yaml")
            # Should be valid JSON
            data = json.loads(yaml_export)
            assert len(data) == 1


class TestModelData:
    """Test ModelData and ModelVariant classes"""
    
    def test_model_variant_creation(self):
        """Test ModelVariant creation"""
        variant = ModelVariant(params="1B", quant="Q4_K_M", vram=1.0, ram=1.5)
        assert variant.params == "1B"
        assert variant.quant == "Q4_K_M"
        assert variant.vram == 1.0
        assert variant.ram == 1.5
    
    def test_model_data_creation(self):
        """Test ModelData creation"""
        variants = {
            "1B-Q4_K_M": ModelVariant(params="1B", quant="Q4_K_M", vram=1.0, ram=1.5)
        }
        
        model = ModelData(
            base_name="Test Model",
            variants=variants,
            context=8000,
            scores={"knowledge": 3, "code": 4},
            tools=True,
            languages=["English"],
            docker="ai/test-model",
            features=["test_feature"],
            model_type="generative"
        )
        
        assert model.base_name == "Test Model"
        assert model.context == 8000
        assert model.tools is True
        assert model.model_type == "generative"
        assert len(model.variants) == 1
        assert len(model.features) == 1


if __name__ == "__main__":
    pytest.main([__file__])