"""
Model management and recommendation logic
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .hardware import MemoryInfo

logger = logging.getLogger(__name__)


@dataclass
class ModelVariant:
    """Model variant specification"""
    params: str
    quant: str
    vram: float
    ram: float


@dataclass
class ModelData:
    """Complete model information"""
    base_name: str
    variants: Dict[str, ModelVariant]
    context: int
    scores: Dict[str, int]
    tools: bool
    languages: List[str]
    docker: str
    features: List[str] = field(default_factory=list)
    model_type: str = "generative"  # "generative" or "embedding"


class ModelManager:
    """Manages model database and recommendations"""
    
    QUANTIZATION_PRIORITY = {
        "F16": 100,      # Maximum quality
        "Q8_0": 90,      # Very high quality
        "Q6_K": 80,      # High quality
        "Q4_K_M": 70,    # Good quality (recommended)
        "Q4_0": 60,      # Acceptable quality
        "Q2_K": 30       # Low quality
    }
    
    USE_CASE_WEIGHTS = {
        "rag": {"knowledge": 0.3, "language": 0.3, "reasoning": 0.2, "multilingual": 0.2},
        "code": {"code": 0.5, "reasoning": 0.3, "language": 0.2},
        "chat": {"language": 0.4, "reasoning": 0.3, "knowledge": 0.3},
        "analysis": {"reasoning": 0.4, "knowledge": 0.3, "language": 0.3},
        "multilingual": {"multilingual": 0.5, "language": 0.3, "knowledge": 0.2},
        "education": {"math": 0.4, "reasoning": 0.3, "language": 0.3},
        "creative": {"language": 0.5, "reasoning": 0.3, "knowledge": 0.2},
        "general": {"knowledge": 0.25, "reasoning": 0.25, "language": 0.25, "code": 0.25}
    }
    
    def __init__(self, models_file: Optional[Path] = None):
        self.models_file = models_file or Path(__file__).parent.parent / "models.json"
        self.models: Dict[str, ModelData] = {}
        self._load_models()
    
    def _load_models(self) -> None:
        """Load models from JSON file"""
        try:
            with open(self.models_file, 'r') as f:
                data = json.load(f)
            
            for model_id, model_info in data["models"].items():
                variants = {}
                for variant_id, variant_info in model_info["variants"].items():
                    variants[variant_id] = ModelVariant(**variant_info)
                
                self.models[model_id] = ModelData(
                    base_name=model_info["base_name"],
                    variants=variants,
                    context=model_info["context"],
                    scores=model_info["scores"],
                    tools=model_info["tools"],
                    languages=model_info["languages"],
                    docker=model_info["docker"],
                    features=model_info.get("features", []),
                    model_type=model_info.get("type", "generative")
                )
                
            logger.info(f"Loaded {len(self.models)} models from {self.models_file}")
            
        except FileNotFoundError:
            logger.error(f"Models file not found: {self.models_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in models file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_model(self, model_id: str) -> Optional[ModelData]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def get_models_by_type(self, model_type: str) -> Dict[str, ModelData]:
        """Get all models of a specific type"""
        return {
            model_id: model_data
            for model_id, model_data in self.models.items()
            if model_data.model_type == model_type
        }
    
    def get_embedding_models(self) -> Dict[str, ModelData]:
        """Get all embedding models"""
        return self.get_models_by_type("embedding")
    
    def get_generative_models(self) -> Dict[str, ModelData]:
        """Get all generative models"""
        return self.get_models_by_type("generative")
    
    def get_quantization_recommendation(
        self, 
        model_id: str, 
        available_memory: float, 
        mode: str
    ) -> Optional[str]:
        """Recommends the best quantization based on available memory"""
        model = self.get_model(model_id)
        if not model:
            return None
        
        suitable_variants = []
        
        for variant_id, variant in model.variants.items():
            memory_req = variant.vram if mode == "gpu" else variant.ram
            if memory_req <= available_memory:
                priority = self.QUANTIZATION_PRIORITY.get(variant.quant, 50)
                suitable_variants.append((variant_id, priority, memory_req))
        
        if not suitable_variants:
            return None
        
        # Sort by priority (quality) descending
        suitable_variants.sort(key=lambda x: x[1], reverse=True)
        return suitable_variants[0][0]  # Return highest quality variant that fits
    
    def calculate_model_score(
        self,
        model_data: ModelData,
        use_case: str,
        context_importance: int,
        languages: List[str],
        needs_tools: bool
    ) -> float:
        """Calculates model score based on requirements"""
        if model_data.model_type == "embedding":
            return 0.0  # Embedding models don't get scored the same way
        
        score = 0.0
        weights = self.USE_CASE_WEIGHTS.get(use_case, self.USE_CASE_WEIGHTS["general"])
        
        # Base score from capabilities
        for metric, weight in weights.items():
            if metric in model_data.scores:
                score += model_data.scores[metric] * weight * 20
        
        # Context length bonus
        if context_importance >= 3 and model_data.context >= 100000:
            score += 20
        elif context_importance >= 2 and model_data.context >= 40000:
            score += 10
        
        # Language support bonus
        multilingual_score = model_data.scores.get("multilingual", 0)
        if "many" in languages and multilingual_score >= 4:
            score += 15
        elif "european" in languages and multilingual_score >= 3:
            score += 10
        elif "spanish" in languages and multilingual_score >= 2:
            score += 5
        elif "asian" in languages and multilingual_score >= 3:
            score += 12
        
        # Tool calling penalty
        if needs_tools and not model_data.tools:
            score -= 20
        
        # Feature bonuses
        feature_bonuses = {
            "multimodal": 5,
            "thinking_modes": 8,
            "reasoning_optimized": 10,
            "agent_capabilities": 6
        }
        
        for feature in model_data.features:
            if feature in feature_bonuses:
                score += feature_bonuses[feature]
        
        return max(0.0, score)
    
    def recommend_models(
        self,
        available_memory: MemoryInfo,
        use_case: str,
        context_importance: int,
        languages: List[str],
        needs_tools: bool,
        top_n: int = 5
    ) -> List[Tuple[str, ModelData, float, str]]:
        """Recommends best models based on requirements and hardware"""
        
        # Calculate available memory for generative models
        if use_case == "rag":
            # Reserve memory for embedding model
            embed_model = self.get_model("mxbai-embed")
            if embed_model:
                embed_variant = embed_model.variants.get("335M-F16")
                if embed_variant:
                    if available_memory.mode == "gpu":
                        remaining_memory = available_memory.vram - embed_variant.vram - 0.5
                    else:
                        remaining_memory = available_memory.ram - embed_variant.ram - 1.0
                else:
                    remaining_memory = available_memory.vram if available_memory.mode == "gpu" else available_memory.ram
            else:
                remaining_memory = available_memory.vram if available_memory.mode == "gpu" else available_memory.ram
        else:
            if available_memory.mode == "gpu":
                remaining_memory = available_memory.vram - 0.5
            else:
                remaining_memory = available_memory.ram - 1.0
        
        suitable_models = []
        generative_models = self.get_generative_models()
        
        for model_id, model_data in generative_models.items():
            # Find best quantization for this model
            best_variant = self.get_quantization_recommendation(
                model_id, remaining_memory, available_memory.mode
            )
            
            if best_variant:
                score = self.calculate_model_score(
                    model_data, use_case, context_importance, languages, needs_tools
                )
                
                suitable_models.append((model_id, model_data, score, best_variant))
        
        # Sort by score descending
        suitable_models.sort(key=lambda x: x[2], reverse=True)
        
        return suitable_models[:top_n]
    
    def get_docker_command(self, model_id: str, variant_id: str) -> str:
        """Generate Docker command for model"""
        model = self.get_model(model_id)
        if not model:
            return ""
        
        variant = model.variants.get(variant_id)
        if not variant:
            return ""
        
        return f"docker run -it --rm {model.docker}:{variant.params}-{variant.quant}"
    
    def get_model_memory_requirements(self, model_id: str, variant_id: str) -> Optional[Tuple[float, float]]:
        """Get memory requirements for a specific model variant"""
        model = self.get_model(model_id)
        if not model:
            return None
        
        variant = model.variants.get(variant_id)
        if not variant:
            return None
        
        return (variant.vram, variant.ram)
    
    def filter_models_by_memory(
        self, 
        available_memory: MemoryInfo, 
        model_type: str = "generative"
    ) -> Dict[str, List[str]]:
        """Filter models that can fit in available memory"""
        suitable_models = {}
        models = self.get_models_by_type(model_type)
        
        for model_id, model_data in models.items():
            suitable_variants = []
            
            for variant_id, variant in model_data.variants.items():
                memory_req = variant.vram if available_memory.mode == "gpu" else variant.ram
                if memory_req <= (available_memory.vram if available_memory.mode == "gpu" else available_memory.ram):
                    suitable_variants.append(variant_id)
            
            if suitable_variants:
                suitable_models[model_id] = suitable_variants
        
        return suitable_models
    
    def update_model_database(self, new_models_data: Dict) -> None:
        """Update the model database with new data"""
        try:
            with open(self.models_file, 'w') as f:
                json.dump(new_models_data, f, indent=2)
            
            # Reload models
            self._load_models()
            logger.info("Model database updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update model database: {e}")
            raise
    
    def export_recommendations(
        self, 
        recommendations: List[Tuple[str, ModelData, float, str]], 
        format: str = "json"
    ) -> str:
        """Export recommendations in various formats"""
        data = []
        
        for model_id, model_data, score, variant_id in recommendations:
            variant = model_data.variants[variant_id]
            entry = {
                "model_id": model_id,
                "name": f"{model_data.base_name} {variant.params}",
                "quantization": variant.quant,
                "score": round(score, 2),
                "memory_requirements": {
                    "vram": variant.vram,
                    "ram": variant.ram
                },
                "context_length": model_data.context,
                "languages": model_data.languages,
                "tools": model_data.tools,
                "features": model_data.features,
                "docker_command": self.get_docker_command(model_id, variant_id)
            }
            data.append(entry)
        
        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "csv":
            return self._export_csv(data)
        elif format == "yaml":
            return self._export_yaml(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_csv(self, data: List[Dict]) -> str:
        """Export data as CSV"""
        import csv
        import io
        
        output = io.StringIO()
        if not data:
            return ""
        
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        
        for row in data:
            # Flatten nested dictionaries
            flat_row = {}
            for key, value in row.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_row[f"{key}_{sub_key}"] = sub_value
                elif isinstance(value, list):
                    flat_row[key] = ", ".join(str(item) for item in value)
                else:
                    flat_row[key] = value
            
            writer.writerow(flat_row)
        
        return output.getvalue()
    
    def _export_yaml(self, data: List[Dict]) -> str:
        """Export data as YAML"""
        try:
            import yaml
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        except ImportError:
            logger.warning("PyYAML not installed, falling back to JSON")
            return json.dumps(data, indent=2)