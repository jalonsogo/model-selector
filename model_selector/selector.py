"""
Model selection interface with rich CLI
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from .hardware import HardwareDetector, SystemInfo, MemoryInfo
from .models import ModelManager, ModelData

logger = logging.getLogger(__name__)


class ModelSelector:
    """Enhanced model selector with rich CLI interface"""
    
    def __init__(self, models_file: Optional[Path] = None):
        self.console = Console()
        self.hardware_detector = HardwareDetector()
        self.model_manager = ModelManager(models_file)
        self.system_info: Optional[SystemInfo] = None
        self.available_memory: Optional[MemoryInfo] = None
        self.user_config = self._load_user_config()
        
    def _load_user_config(self) -> Dict:
        """Load user configuration from file"""
        config_file = Path.home() / ".model-selector" / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load user config: {e}")
        return {}
    
    def _save_user_config(self, config: Dict) -> None:
        """Save user configuration to file"""
        config_file = Path.home() / ".model-selector" / "config.json"
        config_file.parent.mkdir(exist_ok=True)
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save user config: {e}")
    
    async def detect_hardware(self, use_cache: bool = True) -> None:
        """Detect hardware with progress indicator"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Detecting hardware...", total=None)
            
            try:
                self.system_info = await self.hardware_detector.detect_system_info(use_cache)
                self.available_memory = self.hardware_detector.get_available_memory()
                progress.update(task, description="✅ Hardware detection complete")
            except Exception as e:
                progress.update(task, description=f"❌ Hardware detection failed: {e}")
                raise
    
    def display_hardware_info(self) -> None:
        """Display hardware information in a formatted panel"""
        if not self.system_info:
            self.console.print("❌ No hardware information available", style="red")
            return
        
        # System info table
        system_table = Table(title="System Information", show_header=False)
        system_table.add_column("Property", style="cyan")
        system_table.add_column("Value", style="white")
        
        system_table.add_row("Operating System", f"{self.system_info.os} ({self.system_info.arch})")
        system_table.add_row("CPU", self.system_info.cpu.name)
        system_table.add_row("CPU Cores", str(self.system_info.cpu.cores))
        system_table.add_row("Total RAM", f"{self.system_info.ram} GB")
        
        # GPU info table
        if self.system_info.gpus:
            gpu_table = Table(title="GPU Information", show_header=True)
            gpu_table.add_column("Name", style="cyan")
            gpu_table.add_column("VRAM", style="yellow")
            gpu_table.add_column("Vendor", style="green")
            gpu_table.add_column("Notes", style="dim")
            
            for gpu in self.system_info.gpus:
                gpu_table.add_row(
                    gpu.name,
                    f"{gpu.vram} GB",
                    gpu.vendor,
                    gpu.note or ""
                )
        
        # Memory availability
        if self.available_memory:
            memory_table = Table(title="Available Memory", show_header=False)
            memory_table.add_column("Resource", style="cyan")
            memory_table.add_column("Available", style="green")
            
            memory_table.add_row("RAM", f"{self.available_memory.ram:.1f} GB")
            if self.available_memory.mode == "gpu":
                memory_table.add_row("VRAM", f"{self.available_memory.vram:.1f} GB")
            memory_table.add_row("Mode", self.available_memory.mode.upper())
            if self.available_memory.note:
                memory_table.add_row("Note", self.available_memory.note)
        
        # Display all tables
        self.console.print(Panel(system_table, title="🖥️ Hardware Detection Results", border_style="blue"))
        if self.system_info.gpus:
            self.console.print(Panel(gpu_table, title="🎮 GPU Details", border_style="green"))
        if self.available_memory:
            self.console.print(Panel(memory_table, title="💾 Memory Analysis", border_style="yellow"))
    
    def ask_use_case(self) -> str:
        """Ask about the main use case with rich prompts"""
        self.console.print(Panel(
            "[bold cyan]What will you primarily use the model for?[/bold cyan]\\n\\n"
            "[dim]1.[/dim] RAG (Retrieval-Augmented Generation)\\n"
            "[dim]2.[/dim] Programming assistant\\n"
            "[dim]3.[/dim] Conversational chatbot\\n"
            "[dim]4.[/dim] Document analysis and processing\\n"
            "[dim]5.[/dim] Translation and multilingual tasks\\n"
            "[dim]6.[/dim] Education and tutoring (math/science)\\n"
            "[dim]7.[/dim] Creative writing\\n"
            "[dim]8.[/dim] General use/multiple purposes",
            title="🎯 Step 1: Use Case Selection",
            border_style="blue"
        ))
        
        use_case_map = {
            1: "rag", 2: "code", 3: "chat", 4: "analysis",
            5: "multilingual", 6: "education", 7: "creative", 8: "general"
        }
        
        # Check for saved preference
        if "use_case" in self.user_config:
            if Confirm.ask(f"Use previous selection: {self.user_config['use_case']}?"):
                return self.user_config["use_case"]
        
        while True:
            try:
                choice = IntPrompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "6", "7", "8"])
                use_case = use_case_map[choice]
                
                # Save preference
                self.user_config["use_case"] = use_case
                self._save_user_config(self.user_config)
                
                return use_case
            except (ValueError, KeyError):
                self.console.print("[red]Invalid choice. Please select 1-8.[/red]")
    
    def ask_context_importance(self) -> int:
        """Ask about context length importance"""
        self.console.print(Panel(
            "[bold cyan]How important is it to process very long texts?[/bold cyan]\\n\\n"
            "[dim]1.[/dim] Not important (short texts are fine)\\n"
            "[dim]2.[/dim] Somewhat important (medium documents)\\n"
            "[dim]3.[/dim] Very important (need to process long documents)\\n"
            "[dim]4.[/dim] Critical (need maximum context possible)",
            title="📏 Step 2: Context Length Importance",
            border_style="blue"
        ))
        
        # Check for saved preference
        if "context_importance" in self.user_config:
            if Confirm.ask(f"Use previous selection: {self.user_config['context_importance']}?"):
                return self.user_config["context_importance"]
        
        while True:
            try:
                choice = IntPrompt.ask("Select an option", choices=["1", "2", "3", "4"])
                
                # Save preference
                self.user_config["context_importance"] = choice
                self._save_user_config(self.user_config)
                
                return choice
            except ValueError:
                self.console.print("[red]Invalid choice. Please select 1-4.[/red]")
    
    def ask_languages(self) -> List[str]:
        """Ask about required languages"""
        self.console.print(Panel(
            "[bold cyan]Which languages do you need the model to support?[/bold cyan]\\n\\n"
            "[dim]1.[/dim] English only\\n"
            "[dim]2.[/dim] English and Spanish\\n"
            "[dim]3.[/dim] Main European languages\\n"
            "[dim]4.[/dim] I need support for many languages (20+)\\n"
            "[dim]5.[/dim] I need support for Asian languages",
            title="🌍 Step 3: Language Requirements",
            border_style="blue"
        ))
        
        languages_map = {
            1: ["english"],
            2: ["english", "spanish"],
            3: ["european"],
            4: ["many"],
            5: ["asian"]
        }
        
        # Check for saved preference
        if "languages" in self.user_config:
            if Confirm.ask(f"Use previous selection: {self.user_config['languages']}?"):
                return self.user_config["languages"]
        
        while True:
            try:
                choice = IntPrompt.ask("Select an option", choices=["1", "2", "3", "4", "5"])
                languages = languages_map[choice]
                
                # Save preference
                self.user_config["languages"] = languages
                self._save_user_config(self.user_config)
                
                return languages
            except (ValueError, KeyError):
                self.console.print("[red]Invalid choice. Please select 1-5.[/red]")
    
    def ask_tool_calling(self) -> bool:
        """Ask about tool calling needs"""
        self.console.print(Panel(
            "[bold cyan]Do you need the model to call external functions/APIs?[/bold cyan]\\n\\n"
            "[dim]1.[/dim] Yes, it's important\\n"
            "[dim]2.[/dim] Not necessary\\n"
            "[dim]3.[/dim] I don't know what this is",
            title="🔧 Step 4: Tool Integration",
            border_style="blue"
        ))
        
        # Check for saved preference
        if "needs_tools" in self.user_config:
            if Confirm.ask(f"Use previous selection: {self.user_config['needs_tools']}?"):
                return self.user_config["needs_tools"]
        
        while True:
            try:
                choice = IntPrompt.ask("Select an option", choices=["1", "2", "3"])
                needs_tools = choice == 1
                
                # Save preference
                self.user_config["needs_tools"] = needs_tools
                self._save_user_config(self.user_config)
                
                return needs_tools
            except ValueError:
                self.console.print("[red]Invalid choice. Please select 1-3.[/red]")
    
    def display_recommendations(
        self, 
        recommendations: List[Tuple[str, ModelData, float, str]], 
        use_case: str
    ) -> None:
        """Display recommendations in a formatted table"""
        if not recommendations:
            self.console.print(Panel(
                "[red]❌ No models found that meet your requirements.[/red]\\n"
                "Your hardware may have limitations. Consider:\\n"
                "• Using cloud services\\n"
                "• Upgrading your hardware\\n"
                "• Reducing quality requirements",
                title="No Suitable Models",
                border_style="red"
            ))
            return
        
        # Show embedding model info for RAG
        if use_case == "rag":
            embed_info = Panel(
                "[bold cyan]📊 EMBEDDING MODEL (Required for RAG)[/bold cyan]\\n"
                "• mxbai-embed-large (0.63 GB VRAM)\\n"
                "• Optimized for semantic search\\n"
                "• Required alongside generative model",
                title="RAG Components",
                border_style="cyan"
            )
            self.console.print(embed_info)
        
        # Create recommendations table
        table = Table(title=f"🎯 Model Recommendations - {self.available_memory.mode.upper()} Mode")
        table.add_column("Rank", style="bold cyan", width=4)
        table.add_column("Model", style="bold white", min_width=20)
        table.add_column("Memory", style="yellow", justify="right")
        table.add_column("Context", style="green", justify="right")
        table.add_column("Score", style="magenta", justify="center")
        table.add_column("Languages", style="blue", max_width=20)
        table.add_column("Docker Command", style="dim", max_width=30)
        
        for i, (model_id, model_data, score, variant_id) in enumerate(recommendations, 1):
            variant = model_data.variants[variant_id]
            
            # Format memory requirement
            if self.available_memory.mode == "gpu":
                memory_str = f"{variant.vram:.1f} GB VRAM"
            else:
                memory_str = f"{variant.ram:.1f} GB RAM"
                if self.available_memory.mode == "cpu":
                    memory_str += " ⚠️ CPU"
            
            # Format context
            context_str = f"{model_data.context:,}"
            
            # Format score with stars
            stars = "⭐" * min(5, int(score / 20))
            score_str = f"{stars} ({score:.0f})"
            
            # Format languages
            languages = ", ".join(model_data.languages)
            if len(languages) > 20:
                languages = languages[:17] + "..."
            
            # Docker command
            docker_cmd = f"docker run {model_data.docker}:{variant.params}-{variant.quant}"
            if len(docker_cmd) > 30:
                docker_cmd = docker_cmd[:27] + "..."
            
            # Special formatting for top recommendation
            if i == 1:
                rank_style = "bold green"
                model_style = "bold green"
            else:
                rank_style = "cyan"
                model_style = "white"
            
            table.add_row(
                f"#{i}",
                f"{model_data.base_name} {variant.params} ({variant.quant})",
                memory_str,
                context_str,
                score_str,
                languages,
                docker_cmd,
                style=rank_style if i == 1 else None
            )
        
        self.console.print(Panel(table, title="🤖 Recommended Models", border_style="green"))
        
        # Show features for top recommendation
        if recommendations:
            top_model = recommendations[0][1]
            if top_model.features:
                features_text = Text()
                features_text.append("🚀 Special Features: ", style="bold cyan")
                features_text.append(", ".join(top_model.features), style="green")
                self.console.print(Panel(features_text, border_style="cyan"))
    
    def display_summary(self, recommendations: List[Tuple[str, ModelData, float, str]]) -> None:
        """Display configuration summary"""
        if not self.system_info or not self.available_memory:
            return
        
        # System summary
        summary_table = Table(title="Configuration Summary", show_header=False)
        summary_table.add_column("Property", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("CPU", f"{self.system_info.cpu.cores} cores")
        summary_table.add_row("RAM", f"{self.system_info.ram} GB total ({self.available_memory.ram:.1f} GB available)")
        
        if self.system_info.gpus:
            gpu = self.system_info.gpus[0]
            summary_table.add_row("GPU", f"{gpu.name} ({gpu.vram} GB VRAM)")
            summary_table.add_row("Mode", f"GPU ({self.available_memory.vram:.1f} GB VRAM available)")
        else:
            summary_table.add_row("GPU", "Not detected")
            summary_table.add_row("Mode", "CPU (may be slower)")
        
        # User preferences
        if self.user_config:
            summary_table.add_row("Use Case", self.user_config.get("use_case", "Unknown"))
            context_stars = "⭐" * self.user_config.get("context_importance", 0)
            summary_table.add_row("Context Importance", context_stars)
            summary_table.add_row("Tool Calling", "Yes" if self.user_config.get("needs_tools", False) else "No")
        
        self.console.print(Panel(summary_table, title="📋 Session Summary", border_style="blue"))
    
    def save_results(self, recommendations: List[Tuple[str, ModelData, float, str]]) -> None:
        """Save results to file with format choice"""
        if not recommendations:
            return
        
        if not Confirm.ask("Would you like to save these recommendations?"):
            return
        
        # Choose format
        self.console.print("\\n[cyan]Choose export format:[/cyan]")
        self.console.print("1. JSON (default)")
        self.console.print("2. CSV")
        self.console.print("3. YAML")
        
        format_choice = Prompt.ask("Select format", choices=["1", "2", "3"], default="1")
        format_map = {"1": "json", "2": "csv", "3": "yaml"}
        export_format = format_map[format_choice]
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_recommendations_{timestamp}.{export_format}"
        filepath = Path(filename)
        
        try:
            # Export recommendations
            export_data = self.model_manager.export_recommendations(
                recommendations, export_format
            )
            
            # Save to file
            with open(filepath, 'w') as f:
                f.write(export_data)
            
            self.console.print(f"✅ Recommendations saved to [bold green]{filepath}[/bold green]")
            
            # Also save configuration
            config_data = {
                "timestamp": timestamp,
                "hardware": {
                    "os": self.system_info.os,
                    "arch": self.system_info.arch,
                    "cpu": {
                        "name": self.system_info.cpu.name,
                        "cores": self.system_info.cpu.cores
                    },
                    "ram": self.system_info.ram,
                    "gpus": [
                        {
                            "name": gpu.name,
                            "vram": gpu.vram,
                            "vendor": gpu.vendor
                        }
                        for gpu in self.system_info.gpus
                    ]
                },
                "requirements": self.user_config,
                "available_memory": {
                    "vram": self.available_memory.vram,
                    "ram": self.available_memory.ram,
                    "mode": self.available_memory.mode
                }
            }
            
            config_filepath = Path(f"config_{timestamp}.json")
            with open(config_filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.console.print(f"📋 Configuration saved to [bold blue]{config_filepath}[/bold blue]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Failed to save results: {e}[/red]")
    
    async def run_interactive(self) -> None:
        """Run the interactive model selection process"""
        self.console.print(Panel(
            "[bold cyan]🤖 AI Model Selection Assistant[/bold cyan]\\n"
            "Intelligent hardware detection and model recommendations",
            title="Welcome",
            border_style="blue"
        ))
        
        try:
            # Hardware detection
            await self.detect_hardware()
            self.display_hardware_info()
            
            # Ask if user wants to continue with detected hardware
            if not Confirm.ask("\\nProceed with this hardware configuration?"):
                self.console.print("[yellow]Manual hardware configuration not implemented yet.[/yellow]")
                return
            
            # Collect requirements
            self.console.print("\\n[bold cyan]Let's determine your requirements...[/bold cyan]")
            use_case = self.ask_use_case()
            context_importance = self.ask_context_importance()
            languages = self.ask_languages()
            needs_tools = self.ask_tool_calling()
            
            # Get recommendations
            self.console.print("\\n[bold cyan]Analyzing models...[/bold cyan]")
            recommendations = self.model_manager.recommend_models(
                self.available_memory, use_case, context_importance, languages, needs_tools
            )
            
            # Display results
            self.console.print("\\n")
            self.display_recommendations(recommendations, use_case)
            self.display_summary(recommendations)
            
            # Save results
            self.save_results(recommendations)
            
        except KeyboardInterrupt:
            self.console.print("\\n[yellow]Operation cancelled by user.[/yellow]")
        except Exception as e:
            self.console.print(f"\\n[red]❌ Error: {e}[/red]")
            logger.exception("Unexpected error in interactive mode")
    
    def run_batch(self, config: Dict) -> Dict:
        """Run in batch mode with provided configuration"""
        # TODO: Implement batch mode
        raise NotImplementedError("Batch mode not implemented yet")