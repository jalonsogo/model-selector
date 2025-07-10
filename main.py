#!/usr/bin/env python3
"""
AI Model Selection Assistant
Main entry point with CLI interface
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler

from model_selector import ModelSelector
from model_selector.hardware import HardwareDetector
from model_selector.models import ModelManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--models-file', type=click.Path(exists=True), help='Path to models.json file')
@click.pass_context
def cli(ctx, verbose: bool, models_file: Optional[str]):
    """AI Model Selection Assistant - Choose the perfect model for your hardware"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Store common options in context
    ctx.ensure_object(dict)
    ctx.obj['models_file'] = Path(models_file) if models_file else None
    ctx.obj['verbose'] = verbose


@cli.command()
@click.pass_context
def interactive(ctx):
    """Run the interactive model selection process"""
    console = Console()
    
    try:
        selector = ModelSelector(ctx.obj['models_file'])
        asyncio.run(selector.run_interactive())
    except KeyboardInterrupt:
        console.print("\\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if ctx.obj['verbose']:
            logger.exception("Detailed error information")
        sys.exit(1)


@cli.command()
@click.option('--use-cache/--no-cache', default=True, help='Use hardware detection cache')
@click.pass_context
def hardware(ctx, use_cache: bool):
    """Display detailed hardware information"""
    console = Console()
    
    async def detect_and_display():
        try:
            detector = HardwareDetector()
            system_info = await detector.detect_system_info(use_cache)
            available_memory = detector.get_available_memory()
            
            # Create a temporary selector just for display
            selector = ModelSelector(ctx.obj['models_file'])
            selector.system_info = system_info
            selector.available_memory = available_memory
            selector.display_hardware_info()
            
        except Exception as e:
            console.print(f"[red]❌ Hardware detection failed: {e}[/red]")
            if ctx.obj['verbose']:
                logger.exception("Detailed error information")
            sys.exit(1)
    
    try:
        asyncio.run(detect_and_display())
    except KeyboardInterrupt:
        console.print("\\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)


@cli.command()
@click.option('--model-id', help='Specific model ID to show details for')
@click.option('--type', 'model_type', type=click.Choice(['generative', 'embedding', 'all']), 
              default='all', help='Filter by model type')
@click.pass_context
def models(ctx, model_id: Optional[str], model_type: str):
    """List available models and their specifications"""
    console = Console()
    
    try:
        manager = ModelManager(ctx.obj['models_file'])
        
        if model_id:
            # Show specific model details
            model = manager.get_model(model_id)
            if not model:
                console.print(f"[red]❌ Model '{model_id}' not found[/red]")
                sys.exit(1)
            
            from rich.table import Table
            from rich.panel import Panel
            
            # Model info table
            info_table = Table(title=f"Model: {model.base_name}", show_header=False)
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="white")
            
            info_table.add_row("Base Name", model.base_name)
            info_table.add_row("Context Length", f"{model.context:,} tokens")
            info_table.add_row("Tools Support", "Yes" if model.tools else "No")
            info_table.add_row("Languages", ", ".join(model.languages))
            info_table.add_row("Docker", model.docker)
            if model.features:
                info_table.add_row("Features", ", ".join(model.features))
            
            # Variants table
            variants_table = Table(title="Available Variants")
            variants_table.add_column("Variant", style="cyan")
            variants_table.add_column("Parameters", style="yellow")
            variants_table.add_column("Quantization", style="green")
            variants_table.add_column("VRAM", style="red", justify="right")
            variants_table.add_column("RAM", style="blue", justify="right")
            
            for variant_id, variant in model.variants.items():
                variants_table.add_row(
                    variant_id,
                    variant.params,
                    variant.quant,
                    f"{variant.vram:.2f} GB",
                    f"{variant.ram:.2f} GB"
                )
            
            # Scores table
            scores_table = Table(title="Capability Scores")
            scores_table.add_column("Capability", style="cyan")
            scores_table.add_column("Score", style="yellow", justify="center")
            scores_table.add_column("Stars", style="green", justify="center")
            
            for capability, score in model.scores.items():
                stars = "⭐" * score
                scores_table.add_row(capability.title(), str(score), stars)
            
            console.print(Panel(info_table, border_style="blue"))
            console.print(Panel(variants_table, border_style="green"))
            console.print(Panel(scores_table, border_style="yellow"))
            
        else:
            # List all models
            from rich.table import Table
            
            if model_type == 'all':
                models_dict = manager.models
            else:
                models_dict = manager.get_models_by_type(model_type)
            
            table = Table(title=f"Available Models ({len(models_dict)} total)")
            table.add_column("Model ID", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Type", style="yellow")
            table.add_column("Variants", style="green", justify="center")
            table.add_column("Context", style="blue", justify="right")
            table.add_column("Tools", style="magenta", justify="center")
            
            for model_id, model in models_dict.items():
                table.add_row(
                    model_id,
                    model.base_name,
                    model.model_type,
                    str(len(model.variants)),
                    f"{model.context:,}",
                    "✅" if model.tools else "❌"
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]❌ Error loading models: {e}[/red]")
        if ctx.obj['verbose']:
            logger.exception("Detailed error information")
        sys.exit(1)


@cli.command()
@click.option('--use-case', type=click.Choice(['rag', 'code', 'chat', 'analysis', 'multilingual', 'education', 'creative', 'general']))
@click.option('--context-importance', type=click.IntRange(1, 4))
@click.option('--languages', multiple=True, type=click.Choice(['english', 'spanish', 'european', 'many', 'asian']))
@click.option('--tools/--no-tools', default=None)
@click.option('--top-n', default=5, help='Number of recommendations to show')
@click.pass_context
def recommend(ctx, use_case: Optional[str], context_importance: Optional[int], 
              languages: tuple, tools: Optional[bool], top_n: int):
    """Get model recommendations based on requirements"""
    console = Console()
    
    async def get_recommendations():
        try:
            # Detect hardware
            detector = HardwareDetector()
            system_info = await detector.detect_system_info()
            available_memory = detector.get_available_memory()
            
            # Get model recommendations
            manager = ModelManager(ctx.obj['models_file'])
            
            # Use provided parameters or defaults
            use_case_final = use_case or 'general'
            context_importance_final = context_importance or 2
            languages_final = list(languages) if languages else ['english']
            tools_final = tools if tools is not None else False
            
            recommendations = manager.recommend_models(
                available_memory, use_case_final, context_importance_final, 
                languages_final, tools_final, top_n
            )
            
            # Display results
            selector = ModelSelector(ctx.obj['models_file'])
            selector.system_info = system_info
            selector.available_memory = available_memory
            selector.display_recommendations(recommendations, use_case_final)
            
        except Exception as e:
            console.print(f"[red]❌ Error getting recommendations: {e}[/red]")
            if ctx.obj['verbose']:
                logger.exception("Detailed error information")
            sys.exit(1)
    
    try:
        asyncio.run(get_recommendations())
    except KeyboardInterrupt:
        console.print("\\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)


@cli.command()
@click.pass_context
def clear_cache(ctx):
    """Clear hardware detection cache"""
    console = Console()
    
    try:
        cache_file = Path.home() / ".model-selector" / "hardware_cache.json"
        if cache_file.exists():
            cache_file.unlink()
            console.print("[green]✅ Hardware cache cleared[/green]")
        else:
            console.print("[yellow]⚠️ No cache file found[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error clearing cache: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def version(ctx):
    """Show version information"""
    console = Console()
    
    try:
        from model_selector import __version__
        console.print(f"[cyan]AI Model Selection Assistant v{__version__}[/cyan]")
        console.print(f"[dim]Python {sys.version.split()[0]}[/dim]")
    except ImportError:
        console.print("[red]❌ Version information not available[/red]")


if __name__ == '__main__':
    cli()