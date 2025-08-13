#!/usr/bin/env python3
"""
AI Model Token Calculator
A CLI tool to calculate token counts for various AI models from files.
"""

import sys
import os
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Tokenizer imports
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    rprint("[yellow]Warning: tiktoken not available. OpenAI model support disabled.[/yellow]")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    rprint("[yellow]Warning: transformers not available. Hugging Face model support disabled.[/yellow]")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    rprint("[yellow]Warning: anthropic not available. Claude model support disabled.[/yellow]")

# Disable huggingface_hub authentication requirements for offline tokenization
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Suppress authentication and API warnings
import warnings
warnings.filterwarnings("ignore", message=".*authentication.*")
warnings.filterwarnings("ignore", message=".*API key.*")
warnings.filterwarnings("ignore", message=".*auth_token.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

console = Console()

# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4": {"type": "openai", "encoding": "cl100k_base"},
    "gpt-4-turbo": {"type": "openai", "encoding": "cl100k_base"},
    "gpt-4o": {"type": "openai", "encoding": "o200k_base"},
    "gpt-4o-mini": {"type": "openai", "encoding": "o200k_base"},
    "gpt-3.5-turbo": {"type": "openai", "encoding": "cl100k_base"},
    
    # Claude models
    "claude-3-opus": {"type": "anthropic", "model_name": "claude-3-opus-20240229"},
    "claude-3-sonnet": {"type": "anthropic", "model_name": "claude-3-sonnet-20240229"},
    "claude-3-haiku": {"type": "anthropic", "model_name": "claude-3-haiku-20240307"},
    "claude-3.5-sonnet": {"type": "anthropic", "model_name": "claude-3-5-sonnet-20240620"},
    # Aliases
    "claude-3": {"type": "anthropic", "model_name": "claude-3-sonnet-20240229"},
    "sonnet-4": {"type": "anthropic", "model_name": "claude-3-5-sonnet-20240620"},
    "claude-sonnet-4": {"type": "anthropic", "model_name": "claude-3-5-sonnet-20240620"},
    "opus-4.1": {"type": "anthropic", "model_name": "claude-3-opus-20240229"},
    "claude-opus-4.1": {"type": "anthropic", "model_name": "claude-3-opus-20240229"},
    
    # Hugging Face models (using publicly accessible models)
    "bert": {"type": "hf", "model_name": "bert-base-uncased"},
    "distilbert": {"type": "hf", "model_name": "distilbert-base-uncased"},
    "t5": {"type": "hf", "model_name": "google-t5/t5-small"},
    "gpt2": {"type": "hf", "model_name": "openai-community/gpt2"},
    
    # Mistral-compatible models (open-source alternatives with longer context)
    "mistral-small3.2": {"type": "openai", "encoding": "cl100k_base"},
    "magistral": {"type": "openai", "encoding": "cl100k_base"},
    
    # Additional open-source models (publicly accessible)
    "phi-3": {"type": "hf", "model_name": "microsoft/Phi-3-mini-4k-instruct"},
    "falcon-7b": {"type": "hf", "model_name": "tiiuae/falcon-7b"},
}

class TokenCalculator:
    def __init__(self):
        self.tokenizers = {}
        self.anthropic_client = None
        self.shown_errors = set()  # Track shown error messages
        self.anthropic_fallback = False  # Flag for fallback mode
        
    def get_openai_tokenizer(self, encoding_name: str):
        """Get OpenAI tokenizer for the specified encoding."""
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken not available. Install with: uv add tiktoken")
        
        if encoding_name not in self.tokenizers:
            self.tokenizers[encoding_name] = tiktoken.get_encoding(encoding_name)
        return self.tokenizers[encoding_name]

    def get_anthropic_client(self):
        """Get Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic not available. Install with: uv add anthropic")
        
        if self.anthropic_client is None:
            # Try to get API key from environment variable
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.\n"
                    "You can get your API key from: https://console.anthropic.com/settings/keys\n"
                    "Set it with: export ANTHROPIC_API_KEY='your-api-key-here'"
                )
            
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self.anthropic_client

    def get_hf_tokenizer(self, model_name: str):
        """Get Hugging Face tokenizer for the specified model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available. Install with: uv add transformers torch")
        
        if model_name not in self.tokenizers:
            try:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer for {model_name}: {e}")
        return self.tokenizers[model_name]
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens for the given text using the specified model."""
        model_name_lower = model_name.lower()
        
        # Find matching model config
        config = None
        # Prioritize exact match
        if model_name_lower in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_name_lower]
        else: # Fallback to partial match
            for key, cfg in MODEL_CONFIGS.items():
                if key in model_name_lower:
                    config = cfg
                    break
        
        if not config:
            # Default to GPT-4 encoding for unknown models
            console.print(f"[yellow]Warning: Unknown model '{model_name}', using GPT-4 encoding as fallback[/yellow]")
            config = {"type": "openai", "encoding": "cl100k_base"}
        
        if config["type"] == "openai":
            tokenizer = self.get_openai_tokenizer(config["encoding"])
            return len(tokenizer.encode(text))
        elif config["type"] == "anthropic":
            # Check if we should fallback due to previous errors
            if self.anthropic_fallback:
                return self._fallback_tokenizer(text, model_name)
            
            try:
                client = self.get_anthropic_client()
                response = client.messages.count_tokens(
                    model=config["model_name"],
                    messages=[{"role": "user", "content": text}]
                )
                return response.input_tokens
            except Exception as e:
                return self._handle_anthropic_error(e, text, model_name)
        elif config["type"] == "hf":
            tokenizer = self.get_hf_tokenizer(config["model_name"])
            # Suppress warnings about sequence length and just count tokens
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                # Use add_special_tokens=False and truncation=False to just count tokens
                tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
                return len(tokens)
        else:
            raise ValueError(f"Unknown tokenizer type: {config['type']}")
    
    def _handle_anthropic_error(self, error: Exception, text: str, model_name: str) -> int:
        """Handle Anthropic API errors with smart fallback and consolidated error messages."""
        error_msg = str(error)
        
        # Extract meaningful error messages
        if "credit balance is too low" in error_msg:
            error_key = "low_credit"
            if error_key not in self.shown_errors:
                console.print("\n[red]💳 Anthropic API Credit Issue[/red]")
                console.print("[yellow]Your Anthropic API credit balance is too low.[/yellow]")
                console.print("📍 Visit https://console.anthropic.com/settings/billing to add credits")
                console.print("[dim]→ Falling back to OpenAI tokenizer approximation for Claude models[/dim]\n")
                self.shown_errors.add(error_key)
            self.anthropic_fallback = True
            return self._fallback_tokenizer(text, model_name)
            
        elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            error_key = "auth_error"
            if error_key not in self.shown_errors:
                console.print("\n[red]🔑 Anthropic API Authentication Error[/red]")
                console.print("[yellow]Please check your ANTHROPIC_API_KEY environment variable.[/yellow]")
                console.print("[dim]→ Falling back to OpenAI tokenizer approximation for Claude models[/dim]\n")
                self.shown_errors.add(error_key)
            self.anthropic_fallback = True
            return self._fallback_tokenizer(text, model_name)
            
        elif "rate limit" in error_msg.lower():
            error_key = "rate_limit"
            if error_key not in self.shown_errors:
                console.print("\n[red]⏰ Anthropic API Rate Limit[/red]")
                console.print("[yellow]Rate limit exceeded. Please wait a moment before retrying.[/yellow]")
                console.print("[dim]→ Falling back to OpenAI tokenizer approximation for Claude models[/dim]\n")
                self.shown_errors.add(error_key)
            self.anthropic_fallback = True
            return self._fallback_tokenizer(text, model_name)
            
        else:
            # Generic error handling
            error_key = "generic_error"
            if error_key not in self.shown_errors:
                console.print(f"\n[red]❌ Anthropic API Error[/red]")
                console.print(f"[yellow]{error_msg}[/yellow]")
                console.print("[dim]→ Falling back to OpenAI tokenizer approximation for Claude models[/dim]\n")
                self.shown_errors.add(error_key)
            self.anthropic_fallback = True
            return self._fallback_tokenizer(text, model_name)
    
    def _fallback_tokenizer(self, text: str, model_name: str) -> int:
        """Fallback to OpenAI tokenizer for Claude models when API is unavailable."""
        try:
            tokenizer = self.get_openai_tokenizer("cl100k_base")
            return len(tokenizer.encode(text))
        except Exception:
            # Last resort - rough estimation
            return len(text.split()) * 1.3  # Rough token estimation
    
    def read_file(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            console.print(f"[red]Error reading {file_path}: {e}[/red]")
            return ""
    
    def process_files(self, model_name: str, file_patterns: List[str]) -> Dict[str, int]:
        """Process multiple files and return token counts."""
        results = {}
        
        # Expand glob patterns
        all_files = []
        for pattern in file_patterns:
            if "*" in pattern or "?" in pattern:
                all_files.extend(glob.glob(pattern))
            else:
                all_files.append(pattern)
        
        # Remove duplicates and filter existing files
        all_files = list(set(all_files))
        existing_files = [f for f in all_files if os.path.exists(f)]
        
        if not existing_files:
            console.print("[red]No valid files found![/red]")
            return {}
        
        for file_path in existing_files:
            content = self.read_file(file_path)
            if content:
                token_count = self.count_tokens(content, model_name)
                results[file_path] = token_count
        
        return results

def create_table(results: Dict[str, int], model_name: str, calculator: TokenCalculator = None) -> Table:
    """Create a rich table for displaying results."""
    title = f"Token Counts for Model: {model_name}"
    if calculator and calculator.anthropic_fallback:
        # Check if this is an anthropic model
        model_lower = model_name.lower()
        anthropic_models = ['sonnet-4', 'claude-sonnet-4', 'opus-4.1', 'claude-opus-4.1', 
                          'claude-3', 'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'claude-3.5-sonnet']
        if any(am in model_lower for am in anthropic_models):
            title += "\n[dim italic]→ Using OpenAI tokenizer approximation[/dim italic]"
    
    table = Table(title=title)
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Tokens", style="magenta", justify="right")
    
    total_tokens = 0
    for file_path, tokens in results.items():
        table.add_row(file_path, str(tokens))
        total_tokens += tokens
    
    if len(results) > 1:
        table.add_row("TOTAL", str(total_tokens), style="bold green")
    
    return table

def main():
    parser = argparse.ArgumentParser(
        description="Calculate token counts for various AI models from files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python token_calculator.py gpt-4 file.txt
  python token_calculator.py claude-3 file1.txt file2.md
  python token_calculator.py llama-2 *.py --format json
  
Supported models:
  OpenAI: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo
  Claude: claude-3, claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet, sonnet-4, opus-4.1
  Open Source: bert, distilbert, t5, gpt2, mistral-small3.2, magistral, phi-3, falcon-7b
        """
    )
    
    parser.add_argument("model", nargs='?', help="AI model name (e.g., gpt-4, claude-3, llama-2)")
    parser.add_argument("files", nargs="*", help="File paths or patterns to process")
    parser.add_argument("--format", choices=["table", "json"], default="table", 
                       help="Output format (default: table)")
    parser.add_argument("--list-models", action="store_true", 
                       help="List all supported models")
    
    args = parser.parse_args()
    
    if args.list_models:
        console.print("\n[bold]Supported Models:[/bold]")
        
        # OpenAI models (excluding Mistral)
        openai_models = [m for m, c in MODEL_CONFIGS.items() 
                        if c["type"] == "openai" and not m.startswith(("mistral", "magistral"))]
        console.print(f"  [cyan]OpenAI:[/cyan] {', '.join(openai_models)}")
        
        # Claude models
        claude_models = [m for m, c in MODEL_CONFIGS.items() if c["type"] == "anthropic"]
        console.print(f"  [cyan]Claude:[/cyan] {', '.join(claude_models)}")
        
        # Mistral-compatible models
        mistral_models = [m for m in MODEL_CONFIGS.keys() if m.startswith(("mistral", "magistral"))]
        console.print(f"  [cyan]Mistral-compatible:[/cyan] {', '.join(mistral_models)}")
        
        # Other Open Source models
        other_models = [m for m, c in MODEL_CONFIGS.items() 
                       if c["type"] == "hf" and not m.startswith(("mistral", "magistral"))]
        console.print(f"  [cyan]Other Open Source:[/cyan] {', '.join(other_models)}")
        return
    
    # Validate required arguments when not listing models
    if not args.model:
        parser.error("model argument is required when not using --list-models")
    if not args.files:
        parser.error("files argument is required when not using --list-models")
    
    calculator = TokenCalculator()
    
    try:
        results = calculator.process_files(args.model, args.files)
        
        if not results:
            sys.exit(1)
        
        if args.format == "json":
            output = {
                "model": args.model,
                "results": results,
                "total_tokens": sum(results.values())
            }
            print(json.dumps(output, indent=2))
        else:
            table = create_table(results, args.model, calculator)
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()