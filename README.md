# AI Model Token Calculator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-compatible-green.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fast, accurate command-line tool for calculating token counts across various AI models. Supports OpenAI GPT models, Claude (Anthropic), and popular open-source models with intelligent fallback handling.

## ✨ Features

- 🚀 **Multi-Model Support**: OpenAI GPT, Claude, BERT, DistilBERT, T5, GPT-2
- 📁 **Batch Processing**: Handle multiple files and glob patterns
- 🎨 **Beautiful Output**: Rich table formatting and JSON export
- 🔄 **Smart Fallback**: Graceful handling when API keys are unavailable
- ⚡ **Fast & Accurate**: Uses official tokenizers and APIs when available
- 🛡️ **Error Resilient**: Clear error messages with helpful guidance

## 📦 Installation

Simply clone the repository and sync dependencies with uv:

```bash
git clone <repository-url>
cd ai-model-token-calculator
uv sync
```

That's it! All dependencies are automatically managed through the `pyproject.toml` file.

### Prerequisites

- **Python 3.8+**
- **uv** package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

## 🔧 Configuration

### Anthropic API (Optional - for accurate Claude token counts)

For precise Claude model token counting, set up your Anthropic API key:

1. **Get API Key**: Visit [Anthropic Console](https://console.anthropic.com/settings/keys)
2. **Set Environment Variable**:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```
3. **Persistent Setup** (optional):
   ```bash
   echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

> **Note**: Without an API key, Claude models automatically fall back to OpenAI tokenizer approximation with clear visual indicators.

## 🚀 Usage

### Quick Start

```bash
# Count tokens in a single file
uv run python token_calculator.py gpt-4 document.txt

# Process multiple files
uv run python token_calculator.py claude-3 *.py *.md

# JSON output for automation
uv run python token_calculator.py sonnet-4 data.txt --format json

# List all supported models
uv run python token_calculator.py --list-models
```

### Command Reference

```bash
uv run python token_calculator.py [model] [files...] [options]

Arguments:
  model                 AI model name (see supported models below)
  files                 File paths or glob patterns to process

Options:
  --format table|json   Output format (default: table)
  --list-models        Show all supported models
  -h, --help           Show help message
```

### Supported Models

| Category | Models | Tokenization Method |
|----------|---------|-------------------|
| **OpenAI** | `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` | tiktoken (official) |
| **Claude** | `claude-3`, `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, `claude-3.5-sonnet`, `sonnet-4`, `opus-4.1` | Anthropic API (with fallback) |
| **Open Source** | `bert`, `distilbert`, `t5`, `gpt2`, `phi-3`, `falcon-7b` | HuggingFace transformers |
| **Mistral-Compatible** | `mistral-small3.2`, `magistral` | tiktoken (cl100k_base encoding) |

## 💡 Examples

### Basic File Processing

```bash
# Single file analysis
uv run python token_calculator.py gpt-4 README.md
# Output: Tokens: 1,247

# Multiple specific files
uv run python token_calculator.py claude-3 app.py utils.py config.json
# Output: Table with per-file and total counts
```

### Advanced Pattern Matching

```bash
# All Python files in current directory
uv run python token_calculator.py bert *.py

# All markdown files recursively (requires shell expansion)
uv run python token_calculator.py gpt-4 **/*.md

# Mixed file types
uv run python token_calculator.py sonnet-4 *.{py,js,md,txt}
```

### Automation & Integration

```bash
# JSON output for scripts
uv run python token_calculator.py gpt-4 input.txt --format json
# Output: {"model": "gpt-4", "results": {"input.txt": 156}, "total_tokens": 156}

# Model comparison (example script usage)
for model in gpt-4 claude-3 bert; do
  echo "$model: $(uv run python token_calculator.py $model file.txt --format json | jq .total_tokens)"
done
```

### Real-World Use Cases

```bash
# Estimate API costs before processing
uv run python token_calculator.py gpt-4 large-dataset.txt
# → Plan API usage based on token count

# Compare model tokenization differences  
uv run python token_calculator.py gpt-4 prompt.txt
uv run python token_calculator.py claude-3 prompt.txt
# → Choose optimal model for your use case

# Validate input size limits
uv run python token_calculator.py gpt-4 context.txt
# → Ensure content fits within model limits
```

## 🔍 How It Works

The tool employs different tokenization strategies optimized for each model family:

| Model Type | Tokenization Method | Accuracy |
|------------|-------------------|----------|
| **OpenAI Models** | `tiktoken` library with model-specific encodings | 💯 Exact |
| **Claude Models** | Anthropic API `count_tokens` endpoint | 💯 Exact |
| **Claude (Fallback)** | `tiktoken` cl100k_base approximation | 🎯 ~95% |
| **Open Source** | Model-specific `AutoTokenizer` from HuggingFace | 💯 Exact |

### Error Handling & Fallback

The tool includes intelligent error handling:

- **API Issues**: Automatically falls back to approximation methods
- **Rate Limits**: Clear guidance with retry suggestions  
- **Authentication**: Step-by-step setup instructions
- **File Errors**: Graceful handling of missing/unreadable files

## 🛠️ Development

### Project Structure

```
ai-model-token-calculator/
├── token_calculator.py    # Main application
├── pyproject.toml        # Dependencies & config
├── README.md            # This file
└── uv.lock             # Dependency lock file
```

### Contributing

1. **Setup Development Environment**:
   ```bash
   git clone <repository-url>
   cd ai-model-token-calculator
   uv sync
   ```

2. **Run Tests**:
   ```bash
   uv run python token_calculator.py --list-models
   uv run python token_calculator.py gpt-4 README.md
   ```

3. **Code Style**: Follow existing patterns and use descriptive variable names

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ANTHROPIC_API_KEY not found` | Set environment variable or use fallback mode |
| `tiktoken not available` | Run `uv sync` to install dependencies |
| `No valid files found` | Check file paths and permissions |
| `Rate limit exceeded` | Wait and retry, or use different model |

### API Costs

| Operation | Cost |
|-----------|------|
| **Token Counting** | Free for all models |
| **Claude API Calls** | Only for actual content generation (not token counting) |

### Performance Tips

- Use JSON format for automation scripts
- Batch multiple files in single command for efficiency  
- Consider model-specific token limits when processing large files
- Claude fallback mode is ~10x faster than API calls

## 📄 License

MIT License - feel free to use this tool in your projects!

---

**Need help?** Open an issue or check the troubleshooting section above.