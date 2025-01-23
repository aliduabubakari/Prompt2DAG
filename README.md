# Automated Apache Airflow DAG Generation Pipeline

This project implements an automated pipeline for generating Apache Airflow DAGs using Large Language Models (LLMs). It supports multiple LLM providers (DeepInfra, OpenAI, and Claude) and includes comprehensive evaluation capabilities.

## Features

- Multi-LLM support (DeepInfra, OpenAI, Claude)
- Step-by-step DAG generation pipeline
- Modular DAG generation approach
- Comprehensive code evaluation system
- Configurable settings via JSON
- Docker-based DAG support
- Detailed logging and error handling

## Project Structure

```plaintext
project/
├── config.json
├── outputs/
├── evaluations/
│   ├── __init__.py
│   ├── eval_1_code_quality.py
│   ├── eval_2_static_analysis.py
│   └── eval_3_dag_structure.py
├── step_1.py
├── step_2.py
├── step_3.py
└── modular_dag_generator.py
```

## Configuration

The project uses a comprehensive configuration file (`config.json`) that controls all aspects of the pipeline:

```json
{
    "model_settings": {
        "active_provider": "deepinfra",
        "deepinfra": {
            "model_name": "Qwen/Qwen2.5-72B-Instruct",
            "api_key": "B4LYt",
            "max_tokens": 2000,
            "base_url": "https://api.deepinfra.com/v1/openai"
        },
        "openai": {
            "model_name": "gpt-4",
            "api_key": "sk-proj-",
            "max_tokens": 2000,
            "temperature": 0,
            "base_url": "https://api.openai.com/v1"
        },
        "claude": {
            "model_name": "claude-3-5-sonnet-20241022",
            "api_key": "sk-ant-",
            "max_tokens": 4000,
            "temperature": 0
        }
    },
    "modular_generation": {
        "enabled": true,
        "components": {
            "dag_config": {
                "max_tokens": 1000,
                "temperature": 0
            },
            "tasks": {
                "max_tokens": 1500,
                "temperature": 0
            },
            "dependencies": {
                "max_tokens": 500,
                "temperature": 0
            },
            "assembly": {
                "max_tokens": 2000,
                "temperature": 0
            }
        },
        "docker_based": true,
        "output_format": "python"
    }, 
    "pipeline_settings": {
        "input_file": "pipeline_description.txt",
        "is_docker_based": true,
        "output_directory": "outputs"
    },
    "evaluations": {
        "llm_settings": {
            "provider": "openai",
            "openai": {
                "model_name": "gpt-4",
                "api_key": "sk-proj-",
                "max_tokens": 150,
                "temperature": 0
            }
        },
        "code_quality": {
            "enabled": true,
            "weights": {
                "syntax_and_validity": 0.15,
                "code_readability": 0.20,
                "modularity_and_reusability": 0.20,
                "docker_configuration": 0.25,
                "functional_completeness": 0.20
            },
            "iterations": 3,
            "seed": 123,
            "output_format": "json"
        }, 
        "static_analysis": {
            "enabled": true,
            "weights": {
                "code_quality": 0.3,
                "security": 0.3,
                "complexity": 0.2,
                "flake8": 0.2
            },
            "thresholds": {
                "complexity_max": 10,
                "maintainability_min": 65,
                "security_issues_max": 0
            },
            "output_format": "json"
        }, 
        "dag_structure": {
            "enabled": true,
            "weights": {
                "structure": 0.6,
                "docker_config": 0.4
            },
            "thresholds": {
                "min_tasks": 1,
                "min_dependencies": 1,
                "required_docker_configs": [
                    "network_mode",
                    "environment",
                    "image"
                ]
            },
            "output_format": "json"
        }
    },
    "template_settings": {
        "template_format": "yaml",
        "include_docker": true,
        "default_resource_limits": {
            "cpu": 1,
            "memory": "2G"
        },
        "default_retry_policy": {
            "retries": 3,
            "delay": "5 minutes"
        }
    },
    "dag_settings": {
        "docker_based": true,
        "include_tests": true,
        "output_format": "python",
        "code_style": "pep8",
        "include_documentation": true
    },
    "logging": {
        "level": "INFO",
        "log_file": "pipeline_generation.log"
    }
}
```

## Configuration File (config.json) Structure

### Model Settings

#### `model_settings`

```markdown
- **active_provider**: Specifies the current model provider
  - Example: "deepinfra"

- **deepinfra**:
  - `model_name`: Specific Deepinfra model
  - `api_key`: Authentication key
  - `max_tokens`: Maximum token processing limit
  - `base_url`: API endpoint

- **openai**:
  - Similar configuration for OpenAI models
  - Includes `model_name`, `api_key`, `max_tokens`

- **claude**:
  - Configuration for Claude models
  - Includes `model_name`, `api_key`, `max_tokens`
```

### Modular Generation

#### `modular_generation`

```markdown
- **enabled**: Boolean to enable/disable modular generation
- **components**: 
  - `dag_config`: Configuration settings
  - `tasks`: Task-specific configurations
  - `dependencies`: Dependency management
  - `assembly`: Pipeline assembly settings
- **docker_based**: Docker deployment flag
- **output_format**: Generated code language (e.g., "python")
```

### Pipeline Settings

#### `pipeline_settings`

```markdown
- **input_file**: Source description file path
- **is_docker_based**: Docker deployment indicator
- **output_directory**: Generated files storage location
```

### Evaluations

#### `evaluations`

```markdown
- **llm_settings**: Language model evaluation configuration
- **code_quality**:
  - Weights for evaluation categories
  - Iteration count
  - Random seed
  - Output format

- **static_analysis**:
  - Complexity assessment
  - Security issue detection
  - Code quality metrics
  - Weights and thresholds

- **dag_structure**:
  - Minimum task requirements
  - Dependency validation
  - Docker configuration checks
```

### Template Settings

#### `template_settings`

```markdown
- **template_format**: Output template type (e.g., "yaml")
- **include_docker**: Docker configuration inclusion
- **default_resource_limits**:
  - CPU allocation
  - Memory constraints
- **default_retry_policy**:
  - Maximum retry attempts
  - Retry delay
```

### DAG Settings

#### `dag_settings`

```markdown
- **docker_based**: Docker-specific DAG configuration
- **include_tests**: Test integration flag
- **output_format**: Code generation language
- **code_style**: Coding standard (e.g., "pep8")
- **include_documentation**: Inline documentation
```

### Logging

#### `logging`

```markdown
- **level**: Logging verbosity (e.g., "INFO")
- **log_file**: Output log filename
```

## Pipeline Steps

### 1. Pipeline Analysis (step_1.py)
Analyzes the pipeline description and generates a detailed analysis:

```bash
python step_1.py --config config.json --provider [deepinfra|openai|claude]
```

Output: `outputs/pipeline_analysis_[timestamp].txt`

### 2. YAML Template Generation (step_2.py)
Converts the analysis into a structured YAML template:

```bash
python step_2.py --config config.json --analysis outputs/pipeline_analysis_[timestamp].txt --provider [deepinfra|openai|claude]
```

Output: `outputs/generated_yaml_[timestamp].yaml`

### 3. DAG Code Generation (step_3.py)
Generates the final Airflow DAG code:

```bash
python step_3.py --config config.json --yaml outputs/generated_yaml_[timestamp].yaml --provider [deepinfra|openai|claude]
```

Output: `outputs/generated_dag_[timestamp].py`

## Evaluations

The project includes three types of evaluations:

### 1. Code Quality Evaluation
```bash
python evaluations/eval_1_code_quality.py --config config.json --dag outputs/generated_dag_[timestamp].py
```

Evaluates:
- Syntax and validity (15%)
- Code readability (20%)
- Modularity and reusability (20%)
- Docker configuration (25%)
- Functional completeness (20%)

### 2. Static Analysis
```bash
python evaluations/eval_2_static_analysis.py --config config.json --dag outputs/generated_dag_[timestamp].py
```

Analyzes:
- Code quality (30%)
- Security (30%)
- Complexity (20%)
- Flake8 compliance (20%)

### 3. DAG Structure Analysis
```bash
python evaluations/eval_3_dag_structure.py --config config.json --dag outputs/generated_dag_[timestamp].py
```

Evaluates:
- DAG structure (60%)
- Docker configuration (40%)

## Modular Approach

The project also supports a modular approach to DAG generation using `modular_dag_generator.py`:

```bash
python modular_dag_generator.py --config config.json --yaml outputs/generated_yaml_[timestamp].yaml --provider [deepinfra|openai|claude]
```

The modular approach:
1. Generates DAG configuration
2. Creates task definitions
3. Establishes dependencies
4. Assembles the final DAG

## Requirements

```txt
openai
anthropic
python-dateutil
pyyaml
networkx
radon
pylint
bandit
flake8
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/airflow-dag-generator.git
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Configure your API keys in `config.json`

## Usage

1. Create a pipeline description file
2. Update configuration in `config.json`
3. Run the pipeline steps sequentially or use the modular generator
4. Review the generated DAG and evaluation results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI GPT-4
- Anthropic Claude
- Meta Llama
- Apache Airflow community