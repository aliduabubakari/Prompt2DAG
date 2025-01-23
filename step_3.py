import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import os
from typing import Dict, Optional, Any
import anthropic
from openai import OpenAI

class LLMProvider:
    """Handle different LLM provider interactions."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.provider = config['model_settings']['active_provider']
        self.provider_config = config['model_settings'][self.provider]
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """Initialize the appropriate LLM client."""
        if self.provider == 'deepinfra':
            return OpenAI(
                api_key=self.provider_config['api_key'],
                base_url=self.provider_config['base_url']
            )
        elif self.provider == 'openai':
            return OpenAI(api_key=self.provider_config['api_key'])
        elif self.provider == 'claude':
            return anthropic.Anthropic(api_key=self.provider_config['api_key'])
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_completion(self, system_prompt: str, user_input: str) -> str:
        """Generate completion based on provider."""
        try:
            if self.provider == 'deepinfra':
                response = self.client.chat.completions.create(
                    model=self.provider_config['model_name'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=self.provider_config['max_tokens'],
                    temperature=self.provider_config.get('temperature', 0),
                    stream=False
                )
                return response.choices[0].message.content

            elif self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.provider_config['model_name'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=self.provider_config.get('temperature', 0),
                    max_tokens=self.provider_config['max_tokens']
                )
                return response.choices[0].message.content

            elif self.provider == 'claude':
                response = self.client.messages.create(
                    model=self.provider_config['model_name'],
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=self.provider_config['max_tokens'],
                    temperature=self.provider_config.get('temperature', 0)
                )
                return response.content[0].text

        except Exception as e:
            logging.error(f"Error generating completion with {self.provider}: {e}")
            raise


class DAGCodeGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.llm_provider = LLMProvider(config)
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

    def _get_system_prompt(self) -> str:
        return """You are an expert Apache Airflow developer specializing in creating DAGs for containerized data pipelines. 
        Your task is to convert a YAML configuration template into a production-ready, executable Airflow DAG Python file.
        The generated code must be complete, properly formatted, and follow all Airflow best practices."""

    def _get_user_prompt(self, yaml_template: str) -> str:
        prompt = f"""Convert this YAML template into an executable Airflow DAG:

        {yaml_template}

        Requirements for the generated DAG:

        1. Code Structure:
           - Include all necessary imports
           - Use appropriate Airflow operators
           - Implement proper error handling and retries
           - Follow PEP 8 style guidelines
           - Include comprehensive docstrings and comments

        2. Essential Components:
           - DAG configuration with all parameters from YAML
           - Task dependencies as specified in YAML
           - Error handling and monitoring setup
           - Handle resource constraints (CPU, memory)
           - Configure proper retry policies
           - Set up error notifications
           - Use Airflow variables for sensitive data"""

        if self.config['dag_settings']['docker_based']:
            prompt += """
        3. Docker-Specific Requirements:
           - Use `DockerOperator` for containerized tasks
           - Include proper Docker configurations
           - Set up volume mounts and network modes
           - Configure environment variables
           - Implement container cleanup"""

        return prompt

    def generate_dag_code(self, yaml_template: str) -> Optional[str]:
        """Generate DAG code from YAML template."""
        try:
            # Generate code using configured LLM provider
            response = self.llm_provider.generate_completion(
                system_prompt=self._get_system_prompt(),
                user_input=self._get_user_prompt(yaml_template)
            )

            # Extract code from response
            return self.extract_code(response)
        except Exception as e:
            logging.error(f"Error generating DAG code: {e}")
            return None

    def extract_code(self, content: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in content:
            parts = content.split("```python")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        elif "```" in content:
            # Some LLMs might not specify the language
            parts = content.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        
        # If no code blocks found, return the content as is
        return content.strip()

    def save_dag_code(self, code: str, yaml_file: str) -> str:
        """Save DAG code to file using timestamp from YAML filename."""
        try:
            # Extract timestamp from YAML filename
            yaml_filename = os.path.basename(yaml_file)
            timestamp = yaml_filename.split('_', 1)[1].rsplit('.', 1)[0]
            
            # Create output filename
            output_dir = Path(self.config['pipeline_settings']['output_directory'])
            output_file = output_dir / f"generated_dag_{timestamp}.py"
            
            # Ensure output directory exists
            output_dir.mkdir(exist_ok=True)
            
            # Save the content
            with open(output_file, 'w') as f:
                f.write(code)
            logging.info(f"DAG code saved to: {output_file}")
            return str(output_file)
        except Exception as e:
            logging.error(f"Error saving DAG code: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Generate Airflow DAG code from YAML template')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    parser.add_argument('--yaml', required=True, help='Path to YAML template file')
    parser.add_argument('--provider', help='Override the LLM provider (deepinfra, openai, or claude)')
    args = parser.parse_args()

    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Override provider if specified
        if args.provider:
            if args.provider not in ['deepinfra', 'openai', 'claude']:
                raise ValueError(f"Unsupported provider: {args.provider}")
            config['model_settings']['active_provider'] = args.provider

        # Initialize generator
        generator = DAGCodeGenerator(config)
        
        # Load YAML template
        with open(args.yaml, 'r') as f:
            yaml_template = f.read()
        
        # Generate DAG code
        dag_code = generator.generate_dag_code(yaml_template)
        if dag_code:
            # Save DAG code
            output_file = generator.save_dag_code(dag_code, args.yaml)
            logging.info(f"DAG generation completed successfully using {config['model_settings']['active_provider']}: {output_file}")
        else:
            logging.error("Failed to generate DAG code")
            
    except Exception as e:
        logging.error(f"DAG generation failed: {e}")
        raise

if __name__ == "__main__":
    main()