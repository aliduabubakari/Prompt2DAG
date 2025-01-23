import json
import logging
from pathlib import Path
from datetime import datetime
import yaml
import argparse
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

class YAMLTemplateGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.llm_provider = LLMProvider(config)
        self.template_settings = config['template_settings']
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
        return """You are an expert Apache Airflow DAG designer. 
        Your task is to analyze the provided pipeline description and extract structured information 
        that will be used to create a YAML template. The template should be production-ready and 
        follow Airflow best practices."""

    def _get_template_prompt(self, analysis: str) -> str:
       return f"""Based on this analysis:

       {analysis}

       Create a JSON object with the following structure:
       {{
           "dag_config": {{
               "dag_id": "example_dag",
               "schedule_interval": "@daily",
               "default_args": {{
                   "owner": "airflow",
                   "start_date": "2023-01-01",
                   "retries": 3,
                   "retry_delay": "5 minutes"
               }}
           }},
           "tasks": [
               {{
                   "task_id": "task-name",
                   "image": "image-name",
                   "command": "command to run",
                   "environment": {{"key": "${{VARIABLE_NAME}"}},
                   "resources": {{"cpu": 1, "memory": "2G"}},
                   "depends_on": ["dependent-task"]
               }}
           ],
           "volumes": [
               {{
                   "name": "volume-name",
                   "mount_path": "/path",
                   "read_only": false
               }}
           ],
           "docker_config": {{
               "image_prefix": "prefix",
               "network_mode": "bridge"
           }},
           "error_handling": {{
               "email_on_failure": true,
               "execution_timeout": 30,
               "retry_policy": {{
                   "max_retries": 3,
                   "delay": "5 minutes"
               }}
           }}
       }}

       Important:

       - **Output ONLY the JSON object. Do not include any additional text or explanations.**
       - **Ensure the JSON is valid and properly formatted.**
       - **Do not include code snippets, code blocks, or any markdown formatting.**
       - **Use double quotes for all JSON keys and string values as per JSON standard.**
       - **Do not include comments within the JSON.**

       Include all necessary configuration based on the analysis.
       Use environment variables in the format ${{VARIABLE_NAME}} (e.g., ${{API_TOKEN}}, ${{DATASET_ID}}).
       Ensure all tasks, volumes, and configurations from the analysis are included in the JSON structure.
       """
    
    def generate_template(self, analysis: str) -> Optional[str]:
        """Generate YAML template from analysis."""
        try:
            # Get template structure using LLM
            response = self.llm_provider.generate_completion(
                system_prompt=self._get_system_prompt(),
                user_input=self._get_template_prompt(analysis)
            )

            # Parse JSON response
            try:
                config_dict = json.loads(response)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {e}")
                return None

            # Process the configuration
            if config_dict:
                # Quote schedule_interval if it starts with @
                if 'dag_config' in config_dict and 'schedule_interval' in config_dict['dag_config']:
                    schedule = config_dict['dag_config']['schedule_interval']
                    if isinstance(schedule, str) and schedule.startswith('@'):
                        config_dict['dag_config']['schedule_interval'] = f"'{schedule}'"

                # Process environment variables in tasks
                if 'tasks' in config_dict:
                    for task in config_dict['tasks']:
                        if 'environment' in task:
                            env_vars = {}
                            for key, value in task['environment'].items():
                                if isinstance(value, str):
                                    if value.startswith('${') and value.endswith('}'):
                                        var_name = value[2:-1]
                                        env_vars[key] = f"'{{{{ var.value.{var_name} }}}}'"
                                    else:
                                        env_vars[key] = value
                                else:
                                    env_vars[key] = value
                            task['environment'] = env_vars

            # Convert to YAML
            yaml_content = yaml.dump(
                config_dict,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
                indent=2
            )
            
            return yaml_content

        except Exception as e:
            logging.error(f"Error generating template: {e}")
            return None

    def save_template(self, content: str, analysis_file: str) -> str:
        """Save YAML template to file using timestamp from analysis filename."""
        try:
            # Extract timestamp from analysis filename
            timestamp = Path(analysis_file).stem.split('_')[-1]
            
            # Create output filename
            output_dir = Path(self.config['pipeline_settings']['output_directory'])
            output_file = output_dir / f"generated_yaml_{timestamp}.yaml"
            
            # Ensure output directory exists
            output_dir.mkdir(exist_ok=True)
            
            # Save content
            with open(output_file, 'w') as f:
                f.write(content)
            
            logging.info(f"Template saved to: {output_file}")
            return str(output_file)
        except Exception as e:
            logging.error(f"Error saving template: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Generate YAML template from analysis')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    parser.add_argument('--analysis', required=True, help='Path to pipeline analysis file')
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

        # Read analysis content
        with open(args.analysis, 'r') as f:
            analysis_content = f.read()

        # Initialize generator
        generator = YAMLTemplateGenerator(config)
        
        # Generate template
        yaml_content = generator.generate_template(analysis_content)
        if yaml_content:
            # Save template
            output_file = generator.save_template(yaml_content, args.analysis)
            logging.info(f"Template generation completed successfully using {config['model_settings']['active_provider']}")
        else:
            logging.error("Failed to generate template")
            
    except Exception as e:
        logging.error(f"Template generation failed: {e}")
        raise

if __name__ == "__main__":
    main()
