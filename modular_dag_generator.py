import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any 
from openai import OpenAI
import yaml
import anthropic
import argparse

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

class ModularDAGGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.modular_config = config['modular_generation']
        self.llm_provider = LLMProvider(config)  # Initialize LLMProvider
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

    def load_yaml_config(self, filename: str) -> Dict:
        logging.info(f"Loading YAML configuration from {filename}")
        try:
            with open(filename, 'r') as file:
                content = file.read()
                return yaml.safe_load(content)
        except Exception as e:
            logging.error(f"Error loading YAML file: {e}")
            raise

    def create_dag_config_prompt(self, content: Dict) -> str:
        """Create detailed prompt for DAG configuration generation."""
        prompt = f"""
        Create a production-ready Apache Airflow DAG configuration with these specifications:

        Basic Configuration:
        - DAG ID: {content.get('dag_id', 'dynamic_dag')}
        - Schedule Interval: {content.get('schedule_interval', '@daily')}
        
        Default Arguments:
        - Owner: {content['default_args'].get('owner', 'airflow')}
        - Start Date: {content['default_args'].get('start_date', 'current_date')}
        - Retries: {content['default_args'].get('retries', 3)}
        - Retry Delay: {content['default_args'].get('retry_delay', 300)}
        - Depends on Past: {content['default_args'].get('depends_on_past', False)}
        - Email on Failure: {content['default_args'].get('email_on_failure', False)}
        - Email on Retry: {content['default_args'].get('email_on_retry', False)}
        
        Requirements:
        1. Include proper imports for all components
        2. Set up error notification configuration
        3. Configure proper retry policies
        4. Include documentation and description
        5. Use proper Airflow variable references
        """

        if self.modular_config['docker_based']:
            prompt += """
            Docker-Specific Requirements:
            1. Include Docker-specific imports
            2. Configure Docker connection settings
            3. Set up proper Docker network configuration
            4. Include volume mounting configurations
            """

        return prompt.strip()

    def create_tasks_prompt(self, tasks: List[Dict]) -> str:
        """Create detailed prompt for task generation."""
        task_details = []
        for task in tasks:
            task_detail = f"""
            Task: {task.get('task_id')}
            - Operator: {task.get('operator', 'DockerOperator')}
            - Image: {task.get('image')}
            - Command: {task.get('command')}
            - Resources: {json.dumps(task.get('resources', {}), indent=2)}
            - Volumes: {json.dumps(task.get('volumes', []), indent=2)}
            - Docker Connection: {task.get('docker_conn_id', 'docker_default')}
            - Auto Remove: {task.get('auto_remove', True)}
            - API Version: {task.get('api_version', 'auto')}
            """
            task_details.append(task_detail)

        prompt = f"""
        Create Airflow task definitions with these specifications:

        Task Definitions:
        {''.join(task_details)}

        Requirements:
        1. Use DockerOperator for all tasks
        2. Include comprehensive error handling
        3. Set up proper logging
        4. Configure resource limits
        5. Handle volume mounts correctly
        6. Use Airflow variables for paths
        7. Configure proper network settings
        8. Set up proper container cleanup
        """

        return prompt.strip()

    def create_dependencies_prompt(self, content: List[Dict]) -> str:
        """Create detailed prompt for dependency generation."""
        # Build dependency graph representation
        dependencies = []
        for task in content:
            task_id = task.get('task_id', 'unnamed_task')
            deps = task.get('depends_on', [])
            if deps:
                dependencies.append(f"{task_id} depends on: {', '.join(deps)}")

        prompt = f"""
        Create task dependencies for the following relationships:

        Dependency Structure:
        {'\n'.join(dependencies)}

        Requirements:
        1. Use proper Airflow dependency syntax (>>)
        2. Ensure all task references are valid
        3. Create clear dependency chains
        4. Handle parallel task execution where appropriate
        5. Consider task groups if needed
        """

        return prompt.strip()

    def create_assembly_prompt(self, components: Dict[str, str]) -> str:
        """Create detailed prompt for final DAG assembly."""
        prompt = f"""
        Create a complete, production-ready Apache Airflow DAG by combining these components:

        1. DAG Configuration:
        {components['dag_config']}

        2. Task Definitions:
        {components['tasks']}

        3. Task Dependencies:
        {components['dependencies']}

        Assembly Requirements:
        1. Ensure proper organization of imports
        2. Include comprehensive docstring with:
           - DAG description
           - Author information
           - Creation date
           - Dependencies and requirements
        3. Follow PEP 8 style guidelines
        4. Include proper error handling
        5. Add logging statements at key points
        6. Ensure all variables are properly defined
        7. Include any necessary Airflow configurations
        """

        if self.modular_config['docker_based']:
            prompt += """
            Docker-Specific Requirements:
            1. Ensure proper Docker connection configuration
            2. Include necessary Docker-specific imports
            3. Configure Docker network settings
            4. Set up volume mounts correctly
            5. Include container cleanup logic
            """

        return prompt.strip()

    def generate_component(self, component_type: str, content: Dict) -> str:
        """Generate a specific DAG component using appropriate prompt."""
        logging.info(f"Generating {component_type} component")
        component_config = self.modular_config['components'][component_type]
        
        # Get the appropriate prompt creation method
        prompt_methods = {
            'dag_config': self.create_dag_config_prompt,
            'tasks': self.create_tasks_prompt,
            'dependencies': self.create_dependencies_prompt,
            'assembly': self.create_assembly_prompt
        }

        system_prompt = f"You are an expert Apache Airflow DAG developer, specialized in {component_type} generation."
        user_prompt = prompt_methods[component_type](content)
        
        try:
            response = self.llm_provider.generate_completion(
                system_prompt=system_prompt,
                user_input=user_prompt
            )
            return response
        except Exception as e:
            logging.error(f"Error generating {component_type}: {e}")
            raise
    
    def generate_dag(self, yaml_config: Dict) -> str:
        logging.info("Starting modular DAG generation")
        
        # Restructure the YAML content to match our component needs
        dag_config = {
            'dag_id': yaml_config.get('dag_id'),
            'schedule_interval': yaml_config.get('schedule_interval'),
            'default_args': {
                'owner': yaml_config.get('default_args', {}).get('owner', 'airflow'),
                'start_date': yaml_config.get('start_date'),
                'depends_on_past': yaml_config.get('default_args', {}).get('depends_on_past', False),
                'email_on_failure': yaml_config.get('default_args', {}).get('email_on_failure', False),
                'email_on_retry': yaml_config.get('default_args', {}).get('email_on_retry', False),
                'retries': yaml_config.get('default_args', {}).get('retries', 3),
                'retry_delay': yaml_config.get('default_args', {}).get('retry_delay', 300)
            }
        }

        # Generate individual components
        components = {
            'dag_config': self.generate_component('dag_config', dag_config),
            'tasks': self.generate_component('tasks', yaml_config['tasks']),
            'dependencies': self.generate_component('dependencies', yaml_config['tasks'])
        }
        
        # Add network configuration if present
        if 'network' in yaml_config:
            components['network'] = yaml_config['network']
        
        # Assemble final DAG
        final_dag = self.generate_component('assembly', components)
        
        return self.extract_code(final_dag)
    
    def extract_code(self, content: str) -> str:
        """Extract clean Python code from response"""
        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        return content.strip()

    def save_dag_code(self, dag_code: str, yaml_file: str) -> str:
        """Save DAG code to file with timestamp and provider info"""
        timestamp = Path(yaml_file).stem.split('_')[-1]
        provider = self.config['model_settings']['active_provider']
        output_dir = Path(self.config['pipeline_settings']['output_directory'])
        output_file = output_dir / f"generated_dag_{provider}_{timestamp}.py"
        
        output_dir.mkdir(exist_ok=True)
        
        # Add generation metadata as comments
        metadata = f"""# Generated using {provider.upper()} LLM
    # Model: {self.config['model_settings'][provider]['model_name']}
    # Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    # Source YAML: {yaml_file}

    """
        
        with open(output_file, 'w') as f:
            f.write(metadata + dag_code)
        
        logging.info(f"DAG code saved to: {output_file}")
        return str(output_file)

def main():
    parser = argparse.ArgumentParser(description='Generate DAG code using modular approach')
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
        generator = ModularDAGGenerator(config)
        
        # Load YAML configuration
        yaml_config = generator.load_yaml_config(args.yaml)
        
        # Generate DAG
        dag_code = generator.generate_dag(yaml_config)
        
        # Save DAG code
        output_file = generator.save_dag_code(dag_code, args.yaml)
        
        logging.info(f"DAG generation completed successfully using {config['model_settings']['active_provider']}")
        
    except Exception as e:
        logging.error(f"DAG generation failed: {e}")
        raise

if __name__ == "__main__":
    main()