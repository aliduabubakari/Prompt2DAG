import json
import argparse
import logging
import os
import anthropic
import argparse
import logging
from typing import Dict, Any
from datetime import datetime
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

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(config):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['log_file']),
            logging.StreamHandler()
        ]
    )

def read_pipeline_description(input_file):
    """Read pipeline description from input file."""
    try:
        with open(input_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        raise

def analyze_pipeline(user_input: str, config: Dict) -> str:
    """Analyze pipeline using the configured model."""
    if not user_input.strip():
        raise ValueError("Pipeline description cannot be empty.")

    system_prompt = """You are an expert Apache Airflow DAG designer. Your task is to analyze the provided pipeline description and output ONLY the analysis section in the following format:

    <analysis>
    - Key Components:  
    [List of components]
    
    - Dependencies:  
    [List of dependencies]
    
    - Configuration Requirements:  
    [List of configurations]
    
    - Docker Specifications:  
    [Docker details]
    
    - Error Handling:  
    [Error handling details]
    
    - Additional Considerations:  
    [Additional details]
    </analysis>

    Ensure your response contains ONLY the content within the <analysis> tags."""

    try:
        llm_provider = LLMProvider(config)
        return llm_provider.generate_completion(system_prompt, user_input)
    except Exception as e:
        logging.error(f"Error during pipeline analysis: {e}")
        raise

def save_output(content, config):
    """Save analysis output to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config['pipeline_settings']['output_directory']
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(
        output_dir,
        f"pipeline_analysis_{timestamp}.txt"
    )
    
    try:
        with open(output_file, 'w') as f:
            f.write(content)
        logging.info(f"Analysis saved to: {output_file}")
        return output_file
    except IOError as e:
        logging.error(f"Error saving analysis: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Run pipeline analysis')
    parser.add_argument('--config', default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--provider', 
                      help='Override the LLM provider (deepinfra, openai, or claude)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override provider if specified
    if args.provider:
        if args.provider not in ['deepinfra', 'openai', 'claude']:
            raise ValueError(f"Unsupported provider: {args.provider}")
        config['model_settings']['active_provider'] = args.provider
    
    # Setup logging
    setup_logging(config)
    
    try:
        # Read pipeline description
        pipeline_description = read_pipeline_description(
            config['pipeline_settings']['input_file']
        )
        
        # Analyze pipeline
        analysis = analyze_pipeline(pipeline_description, config)
        
        # Save output
        output_file = save_output(analysis, config)
        
        logging.info(f"Pipeline analysis completed successfully using {config['model_settings']['active_provider']}")
        
    except Exception as e:
        logging.error(f"Pipeline analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()