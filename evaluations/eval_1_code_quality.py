import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import openai
import re
from typing import Dict, List

class EnhancedDockerDAGAnalyzer:
    def __init__(self, config: dict, dag_code: str):
        self.config = config
        self.dag_code = dag_code
        self.weights = config['evaluations']['code_quality']['weights']
        
        # Use OpenAI settings from evaluations section
        openai_config = config['evaluations']['llm_settings']['openai']
        self.openai_client = openai.OpenAI(
            api_key=openai_config['api_key']
        )
        self.model_name = openai_config['model_name']
        self.max_tokens = openai_config['max_tokens']
        self.temperature = openai_config['temperature']
        
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

    def extract_score(self, response_text: str) -> float:
        match = re.search(r'\b\d+(\.\d+)?\b', response_text)
        if match:
            return float(match.group())
        raise ValueError("No numeric score found in the response.")

    def evaluate_category(self, category: str) -> float:
        prompts = {
            'syntax_and_validity': """
                Evaluate the syntax and validity of the following Python code for an Airflow DAG:
                Consider: Compliance with Python syntax rules, absence of errors, proper imports
            """,
            'code_readability': """
                Evaluate the readability of the following Python code for an Airflow DAG:
                Consider: Meaningful names, comments, docstrings, formatting, style guidelines
            """,
            'modularity_and_reusability': """
                Evaluate the modularity and reusability of the following Python code for an Airflow DAG:
                Consider: Function/class usage, separation of concerns, reusability potential
            """,
            'docker_configuration': """
                Evaluate the Docker configuration in the following Python code for an Airflow DAG:
                Consider: Docker parameters, environment variables, volume mounting, best practices
            """,
            'functional_completeness': """
                Evaluate the functional completeness of the following Python code for an Airflow DAG:
                Consider: Task dependencies, error handling, logging, monitoring, documentation
            """
        }

        prompt = f"{prompts[category]}\n\n{self.dag_code}\n\nProvide a numeric score between 0 and 10."

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a code evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                seed=self.config['evaluations']['code_quality']['seed'],
                max_tokens=self.max_tokens
            )
            return self.extract_score(response.choices[0].message.content.strip())
        except Exception as e:
            logging.error(f"Error evaluating {category}: {str(e)}")
            raise

    def calculate_total_score(self) -> dict:
        scores = {
            category: self.evaluate_category(category)
            for category in self.weights.keys()
        }

        weighted_scores = {
            category: score * self.weights[category]
            for category, score in scores.items()
        }

        total_score = sum(weighted_scores.values())

        return {
            'total_score': round(total_score, 2),
            'category_scores': {k: round(v, 2) for k, v in scores.items()},
            'weighted_scores': {k: round(v, 2) for k, v in weighted_scores.items()}
        }

def save_evaluation_results(results: Dict, dag_file: str, config: Dict):
    """Save evaluation results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['pipeline_settings']['output_directory'])
    output_file = output_dir / f"evaluation_1_results_{timestamp}.json"
    
    evaluation_results = {
        'dag_file': dag_file,
        'timestamp': timestamp,
        'results': results,
        'evaluation_provider': config['evaluations']['llm_settings']['provider'],
        'model_name': config['evaluations']['llm_settings']['openai']['model_name']
    }
    
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logging.info(f"Evaluation results saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Evaluate DAG code quality')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    parser.add_argument('--dag', required=True, help='Path to DAG Python file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Verify evaluation settings
    if not config['evaluations']['code_quality']['enabled']:
        logging.error("Code quality evaluation is disabled in configuration")
        return

    if config['evaluations']['llm_settings']['provider'] != 'openai':
        logging.error(f"Unsupported LLM provider for evaluation: {config['evaluations']['llm_settings']['provider']}")
        return

    try:
        # Read DAG code
        with open(args.dag, 'r') as f:
            dag_code = f.read()

        # Initialize analyzer
        analyzer = EnhancedDockerDAGAnalyzer(config, dag_code)
        
        # Run multiple iterations
        iterations = config['evaluations']['code_quality']['iterations']
        all_results = []
        
        for i in range(iterations):
            logging.info(f"Running evaluation iteration {i+1}/{iterations}")
            results = analyzer.calculate_total_score()
            all_results.append(results)

        # Calculate averages
        avg_results = {
            'total_score': sum(r['total_score'] for r in all_results) / iterations,
            'category_scores': {
                category: sum(r['category_scores'][category] for r in all_results) / iterations
                for category in all_results[0]['category_scores']
            },
            'weighted_scores': {
                category: sum(r['weighted_scores'][category] for r in all_results) / iterations
                for category in all_results[0]['weighted_scores']
            }
        }

        # Save results
        output_file = save_evaluation_results(avg_results, args.dag, config)
        
        # Display results
        model_name = config['evaluations']['llm_settings']['openai']['model_name']
        print(f"\nEvaluation Results (using {model_name}):")
        print(f"Total Score: {avg_results['total_score']:.2f}")
        print("\nCategory Scores:")
        for category, score in avg_results['category_scores'].items():
            print(f"{category}: {score:.2f}")
        print("\nWeighted Scores:")
        for category, score in avg_results['weighted_scores'].items():
            print(f"{category}: {score:.2f}")
            
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()