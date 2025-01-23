import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import ast
from typing import Dict, List, Tuple
import networkx as nx

class DAGStructureAnalyzer:
    def __init__(self, config: dict, dag_code: str):
        self.config = config
        self.dag_code = dag_code
        self.weights = config['evaluations']['dag_structure']['weights']
        self.thresholds = config['evaluations']['dag_structure']['thresholds']
        self.dag_structure = self._parse_dag_structure()
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

    def _parse_dag_structure(self) -> Dict:
        """Parse DAG structure from code using AST"""
        logging.info("Parsing DAG structure...")
        try:
            tree = ast.parse(self.dag_code)
            dag_info = {
                'tasks': [],
                'dependencies': [],
                'docker_operators': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if isinstance(node.value, ast.Call):
                                if hasattr(node.value.func, 'id'):
                                    if node.value.func.id == 'DockerOperator':
                                        dag_info['docker_operators'].append({
                                            'task_id': self._extract_task_id(node),
                                            'image': self._extract_docker_image(node),
                                            'config': self._extract_docker_config(node)
                                        })
                                    dag_info['tasks'].append(target.id)
                
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
                    dep = (self._extract_task_name(node.left), 
                          self._extract_task_name(node.right))
                    if all(dep):
                        dag_info['dependencies'].append(dep)
            
            return dag_info
        except Exception as e:
            logging.error(f"Failed to parse DAG structure: {e}")
            return {'tasks': [], 'dependencies': [], 'docker_operators': []}

    def _extract_task_id(self, node):
        """Extract task_id from DockerOperator definition"""
        for kw in node.value.keywords:
            if kw.arg == 'task_id':
                if isinstance(kw.value, ast.Str):
                    return kw.value.s
        return None

    def _extract_docker_image(self, node):
        """Extract image from DockerOperator definition"""
        for kw in node.value.keywords:
            if kw.arg == 'image':
                if isinstance(kw.value, ast.Str):
                    return kw.value.s
        return None

    def _extract_docker_config(self, node):
        """Extract Docker configuration from operator"""
        config = {}
        for kw in node.value.keywords:
            if isinstance(kw.value, ast.Str):
                config[kw.arg] = kw.value.s
            elif isinstance(kw.value, ast.Dict):
                config[kw.arg] = 'dict_config_present'
        return config

    def _extract_task_name(self, node):
        """Extract task name from node"""
        if isinstance(node, ast.Name):
            return node.id
        return None

    def analyze_structure(self) -> Dict:
        """Analyze DAG structure"""
        logging.info("Analyzing DAG structure...")
        try:
            score = 100
            issues = []
            
            if len(self.dag_structure['tasks']) < self.thresholds['min_tasks']:
                score -= 30
                issues.append("Insufficient number of tasks")
            
            if len(self.dag_structure['dependencies']) < self.thresholds['min_dependencies']:
                score -= 20
                issues.append("Insufficient task dependencies")
            
            if not self.dag_structure['docker_operators']:
                score -= 20
                issues.append("No DockerOperators found")
            
            return {
                'score': max(0, score),
                'issues': issues,
                'structure': self.dag_structure
            }
        except Exception as e:
            logging.error(f"Structure analysis failed: {e}")
            return {'score': 0, 'error': str(e)}

    def analyze_docker_config(self) -> Dict:
        """Analyze Docker configurations"""
        logging.info("Analyzing Docker configurations...")
        try:
            score = 100
            issues = []
            required_configs = self.thresholds['required_docker_configs']
            
            for operator in self.dag_structure['docker_operators']:
                for config in required_configs:
                    if config not in operator['config']:
                        score -= (100 / len(required_configs)) / len(self.dag_structure['docker_operators'])
                        issues.append(f"Missing {config} for task {operator['task_id']}")
            
            return {
                'score': max(0, score),
                'issues': issues
            }
        except Exception as e:
            logging.error(f"Docker configuration analysis failed: {e}")
            return {'score': 0, 'error': str(e)}

    def run_analysis(self) -> Dict:
        """Run complete DAG analysis"""
        structure_results = self.analyze_structure()
        docker_results = self.analyze_docker_config()
        
        weighted_score = (
            structure_results['score'] * self.weights['structure'] +
            docker_results['score'] * self.weights['docker_config']
        )
        
        return {
            'overall_score': round(weighted_score, 2),
            'structure_analysis': structure_results,
            'docker_analysis': docker_results
        }

def save_analysis_results(results: Dict, dag_file: str, config: Dict):
    """Save analysis results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['pipeline_settings']['output_directory'])
    output_file = output_dir / f"evaluation_3_results_{timestamp}.json"
    
    analysis_results = {
        'dag_file': dag_file,
        'timestamp': timestamp,
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    logging.info(f"Analysis results saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Analyze DAG structure and configuration')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    parser.add_argument('--dag', required=True, help='Path to DAG Python file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    try:
        # Read DAG code
        with open(args.dag, 'r') as f:
            dag_code = f.read()

        # Initialize analyzer
        analyzer = DAGStructureAnalyzer(config, dag_code)
        
        # Run analysis
        logging.info("Starting DAG structure analysis...")
        results = analyzer.run_analysis()
        
        # Save results
        output_file = save_analysis_results(results, args.dag, config)
        
        # Display results
        print(f"\nDAG Structure Analysis Results:")
        print(f"Overall Score: {results['overall_score']:.2f}")
        print("\nStructure Analysis:")
        print(f"Score: {results['structure_analysis']['score']:.2f}")
        if results['structure_analysis']['issues']:
            print("Issues:")
            for issue in results['structure_analysis']['issues']:
                print(f"- {issue}")
        
        print("\nDocker Configuration Analysis:")
        print(f"Score: {results['docker_analysis']['score']:.2f}")
        if results['docker_analysis']['issues']:
            print("Issues:")
            for issue in results['docker_analysis']['issues']:
                print(f"- {issue}")
            
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()