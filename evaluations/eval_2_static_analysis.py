import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import ast
from typing import Dict
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from pylint.lint import Run
from pylint.reporters import JSONReporter
from io import StringIO
import tempfile
import os
import bandit
from bandit.core import manager as bandit_manager
from bandit.core import config as bandit_config
import subprocess

class StaticDAGAnalyzer:
    def __init__(self, config: dict, dag_code: str):
        self.config = config
        self.dag_code = dag_code
        self.weights = config['evaluations']['static_analysis']['weights']
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

    def analyze_with_flake8(self) -> Dict:
        """Analyze code using flake8"""
        logging.info("Running Flake8 analysis...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(self.dag_code)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(
                ['flake8', '--format=json', temp_file_path],
                capture_output=True,
                text=True
            )
            
            flake8_results = []
            if result.stdout:
                try:
                    flake8_results = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass

            violation_count = len(flake8_results)
            score = max(0.0, 10.0 - (violation_count * 0.5))

            return {
                'score': score,
                'violations': flake8_results
            }
        finally:
            os.unlink(temp_file_path)

    def analyze_code_quality(self) -> Dict:
        """Analyze code quality using Pylint"""
        logging.info("Running Pylint analysis...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(self.dag_code)
            temp_file_path = temp_file.name
        
        try:
            output_buffer = StringIO()
            reporter = JSONReporter(output_buffer)
            
            try:
                Run([temp_file_path], reporter=reporter)
            except SystemExit:
                pass
            
            output_buffer.seek(0)
            pylint_json = json.loads(output_buffer.getvalue())
            
            error_count = sum(1 for msg in pylint_json if msg.get('type') in ['error', 'fatal'])
            warning_count = sum(1 for msg in pylint_json if msg.get('type') == 'warning')
            score = max(0.0, 10.0 - (error_count * 2 + warning_count * 0.5))
            
            return {
                'score': score,
                'messages': [
                    {
                        'type': msg.get('type', ''),
                        'module': msg.get('module', ''),
                        'obj': msg.get('obj', ''),
                        'line': msg.get('line', 0),
                        'column': msg.get('column', 0),
                        'message': msg.get('message', ''),
                        'symbol': msg.get('symbol', '')
                    }
                    for msg in pylint_json
                ]
            }
        finally:
            os.unlink(temp_file_path)

    def analyze_security(self) -> Dict:
        """Analyze security using Bandit"""
        logging.info("Running security analysis with Bandit...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(self.dag_code)
            temp_file_path = temp_file.name
                
        try:
            # Configure and run Bandit
            b_conf = bandit_config.BanditConfig()
            b_mgr = bandit_manager.BanditManager(b_conf, 'json')
            b_mgr.discover_files([temp_file_path])
            b_mgr.run_tests()
            
            # Get results
            bandit_output = b_mgr.get_issue_list()
            
            # Process results
            issue_count = len(bandit_output)
            security_score = max(0.0, 10.0 - (issue_count * 2))
            
            # Safely extract issue details
            findings = []
            for issue in bandit_output:
                finding = {
                    'severity': getattr(issue, 'severity', 'unknown'),
                    'confidence': getattr(issue, 'confidence', 'unknown'),
                    'description': getattr(issue, 'text', ''),
                    'line': getattr(issue, 'lineno', 0),
                    'filename': getattr(issue, 'fname', ''),
                }
                
                # Add optional attributes if they exist
                if hasattr(issue, 'test_id'):
                    finding['test_id'] = issue.test_id
                if hasattr(issue, 'issue_text'):
                    finding['issue_text'] = issue.issue_text
                
                findings.append(finding)

            return {
                'score': security_score,
                'security_issues': issue_count,
                'findings': findings,
                'summary': {
                    'total_issues': issue_count,
                    'high_severity': len([i for i in bandit_output if getattr(i, 'severity', '') == 'HIGH']),
                    'medium_severity': len([i for i in bandit_output if getattr(i, 'severity', '') == 'MEDIUM']),
                    'low_severity': len([i for i in bandit_output if getattr(i, 'severity', '') == 'LOW'])
                }
            }
        except Exception as e:
            logging.error(f"Error in security analysis: {e}")
            return {
                'score': 0.0,
                'security_issues': 0,
                'findings': [],
                'summary': {
                    'total_issues': 0,
                    'high_severity': 0,
                    'medium_severity': 0,
                    'low_severity': 0
                },
                'error': str(e)
            }
        finally:
            os.unlink(temp_file_path)

    def analyze_complexity(self) -> Dict:
        """Analyze code complexity using Radon"""
        logging.info("Running complexity analysis with Radon...")
        try:
            # Get complexity metrics
            complexity_results = list(cc_visit(self.dag_code))
            
            # Calculate maintainability index
            mi_result = mi_visit(self.dag_code, multi=True)
            
            # Process complexity results
            if complexity_results:
                # Calculate metrics
                complexities = [result.complexity for result in complexity_results]
                total_complexity = sum(complexities)
                avg_complexity = total_complexity / len(complexities)
                max_complexity = max(complexities)
                
                # Score calculation based on average complexity
                # Adjust these thresholds as needed
                if avg_complexity <= 5:
                    complexity_score = 10.0
                elif avg_complexity <= 10:
                    complexity_score = 7.5
                elif avg_complexity <= 20:
                    complexity_score = 5.0
                else:
                    complexity_score = 2.5
                
                # Safely extract function details
                complexity_details = []
                for result in complexity_results:
                    detail = {
                        'name': getattr(result, 'name', 'unknown'),
                        'complexity': result.complexity,
                        'lineno': getattr(result, 'lineno', 0),
                    }
                    
                    # Add optional attributes if they exist
                    if hasattr(result, 'classname'):
                        detail['classname'] = result.classname
                    if hasattr(result, 'method'):
                        detail['method'] = result.method
                        
                    complexity_details.append(detail)

                return {
                    'score': complexity_score,
                    'metrics': {
                        'average_complexity': round(avg_complexity, 2),
                        'max_complexity': max_complexity,
                        'total_complexity': total_complexity,
                        'total_functions': len(complexity_results),
                        'maintainability_index': mi_result
                    },
                    'details': complexity_details,
                    'complexity_distribution': {
                        'simple': len([r for r in complexity_results if r.complexity <= 5]),
                        'moderate': len([r for r in complexity_results if 5 < r.complexity <= 10]),
                        'complex': len([r for r in complexity_results if r.complexity > 10])
                    },
                    'summary': {
                        'complexity_risk': 'Low' if avg_complexity <= 5 else 
                                        'Medium' if avg_complexity <= 10 else 
                                        'High'
                    }
                }
            else:
                return {
                    'score': 10.0,  # Perfect score if no complex functions found
                    'metrics': {
                        'average_complexity': 0,
                        'max_complexity': 0,
                        'total_complexity': 0,
                        'total_functions': 0,
                        'maintainability_index': mi_result
                    },
                    'details': [],
                    'complexity_distribution': {
                        'simple': 0,
                        'moderate': 0,
                        'complex': 0
                    },
                    'summary': {
                        'complexity_risk': 'Low'
                    }
                }
                
        except Exception as e:
            logging.error(f"Error in complexity analysis: {e}")
            return {
                'score': 0.0,
                'error': str(e),
                'metrics': {
                    'average_complexity': 0,
                    'max_complexity': 0,
                    'total_functions': 0,
                    'maintainability_index': None
                },
                'details': [],
                'complexity_distribution': {
                    'simple': 0,
                    'moderate': 0,
                    'complex': 0
                },
                'summary': {
                    'complexity_risk': 'Unknown'
                }
            }

    def run_analysis(self) -> Dict:
        """Run all analyses and combine results"""
        results = {}
        
        analyses = {
            'code_quality': self.analyze_code_quality,
            'security': self.analyze_security,
            'complexity': self.analyze_complexity,
            'flake8': self.analyze_with_flake8
        }
        
        for category, analysis_func in analyses.items():
            try:
                results[category] = analysis_func()
                logging.info(f"Completed {category} analysis with score: {results[category]['score']}")
            except Exception as e:
                logging.error(f"Error in {category} analysis: {e}")
                results[category] = {
                    'score': 0.0,
                    'error': str(e)
                }
        
        # Calculate weighted score
        weighted_scores = []
        for category, weight in self.weights.items():
            if category in results and 'score' in results[category]:
                weighted_scores.append(results[category]['score'] * weight)
        
        total_score = sum(weighted_scores)
        
        # Add metadata
        results['overall_score'] = round(total_score, 2)
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'weights_used': self.weights,
            'successful_analyses': [
                category for category in self.weights.keys()
                if category in results and 'error' not in results[category]
            ],
            'thresholds': self.config['evaluations']['static_analysis']['thresholds']
        }
        
        return results

def save_analysis_results(results: Dict, dag_file: str, config: Dict):
    """Save analysis results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['pipeline_settings']['output_directory'])
    output_file = output_dir / f"evaluation_2_results_{timestamp}.json"
    
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
    parser = argparse.ArgumentParser(description='Static analysis of DAG code')
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
        analyzer = StaticDAGAnalyzer(config, dag_code)
        
        # Run analysis
        logging.info("Starting static analysis...")
        results = analyzer.run_analysis()
        
        # Save results
        output_file = save_analysis_results(results, args.dag, config)
        
        # Display results
        print(f"\nStatic Analysis Results:")
        print(f"Overall Score: {results['overall_score']:.2f}")
        print("\nCategory Scores:")
        for category in analyzer.weights.keys():
            print(f"{category}: {results[category]['score']:.2f}")
            
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()