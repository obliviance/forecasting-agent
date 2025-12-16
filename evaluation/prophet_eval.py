import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from agent.agentcast import agentcast


class ProphetsEvaluator:
    """Evaluator for the AgentCast forecasting agent against the Manifold dataset."""

    def __init__(self, dataset_path: str):
        """
        Initialize the evaluator.
        
        Args:
            dataset_path: Path to the JSON dataset file
        """
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
        self.results = []

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the dataset from JSON file."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return list(data.values())
                else:
                    raise ValueError("Dataset must be a list or dictionary")
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {self.dataset_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON: {e}")
            return []

    def _extract_question(self, item: Dict[str, Any]) -> str:
        """
        Extract the question from a dataset item.
        
        Args:
            item: A dataset item
            
        Returns:
            The question string
        """
        # Try multiple possible keys for the question
        question_keys = ['question', 'title', 'description', 'prompt']
        for key in question_keys:
            if key in item:
                return item[key]
        
        # If no standard key found, return empty string
        return ""

    def _extract_ground_truth(self, item: Dict[str, Any]) -> float:
        """
        Extract the ground truth probability from a dataset item.
        
        Args:
            item: A dataset item
            
        Returns:
            Ground truth probability as a float between 0 and 1
        """
        # Try multiple possible keys for ground truth
        truth_keys = ['ground_truth', 'actual', 'resolution', 'resolved', 'community_prediction']
        
        for key in truth_keys:
            if key in item:
                value = item[key]
                # Handle percentage strings like "14%"
                if isinstance(value, str):
                    if '%' in value:
                        try:
                            return float(value.strip('%')) / 100.0
                        except ValueError:
                            pass
                    # Try parsing as float
                    try:
                        return float(value)
                    except ValueError:
                        pass
                elif isinstance(value, (int, float)):
                    # Normalize to 0-1 range if needed
                    if value > 1:
                        return value / 100.0
                    return float(value)
        
        return None

    def _calculate_brier_score(self, prediction: float, ground_truth: float) -> float:
        """
        Calculate the Brier score for a single prediction.
        
        Brier Score = (prediction - ground_truth)^2
        Lower is better (0 = perfect, 1 = worst)
        
        Args:
            prediction: Predicted probability (0-1)
            ground_truth: Ground truth probability (0-1)
            
        Returns:
            Brier score
        """
        if prediction is None or ground_truth is None:
            return None
        
        # Clamp prediction to [0, 1] range
        prediction = max(0.0, min(1.0, prediction))
        ground_truth = max(0.0, min(1.0, ground_truth))
        
        return (prediction - ground_truth) ** 2

    def _calculate_log_loss(self, prediction: float, ground_truth: float) -> float:
        """
        Calculate log loss (cross-entropy) for a single prediction.
        
        Log Loss = -[y * log(p) + (1-y) * log(1-p)]
        Lower is better.
        
        Args:
            prediction: Predicted probability (0-1)
            ground_truth: Ground truth (0 or 1)
            
        Returns:
            Log loss
        """
        import math
        
        if prediction is None or ground_truth is None:
            return None
        
        # Clamp prediction to avoid log(0)
        epsilon = 1e-15
        prediction = max(epsilon, min(1 - epsilon, prediction))
        
        # Treat ground truth as binary (0 or 1)
        ground_truth = 1 if ground_truth > 0.5 else 0
        
        return -(ground_truth * math.log(prediction) + (1 - ground_truth) * math.log(1 - prediction))

    def _extract_prediction_float(self, prediction_result: Any) -> float:
        """
        Extract a float prediction from various possible formats.
        
        Args:
            prediction_result: The prediction result from agentcast
            
        Returns:
            A float between 0 and 1, or None if extraction failed
        """
        if isinstance(prediction_result, str):
            # Try to parse from string
            try:
                # Remove percentage signs
                cleaned = prediction_result.strip().replace('%', '')
                return float(cleaned) / 100.0
            except (ValueError, AttributeError):
                return None
        elif isinstance(prediction_result, (int, float)):
            # Normalize to 0-1 range if needed
            value = float(prediction_result)
            if value > 1:
                return value / 100.0
            return value
        elif isinstance(prediction_result, dict):
            # Try to extract from dictionary
            if 'prediction' in prediction_result:
                return self._extract_prediction_float(prediction_result['prediction'])
        
        return None

    def evaluate_single(self, item: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        """
        Evaluate the agent on a single dataset item.
        
        Args:
            item: A dataset item
            index: Index of the item in the dataset
            
        Returns:
            A dictionary containing evaluation results
        """
        question = self._extract_question(item)
        ground_truth = self._extract_ground_truth(item)
        
        if not question:
            return {
                'index': index,
                'question': 'N/A',
                'error': 'Could not extract question from item',
                'ground_truth': ground_truth,
                'prediction': None,
                'brier_score': None,
                'log_loss': None
            }
        
        if ground_truth is None:
            return {
                'index': index,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'error': 'Could not extract ground truth from item',
                'ground_truth': None,
                'prediction': None,
                'brier_score': None,
                'log_loss': None
            }
        
        try:
            # Get prediction from agentcast
            prediction_result, reasoning = agentcast(question)
            prediction = self._extract_prediction_float(prediction_result)
            
            if prediction is None:
                return {
                    'index': index,
                    'question': question[:100] + '...' if len(question) > 100 else question,
                    'error': f'Could not parse prediction: {prediction_result}',
                    'ground_truth': ground_truth,
                    'prediction': None,
                    'brier_score': None,
                    'log_loss': None,
                    'reasoning': reasoning[:200] + '...' if len(reasoning) > 200 else reasoning
                }
            
            # Calculate metrics
            brier_score = self._calculate_brier_score(prediction, ground_truth)
            log_loss = self._calculate_log_loss(prediction, ground_truth)
            
            return {
                'index': index,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'error': None,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'brier_score': brier_score,
                'log_loss': log_loss,
                'reasoning': reasoning[:200] + '...' if len(reasoning) > 200 else reasoning
            }
        
        except Exception as e:
            return {
                'index': index,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'error': f'Agent error: {str(e)}',
                'ground_truth': ground_truth,
                'prediction': None,
                'brier_score': None,
                'log_loss': None
            }

    def evaluate_batch(self, num_samples: int = None) -> List[Dict[str, Any]]:
        """
        Evaluate the agent on multiple dataset items.
        
        Args:
            num_samples: Number of samples to evaluate. If None, evaluate all.
            
        Returns:
            List of evaluation results
        """
        if not self.dataset:
            print("Error: No dataset loaded")
            return []
        
        samples_to_evaluate = self.dataset[:num_samples] if num_samples else self.dataset
        self.results = []
        
        print(f"Evaluating {len(samples_to_evaluate)} samples...")
        
        for i, item in enumerate(samples_to_evaluate):
            print(f"Progress: {i+1}/{len(samples_to_evaluate)}", end='\r')
            result = self.evaluate_single(item, i)
            self.results.append(result)
        
        print(f"\nEvaluation complete!")
        return self.results

    def calculate_aggregate_metrics(self) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all evaluation results.
        
        Returns:
            Dictionary containing aggregate metrics
        """
        if not self.results:
            return {
                'total_evaluated': 0,
                'successful': 0,
                'failed': 0,
                'average_brier_score': None,
                'average_log_loss': None
            }
        
        brier_scores = [r['brier_score'] for r in self.results if r['brier_score'] is not None]
        log_losses = [r['log_loss'] for r in self.results if r['log_loss'] is not None]
        
        return {
            'total_evaluated': len(self.results),
            'successful': len([r for r in self.results if r['error'] is None]),
            'failed': len([r for r in self.results if r['error'] is not None]),
            'average_brier_score': sum(brier_scores) / len(brier_scores) if brier_scores else None,
            'average_log_loss': sum(log_losses) / len(log_losses) if log_losses else None,
            'min_brier_score': min(brier_scores) if brier_scores else None,
            'max_brier_score': max(brier_scores) if brier_scores else None,
            'min_log_loss': min(log_losses) if log_losses else None,
            'max_log_loss': max(log_losses) if log_losses else None
        }

    def save_results(self, output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        output = {
            'timestamp': datetime.now().isoformat(),
            'aggregate_metrics': self.calculate_aggregate_metrics(),
            'individual_results': self.results
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def print_summary(self) -> None:
        """Print a summary of the evaluation results."""
        metrics = self.calculate_aggregate_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Samples Evaluated: {metrics['total_evaluated']}")
        print(f"Successful: {metrics['successful']}")
        print(f"Failed: {metrics['failed']}")
        
        if metrics['average_brier_score'] is not None:
            print(f"\nBrier Score (lower is better):")
            print(f"  Average: {metrics['average_brier_score']:.6f}")
            print(f"  Min: {metrics['min_brier_score']:.6f}")
            print(f"  Max: {metrics['max_brier_score']:.6f}")
        
        if metrics['average_log_loss'] is not None:
            print(f"\nLog Loss (lower is better):")
            print(f"  Average: {metrics['average_log_loss']:.6f}")
            print(f"  Min: {metrics['min_log_loss']:.6f}")
            print(f"  Max: {metrics['max_log_loss']:.6f}")
        
        print("="*60 + "\n")


def main():
    """Main function to run the evaluation."""
    # Configure paths
    current_dir = Path(__file__).parent
    dataset_path = current_dir / "dataset_L1.json"
    output_path = current_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = ProphetsEvaluator(str(dataset_path))
    
    # Evaluate a small batch for testing (set to None to evaluate all)
    num_samples = 5  # Change this to evaluate more samples
    results = evaluator.evaluate_batch(num_samples=num_samples)
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results(str(output_path))
    
    # Print detailed results for first few samples
    print("\nDetailed Results (first 3 samples):")
    print("-" * 60)
    for result in results[:3]:
        print(f"\nSample {result['index']}:")
        print(f"  Question: {result['question']}")
        print(f"  Ground Truth: {result['ground_truth']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Brier Score: {result['brier_score']}")
        print(f"  Log Loss: {result['log_loss']}")
        if result['error']:
            print(f"  Error: {result['error']}")


if __name__ == "__main__":
    main()
