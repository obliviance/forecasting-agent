import json

from .base_model import BaseModel

def decompose_question(question: str) -> list[str]:
    """Decomposes a complex forecasting question into simpler sub-questions."""
    # Placeholder implementation; in practice, this would use NLP techniques
    sub_questions = [
        f"What are the key factors influencing the outcome of: {question}?",
        f"What historical data is relevant to: {question}?",
        f"What are the potential scenarios for: {question}?"
    ]
    return sub_questions

def source_data(sub_questions: list[str]) -> dict[str, str]:
    """Sources data for each sub-question from external APIs or databases."""
    # Placeholder implementation; in practice, this would involve API calls or database queries
    sourced_data = {}
    for sq in sub_questions:
        sourced_data[sq] = f"Sourced data relevant to: {sq}"
    return sourced_data

# Simulates an agent that predicts a high probability for the event
# Reads source data and formulates a prediction and reasoning for the highest possible probability for the event
def max_agent(model: BaseModel, sourced_data: dict[str, str]) -> tuple[float, str]:
    """Generates a prediction and reasoning based on sourced data."""
    user_prompt = (f"Based on the following sourced data:\n"
                     f"{sourced_data}\n"
                        f"Provide a prediction and reasoning that supports the highest possible probability for the event.\n"
                        f"Return the prediction as a float between 0 and 1, and the reasoning as text.\n"
                        f"Format your response as JSON with keys 'prediction' and 'reasoning'.")
    response = model.forecast(user_prompt)
    answer = json.dumps(response)
    try:
        answer_json = json.loads(answer)
        prediction = float(answer_json["prediction"])
        reasoning = str(answer_json["reasoning"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse max agent response: {answer}") from e

    return prediction, reasoning

def min_agent(model: BaseModel, sourced_data: dict[str, str]) -> tuple[float, str]:
    """Generates a prediction and reasoning based on sourced data."""
    user_prompt = (f"Based on the following sourced data:\n"
                     f"{sourced_data}\n"
                        f"Provide a prediction and reasoning that supports the lowest possible probability for the event.\n"
                        f"Return the prediction as a float between 0 and 1, and the reasoning as text.\n"
                        f"Format your response as JSON with keys 'prediction' and 'reasoning'.")
    response = model.forecast(user_prompt)
    answer = json.dumps(response)
    try:
        answer_json = json.loads(answer)
        prediction = float(answer_json["prediction"])
        reasoning = str(answer_json["reasoning"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse max agent response: {answer}") from e

    return prediction, reasoning

def refine_prediction(model: BaseModel, min_prediction: float, min_reasoning: str, max_prediction: float, max_reasoning: str) -> tuple[float, str]:
    """Refines the final prediction based on min and max agent outputs."""
    # Simple average for placeholder implementation
    user_prompt = (f"Based on the following predictions and reasonings:\n"
                    f"Max Agent Prediction: {max_prediction}\n"
                    f"Max Agent Reasoning: {max_reasoning}\n"   
                    f"Min Agent Prediction: {min_prediction}\n"
                    f"Min Agent Reasoning: {min_reasoning}\n"
                    f"Provide a refined final prediction as a probability between 0 and 1.\n"
                    f"Additionally, provide a clear explanation of your reasoning behind this refined prediction.\n"
                    f"Format your response as JSON with keys 'prediction' and 'reasoning'.")
    response = model.forecast(user_prompt)
    answer = json.dumps(response)
    try:
        answer_json = json.loads(answer)
        refined_prediction = float(answer_json["prediction"])
        refined_reasoning = str(answer_json["reasoning"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse refine prediction response: {answer}") from e
    
    return refine_prediction, refined_reasoning




