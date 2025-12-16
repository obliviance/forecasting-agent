import requests
import json
import os
import dotenv

import agent_functions
from base_model import BaseModel

# Load API key from .env file
dotenv.load_dotenv()

def agentcast(question: str) -> dict[str, str]:
    """Main function to run the forecasting agent."""
    model = BaseModel(model="gpt-4o-mini", question=question)

    # Step 1: Decompose the question
    sub_questions = agent_functions.decompose_question(question)

    # Step 2: Source data for each sub-question
    sourced_data = agent_functions.source_data(sub_questions)

    # Step 3: Get predictions from max and min agents
    max_prediction, max_reasoning = agent_functions.max_agent(question, sourced_data)
    min_prediction, min_reasoning = agent_functions.min_agent(question, sourced_data)

    refined_prediction, refined_reasoning = agent_functions.refine_prediction(model, min_prediction, min_reasoning, max_prediction, max_reasoning)

    return refined_prediction, refined_reasoning






