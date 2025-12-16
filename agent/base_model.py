import os
import json
from dataclasses import dataclass
from typing import Optional
from datetime import date
import httpx

from cutoffs import get_cutoff_date

class BaseModel:
	"""
	Minimal OpenRouter base model caller.
	"""

	def __init__(self, model: str, question: str, api_key: Optional[str] = None, referer: Optional[str] = None, title: Optional[str] = None):
		self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
		self.referer = referer
		self.title = title
		self.model = model
		self.question = question
		self.temperature = 0.2
		
		if not self.api_key:
			raise RuntimeError(
				"Missing OPENROUTER_API_KEY. Set it in your environment to use BaseModel."
			)

		self._client = httpx.Client(
			base_url="https://openrouter.ai/api/v1",
			timeout=30.0,
		)

	def forecast(self, user_prompt: str) -> str:
		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}
		if self.referer:
			headers["HTTP-Referer"] = self.referer
		if self.title:
			headers["X-Title"] = self.title

		system_prompt = (
			f"""You are an expert superforecaster with experience providing calibrated probabilistic
				forecasts under uncertainty, with your performance evaluated according to the Brier score. When
				forecasting, do not treat 0.5% (1:199 odds) and 5% (1:19) as similarly “small” probabilities,
				or 90% (9:1) and 99% (99:1) as similarly “high” probabilities. As the odds show, they are
				markedly different, so output your probabilities accordingly.
				Question:
				{self.question}
				Today’s date: {date.today().strftime("%B %d, %Y")}
				Your pretraining knowledge cutoff: {get_cutoff_date(self.model)}"""
		)

		payload = {
			"model": self.model,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			"temperature": self.temperature,
		}

		resp = self._client.post("/chat/completions", headers=headers, json=payload)
		resp.raise_for_status()
		data = resp.json()

		try:
			content = data["choices"][0]["message"]["content"]
		except (KeyError, IndexError) as e:
			raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from e
		return content

