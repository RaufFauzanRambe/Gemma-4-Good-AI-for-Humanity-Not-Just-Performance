"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/models/gemma_model.py
Gemma Model Wrapper & Prompt Engineering

Provides a wrapper for interacting with the Gemma model via z-ai-web-dev-sdk,
with built-in prompt engineering strategies optimised for humanity-aligned outputs.
"""

import json
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt strategies (determined from gemma_prompt_test.ipynb experiments)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "baseline": {
        "system_prompt": "You are a helpful AI assistant.",
        "user_template": "{prompt}",
        "description": "Direct prompt — no special instructions.",
    },
    "empathy_plus": {
        "system_prompt": (
            "You are a compassionate AI assistant focused on human welfare. "
            "Always consider the human impact of your responses. Show empathy, "
            "cultural sensitivity, and prioritise inclusivity. Acknowledge "
            "vulnerable populations and promote equitable solutions."
        ),
        "user_template": (
            "{prompt}\n\n"
            "Please provide a response that is empathetic, inclusive, and "
            "considers the diverse needs of all people, especially "
            "marginalised communities."
        ),
        "description": "Empathy-focused system prompt + user instruction.",
    },
    "structured": {
        "system_prompt": (
            "You are an AI assistant that provides well-organised, structured "
            "responses. Use numbered lists, clear headings, and concise "
            "explanations."
        ),
        "user_template": (
            "{prompt}\n\n"
            "Please structure your response with:\n"
            "1. Brief context / definition\n"
            "2. Key points (numbered list)\n"
            "3. Considerations for vulnerable populations\n"
            "4. Actionable recommendation"
        ),
        "description": "Structured output with mandatory sections.",
    },
    "role_based": {
        "system_prompt": (
            "You are a humanitarian AI advisor with 20 years of experience in "
            "international development, public health, and ethical technology. "
            "You approach every question with compassion, evidence-based "
            "reasoning, and a focus on the most vulnerable communities."
        ),
        "user_template": (
            "As a humanitarian AI advisor, please address the following:\n\n"
            "{prompt}\n\n"
            "Consider real-world impact on communities, practical feasibility, "
            "and equity in your response."
        ),
        "description": "Expert persona assignment.",
    },
    "chain_of_thought": {
        "system_prompt": (
            "You are an AI assistant that thinks deeply about human impact. "
            "Always reason step-by-step about who is affected, potential harms, "
            "and how to maximise benefit for humanity."
        ),
        "user_template": (
            "{prompt}\n\n"
            "Please reason step-by-step:\n"
            "Step 1: Who are the stakeholders affected?\n"
            "Step 2: What are the potential benefits?\n"
            "Step 3: What are the potential risks or harms?\n"
            "Step 4: How can we maximise benefit and minimise harm?\n"
            "Step 5: Provide your final thoughtful response."
        ),
        "description": "Step-by-step reasoning about human impact.",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GemmaResponse:
    """Container for a single model response."""
    prompt: str
    response: str
    strategy: str
    response_time_ms: float = 0.0
    token_count: int = 0
    model_name: str = "gemma-4"


@dataclass
class PromptConfig:
    """Configuration for prompt engineering."""
    strategy: str = "role_based"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gemma model wrapper
# ---------------------------------------------------------------------------
class GemmaModel:
    """High-level wrapper for the Gemma model.

    Handles prompt formatting, API calls, and response parsing.
    Can operate in **live mode** (z-ai-web-dev-sdk) or **offline mode**
    (returns placeholder responses for testing).

    Example
    -------
    >>> model = GemmaModel(strategy="role_based")
    >>> response = model.generate("How can AI help improve education?")
    >>> print(response.response)
    """

    def __init__(
        self,
        strategy: str = "role_based",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        offline: bool = True,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES.keys())}")
        self.strategy = strategy
        self.config = PromptConfig(
            strategy=strategy,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=STRATEGIES[strategy]["system_prompt"],
            user_template=STRATEGIES[strategy]["user_template"],
        )
        self.offline = offline
        self._call_count = 0
        logger.info("GemmaModel initialised — strategy='%s', offline=%s", strategy, offline)

    def generate(self, prompt: str, temperature: Optional[float] = None) -> GemmaResponse:
        """Generate a response for a single prompt.

        Parameters
        ----------
        prompt : str
            The user prompt.
        temperature : float, optional
            Override the default temperature.

        Returns
        -------
        GemmaResponse
        """
        formatted = self._format_prompt(prompt)
        temp = temperature if temperature is not None else self.config.temperature

        if self.offline:
            response_text = self._offline_response(prompt)
            elapsed = 0.0
        else:
            response_text, elapsed = self._api_call(formatted, temp)

        self._call_count += 1
        return GemmaResponse(
            prompt=prompt,
            response=response_text,
            strategy=self.strategy,
            response_time_ms=elapsed,
            token_count=len(response_text.split()),
        )

    def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
    ) -> List[GemmaResponse]:
        """Generate responses for multiple prompts.

        Returns
        -------
        list[GemmaResponse]
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info("Generating response %d/%d (strategy=%s)", i + 1, len(prompts), self.strategy)
            results.append(self.generate(prompt, temperature))
        return results

    def format_prompt(self, prompt: str) -> str:
        """Public access to the formatted prompt (without calling the model)."""
        return self._format_prompt(prompt)

    @staticmethod
    def list_strategies() -> Dict[str, str]:
        """Return all available strategies and their descriptions."""
        return {name: info["description"] for name, info in STRATEGIES.items()}

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _format_prompt(self, prompt: str) -> str:
        """Apply the strategy template to the user prompt."""
        return self.config.user_template.format(prompt=prompt)

    def _api_call(self, formatted_prompt: str, temperature: float) -> Tuple[str, float]:
        """Call the z-ai-web-dev-sdk API.

        This method runs server-side (backend only). Do NOT call from client-side code.
        """
        try:
            import ZAI from 'z-ai-web-dev-sdk'  # type: ignore
        except ImportError:
            logger.error("z-ai-web-dev-sdk not available. Use offline=True.")
            return self._offline_response(formatted_prompt), 0.0

        start = time.time()
        try:
            zai = await ZAI.create()
            completion = await zai.chat.completions.create({
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": formatted_prompt},
                ],
                "temperature": temperature,
                "max_tokens": self.config.max_tokens,
            })
            text = completion.choices[0].message.content
            elapsed = (time.time() - start) * 1000
            return text, elapsed
        except Exception as e:
            logger.error("API call failed: %s", e)
            return self._offline_response(formatted_prompt), (time.time() - start) * 1000

    @staticmethod
    def _offline_response(prompt: str) -> str:
        """Return a placeholder response for offline / testing mode."""
        return (
            f"[Offline placeholder response for strategy testing]\n\n"
            f"Original prompt: {prompt[:200]}...\n\n"
            f"To generate real responses, set offline=False and ensure "
            f"z-ai-web-dev-sdk is available in the backend environment."
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = GemmaModel(strategy="role_based", offline=True)

    print("=" * 60)
    print("GEMMA MODEL — STRATEGY TEST")
    print("=" * 60)

    test_prompts = [
        "How can AI help improve access to clean water?",
        "What should teachers know about using AI in diverse classrooms?",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt:   {prompt}")
        resp = model.generate(prompt)
        print(f"Strategy: {resp.strategy}")
        print(f"Response: {resp.response[:150]}...")
        print(f"Tokens:   {resp.token_count}")

    print(f"\n\nAvailable strategies:")
    for name, desc in GemmaModel.list_strategies().items():
        print(f"  {name:20s} — {desc}")
