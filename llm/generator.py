"""
LLM Answer Generation Module

Generates answers to user questions using Azure OpenAI, based on
retrieved context from the hybrid search pipeline.

All LLM parameters (temperature, max_tokens, prompts) are driven by
config.yaml → llm section. Token usage and latency are tracked for
LLMOps monitoring (Step 2 logging + Step 9 Prometheus metrics).
"""

import os
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

# [Step 1] Import config loader — temperature, max_tokens, prompts from config.yaml
from config.settings import get as cfg

# [Step 2] Import structured logger and LLM call tracking
from utils.logger import get_logger, log_llm_call, LOG_PROMPTS

# [Step 9] Import Prometheus metrics recorder for token/latency tracking
from monitoring.prometheus_metrics import record_llm_call

load_dotenv()

logger = get_logger(__name__)

# [Step 1] Load LLM config section (api_version, temperature, max_tokens, prompts)
_llm = cfg("llm")

# [Step 1] api_version now comes from config.yaml instead of being hardcoded
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=_llm["api_version"],
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def generate_answer(query, context):
    """
    Generate an answer using Azure OpenAI based on the provided context.

    Args:
        query: The user's question
        context: Relevant context from retrieved documents

    Returns:
        Generated answer as a string
    """
    # [Step 1] Use prompt template from config.yaml instead of hardcoded prompt
    user_prompt = _llm["user_prompt_template"].format(context=context, query=query)

    # [Step 2] Optionally log the full prompt for debugging (controlled by config)
    if LOG_PROMPTS:
        logger.debug("Generation prompt: %s", user_prompt[:500])

    # [Step 2] Measure LLM call latency
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=[
            # [Step 1] System prompt from config.yaml
            {"role": "system", "content": _llm["system_prompt"]},
            {"role": "user", "content": user_prompt}
        ],
        # [Step 1] temperature and max_tokens from config.yaml
        temperature=_llm["temperature"],
        max_tokens=_llm["max_tokens"]
    )
    latency = time.perf_counter() - start

    # [Step 2] Log token usage and latency via structured logger
    log_llm_call(logger, response, call_type="generation", latency_s=latency)

    # [Step 9] Record Prometheus metrics for monitoring dashboard
    record_llm_call("generation", response, latency)

    return response.choices[0].message.content
