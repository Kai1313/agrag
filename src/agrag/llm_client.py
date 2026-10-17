"""Unified LLM client interface for DeepSeek API with automatic mock fallback."""

import json
import logging
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI
from agrag.config import settings
from agrag.mock_llm import mock_llm

logger = logging.getLogger(__name__)


class LLMClient:
    """Handles communication with DeepSeek API, with zero-credential mock fallback."""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        if not settings.is_mock and settings.deepseek_api_key:
            self._client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )

    @property
    def is_mock(self) -> bool:
        return self._client is None or settings.is_mock

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: int = 2048,
    ) -> str:
        """Call DeepSeek API or route through mock engine if mock mode is active."""
        target_model = model or settings.deepseek_default_model

        if self.is_mock:
            return self._mock_completion(messages, target_model, response_format)

        try:
            kwargs: Dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            resp = await self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}. Falling back to mock engine.")
            return self._mock_completion(messages, target_model, response_format)

    def _mock_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        response_format: Optional[Dict[str, str]],
    ) -> str:
        """Simulate realistic responses for prompts sent to micro-agent nodes."""
        system_prompt = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        last_user_message = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # 1. Query Rewriter prompt simulation
        if "query rewriter" in system_prompt.lower():
            rewritten = mock_llm.rewrite_query(last_user_message, [], [])
            if response_format and response_format.get("type") == "json_object":
                return json.dumps({"updated_query": rewritten})
            return rewritten

        # 2. Info Need Classifier prompt simulation
        if "retrieval need" in system_prompt.lower() or "classifier" in system_prompt.lower():
            need = mock_llm.classify_info_need(last_user_message)
            if response_format and response_format.get("type") == "json_object":
                return json.dumps({"needs_retrieval": need, "reasoning": "Determined by mock classifier"})
            return json.dumps({"needs_retrieval": need})

        # 3. Source Selector prompt simulation
        if "source selector" in system_prompt.lower() or "route" in system_prompt.lower():
            sources = mock_llm.route_sources(last_user_message)
            tool_calls = []
            if "tool_api" in sources:
                if any(w in last_user_message.lower() for w in ["sarah", "sarah connor"]):
                    tool_calls.append({"tool_name": "get_employee_account_status", "arguments": {"username": "sarah"}})
                elif any(w in last_user_message.lower() for w in ["alex", "alex mercer"]):
                    tool_calls.append({"tool_name": "get_employee_account_status", "arguments": {"username": "alex"}})
                else:
                    tool_calls.append({"tool_name": "get_employee_account_status", "arguments": {"username": "alex"}})
            
            return json.dumps({
                "selected_sources": sources,
                "tool_calls": tool_calls,
                "reasoning": f"Routed to {sources} based on query keywords"
            })

        # 4. Response Evaluator prompt simulation
        if "evaluator" in system_prompt.lower():
            # Check context from user message
            verdict = mock_llm.evaluate_response(last_user_message, "Response draft", None)
            return json.dumps({
                "is_valid": verdict.is_valid,
                "failure_type": verdict.failure_type,
                "critique": verdict.critique,
            })

        # 5. Generator prompt simulation
        return mock_llm.generate_response(last_user_message, last_user_message, model)


llm_client = LLMClient()
