"""Deterministic Mock LLM Engine for Offline Testing and CI."""

import json
from typing import Dict, List, Optional
from agrag.state import EvaluationVerdict, SourceType


class MockLLMEngine:
    """Simulates realistic responses for each agent node in the pipeline."""

    @staticmethod
    def rewrite_query(
        user_query: str,
        messages_history: List[Dict[str, str]],
        critique_history: List[str],
    ) -> str:
        q_lower = user_query.lower()
        if critique_history:
            last_critique = critique_history[-1]
            return f"{user_query} [Refocused query addressing critique: {last_critique[:40]}]"

        # Coreference resolution simulation
        if "it" in q_lower or "its" in q_lower or "policy" in q_lower:
            return f"What are Acme Cloud Corp's requirements regarding {user_query}?"
        return user_query

    @staticmethod
    def classify_info_need(query: str) -> bool:
        q_lower = query.lower().strip()
        greetings = {"hi", "hello", "hey", "good morning", "good evening", "howdy"}
        if q_lower in greetings or any(q_lower.startswith(g) for g in greetings):
            return False
        if "2+2" in q_lower or "capital of france" in q_lower:
            return False
        return True

    @staticmethod
    def route_sources(query: str) -> List[SourceType]:
        q_lower = query.lower()
        sources: List[SourceType] = []
        if any(w in q_lower for w in ["alex", "sarah", "david", "employee", "account", "status", "ticket"]):
            sources.append("tool_api")
        if any(w in q_lower for w in ["policy", "mfa", "timeout", "vpn", "travel", "storage", "handbook", "acme"]):
            sources.append("vector_db")
        if any(w in q_lower for w in ["industry", "csa", "nist", "news", "external", "standards"]):
            sources.append("web_search")

        # Fallback if no keyword matched
        if not sources:
            sources = ["vector_db", "tool_api"]
        return sources


    @staticmethod
    def generate_response(
        query: str,
        fused_context: Optional[str],
        model_name: str = "deepseek-chat",
    ) -> str:
        if not fused_context:
            return (
                f"Hello! I am your Acme Cloud Corp AI assistant. How can I assist you with "
                f"our IT policies, tools, or operations today?"
            )

        # Grounded response generation with citations
        lines = [
            f"Based on Acme Cloud Corp's corporate documentation and operational records, here is the information for your query:\n"
        ]

        if "alex" in fused_context.lower():
            lines.append(
                "- **Employee Status (Alex Mercer)**: Active Senior Cloud Architect in Engineering. "
                "MFA is enabled using a FIDO2 hardware key, last rotated on 2026-08-01 (Compliant). [Source: Employee Record: Alex Mercer (alex)]"
            )
        if "sarah" in fused_context.lower():
            lines.append(
                "- **Employee Status (Sarah Connor)**: Active DevOps Lead. "
                "MFA last rotated on 2026-05-10 (> 90 days, Non-compliant). [Source: Employee Record: Sarah Connor (sarah)]"
            )
        if "mfa" in fused_context.lower():
            lines.append(
                "- **MFA Policy**: All staff must rotate MFA credentials at least every 90 days. "
                "Approved methods: FIDO2 hardware keys and TOTP apps. SMS verification is strictly forbidden. "
                "[Source: Acme IT Security Policy - MFA Requirements (Sec. 4.2)]"
            )
        if "session" in fused_context.lower() or "timeout" in fused_context.lower():
            lines.append(
                "- **Session Lockout**: Sessions must automatically lock after 15 minutes of inactivity. "
                "Passwords require 16+ characters with at least 3 character types. [Source: Acme IT Security Policy - Session Timeout (Sec. 4.5)]"
            )
        if "wireguard" in fused_context.lower() or "vpn" in fused_context.lower():
            lines.append(
                "- **Remote Access**: Must use Acme WireGuard VPN. Split-tunneling is prohibited. "
                "[Source: Acme Remote Work & Travel Security Handbook (Sec. 2.1)]"
            )

        if len(lines) == 1:
            # Context was present but specific mock keywords didn't match
            lines.append(
                f"According to the provided sources:\n{fused_context[:300]}..."
            )

        return "\n".join(lines)

    @staticmethod
    def evaluate_response(
        query: str,
        response: str,
        fused_context: Optional[str],
    ) -> EvaluationVerdict:
        # Check if the query was asking for non-existent proprietary info
        q_lower = query.lower()
        if "2029 quantum roadmap" in q_lower or "secret missile plans" in q_lower:
            return EvaluationVerdict(
                is_valid=False,
                failure_type="missing_info",
                critique="The requested information does not exist in any internal or external knowledge source.",
            )

        # Check for simulated hallucination test keyword
        if "simulate_hallucination" in q_lower:
            return EvaluationVerdict(
                is_valid=False,
                failure_type="hallucination",
                critique="The response claimed MFA rotation is annual (365 days), but policy states 90 days.",
            )

        # Default: Valid response
        return EvaluationVerdict(
            is_valid=True,
            failure_type="none",
            critique=None,
        )


mock_llm = MockLLMEngine()
