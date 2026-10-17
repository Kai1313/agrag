"""Operational tools and API implementations for Acme Cloud Corp."""

import json
from typing import Any, Dict, List, Optional
from agrag.state import RetrievedChunk
from agrag.tools.base import BaseTool


class EmployeeAccountStatusTool(BaseTool):
    """Tool to query live employee directory and security posture records."""

    name = "get_employee_account_status"
    description = (
        "Retrieve live employee directory record including MFA status, last MFA rotation date, "
        "department, and active account flags."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "description": "The corporate username of the employee (e.g. 'alex', 'sarah', 'david')",
            }
        },
        "required": ["username"],
    }

    # In-memory mock database of Acme Cloud Corp employees
    _DATABASE = {
        "alex": {
            "username": "alex",
            "full_name": "Alex Mercer",
            "department": "Engineering",
            "role": "Senior Cloud Architect",
            "status": "active",
            "mfa_enabled": True,
            "mfa_type": "FIDO2 / Hardware Security Key",
            "mfa_last_rotated": "2026-08-01",
            "mfa_compliant": True,
            "vpn_access": True,
        },
        "sarah": {
            "username": "sarah",
            "full_name": "Sarah Connor",
            "department": "Operations",
            "role": "DevOps Lead",
            "status": "active",
            "mfa_enabled": True,
            "mfa_type": "Authenticator App (TOTP)",
            "mfa_last_rotated": "2026-05-10",
            "mfa_compliant": False,  # Over 90 days policy!
            "vpn_access": True,
        },
        "david": {
            "username": "david",
            "full_name": "David Miller",
            "department": "Finance",
            "role": "Financial Analyst",
            "status": "active",
            "mfa_enabled": True,
            "mfa_type": "SMS (Deprecated)",
            "mfa_last_rotated": "2026-07-15",
            "mfa_compliant": False,  # SMS is disallowed
            "vpn_access": False,
        },
    }

    async def execute(self, **kwargs) -> RetrievedChunk:
        username = kwargs.get("username", "").lower().strip()
        record = self._DATABASE.get(username)

        if record:
            content = json.dumps(record, indent=2)
            title = f"Employee Record: {record['full_name']} ({username})"
        else:
            content = json.dumps(
                {"error": f"Employee '{username}' not found in Acme directory."},
                indent=2,
            )
            title = f"Employee Record Lookup Failed: {username}"

        return RetrievedChunk(
            source_type="tool_api",
            source_title=title,
            content=content,
            metadata={"tool_name": self.name, "username": username},
            score=1.0,
        )


class IncidentTicketLookupTool(BaseTool):
    """Tool to search recent IT security incident tickets in ServiceNow/Jira."""

    name = "lookup_incident_tickets"
    description = (
        "Search active and recent IT security tickets, incident post-mortems, and system outages."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to search incident tickets (e.g. 'vpn', 'mfa', 'outage')",
            }
        },
        "required": ["query"],
    }

    _TICKETS = [
        {
            "ticket_id": "SEC-8492",
            "title": "Automated MFA expiration warnings failing for EU region",
            "status": "in_progress",
            "priority": "P2",
            "assignee": "SecOps Team",
            "resolution_notes": "Cron job schedule fixed; manual rotation reminders dispatched.",
        },
        {
            "ticket_id": "IT-10294",
            "title": "VPN gateway cert renewal scheduled for Sunday 02:00 UTC",
            "status": "scheduled",
            "priority": "P3",
            "assignee": "NetOps Team",
            "resolution_notes": "Zero expected downtime; secondary tunnel active.",
        },
    ]

    async def execute(self, **kwargs) -> RetrievedChunk:
        search_query = kwargs.get("query", "").lower()
        matched = [
            t
            for t in self._TICKETS
            if search_query in t["title"].lower() or search_query in t["ticket_id"].lower()
        ]

        if not matched:
            matched = self._TICKETS[:2]  # Fallback to recent tickets

        return RetrievedChunk(
            source_type="tool_api",
            source_title="Acme Incident Tickets",
            content=json.dumps(matched, indent=2),
            metadata={"tool_name": self.name, "matched_count": len(matched)},
            score=0.9,
        )


class ToolRegistry:
    """Registry maintaining active tool instances."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {
            EmployeeAccountStatusTool.name: EmployeeAccountStatusTool(),
            IncidentTicketLookupTool.name: IncidentTicketLookupTool(),
        }

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_tools_description_prompt(self) -> str:
        """Generates a text description of available tools for LLM routing."""
        lines = []
        for t in self._tools.values():
            lines.append(f"- **{t.name}**: {t.description}")
            lines.append(f"  Parameters: {json.dumps(t.parameters_schema)}")
        return "\n".join(lines)


tool_registry = ToolRegistry()
