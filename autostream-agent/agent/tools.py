"""
tools.py
--------
Defines the mock lead capture tool used by the agent when a
high-intent user has provided all required information.
"""

import json
from datetime import datetime
from langchain_core.tools import tool


# Mock Lead Capture 

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API call that simulates capturing a lead in the CRM.
    In production, this would POST to a real CRM endpoint.
    """
    lead = {
        "status": "success",
        "lead_id": f"LEAD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.now().isoformat(),
    }

    # Simulated CRM response output
    print("\n" + "=" * 50)
    print("🎯  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 50)
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Lead ID  : {lead['lead_id']}")
    print(f"  Time     : {lead['captured_at']}")
    print("=" * 50 + "\n")

    return lead



# LangChain Tool Wrapper

@tool
def capture_lead_tool(name: str, email: str, platform: str) -> str:
    """
    Call this tool ONLY when you have collected the user's name, email,
    and creator platform. Triggers lead capture in the CRM system.
    
    Args:
        name: Full name of the lead
        email: Email address of the lead
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)
    """
    result = mock_lead_capture(name, email, platform)
    return json.dumps(result)
