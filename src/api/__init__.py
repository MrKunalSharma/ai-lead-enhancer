"""
API Module for AI Lead Enhancer

This module provides FastAPI endpoints for lead scoring and enrichment services.
"""

from .main import app, get_lead_scorer, get_enricher

__all__ = [
    "app",
    "get_lead_scorer",
    "get_enricher",
]

# API Version
API_VERSION = "v1"
API_TITLE = "AI Lead Enhancer API"
API_DESCRIPTION = """
## AI-Powered Lead Scoring and Enrichment API

This API provides endpoints for:
- Lead scoring using machine learning
- Data enrichment from multiple sources
- Batch processing of leads
- Real-time lead qualification

### Features:
- 🚀 Fast response times (<200ms)
- 🔒 Secure data handling
- 📊 Detailed scoring breakdown
- 🔄 Batch processing support
"""

# API Tags for documentation
TAGS_METADATA = [
    {
        "name": "leads",
        "description": "Operations related to lead processing",
    },
    {
        "name": "scoring",
        "description": "Lead scoring endpoints",
    },
    {
        "name": "enrichment",
        "description": "Data enrichment endpoints",
    },
    {
        "name": "health",
        "description": "API health and status checks",
    },
]

# Response examples for documentation
RESPONSE_EXAMPLES = {
    "lead_score": {
        "200": {
            "description": "Successful lead scoring",
            "content": {
                "application/json": {
                    "example": {
                        "lead_id": "12345",
                        "score": 85,
                        "grade": "A",
                        "breakdown": {
                            "company_size": 22,
                            "industry_fit": 18,
                            "technology_stack": 17,
                            "engagement_signals": 13,
                            "growth_indicators": 15
                        },
                        "recommendations": [
                            "High-value prospect - prioritize outreach",
                            "Decision maker identified",
                            "Recent funding indicates budget availability"
                        ]
                    }
                }
            }
        }
    }
}