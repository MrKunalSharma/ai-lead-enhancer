"""
UI Module for AI Lead Enhancer

This module contains the Streamlit-based user interface for the lead enhancement tool.
"""

from .streamlit_app import main, render_sidebar, render_main_content

__all__ = [
    "main",
    "render_sidebar",
    "render_main_content",
]

# UI Configuration
UI_CONFIG = {
    "theme": {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730",
        "font": "sans serif"
    },
    "layout": {
        "wide_mode": True,
        "sidebar_state": "expanded",
        "show_footer": True
    },
    "features": {
        "single_lead": True,
        "batch_upload": True,
        "real_time_scoring": True,
        "export_options": ["CSV", "JSON", "Excel"],
        "visualization": True
    },
    "branding": {
        "app_name": "AI Lead Enhancer",
        "tagline": "Transform Raw Leads into Sales Intelligence",
        "logo_url": None,  # Add logo URL if available
        "company": "Built for Caprae Capital"
    }
}

# Page configuration
PAGE_CONFIG = {
    "page_title": "AI Lead Enhancer - Caprae Capital",
    "page_icon": "🚀",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "Get Help": "https://github.com/yourusername/ai-lead-enhancer",
        "Report a bug": "https://github.com/yourusername/ai-lead-enhancer/issues",
        "About": "AI-powered lead scoring and enrichment tool built for Caprae Capital's AI-Readiness Challenge"
    }
}

# Dashboard metrics configuration
METRICS_CONFIG = {
    "overview": [
        {
            "label": "Total Leads Processed",
            "key": "total_processed",
            "format": "{:,}",
            "delta_key": "daily_change"
        },
        {
            "label": "Average Score",
            "key": "avg_score",
            "format": "{:.1f}",
            "suffix": "/100"
        },
        {
            "label": "Enrichment Success Rate",
            "key": "enrichment_rate",
            "format": "{:.1%}",
            "delta_key": "rate_change"
        },
        {
            "label": "Processing Time",
            "key": "avg_time",
            "format": "{:.2f}",
            "suffix": " sec"
        }
    ],
    "charts": {
        "score_distribution": {
            "type": "histogram",
            "title": "Lead Score Distribution",
            "x_label": "Score",
            "y_label": "Count"
        },
        "industry_breakdown": {
            "type": "pie",
            "title": "Leads by Industry",
            "show_legend": True
        },
        "company_size": {
            "type": "bar",
            "title": "Company Size Distribution",
            "x_label": "Size Category",
            "y_label": "Count"
        },
        "daily_trend": {
            "type": "line",
            "title": "Daily Processing Trend",
            "x_label": "Date",
            "y_label": "Leads Processed"
        }
    }
}

# Form field configuration
FORM_FIELDS = {
    "required": [
        {
            "key": "company_name",
            "label": "Company Name",
            "type": "text",
            "placeholder": "Enter company name",
            "help": "Legal name of the company"
        }
    ],
    "optional": [
        {
            "key": "website",
            "label": "Website URL",
            "type": "text",
            "placeholder": "https://example.com",
            "help": "Company website for enrichment"
        },
        {
            "key": "employee_count",
            "label": "Employee Count",
            "type": "number",
            "min_value": 1,
            "max_value": 1000000,
            "help": "Approximate number of employees"
        },
        {
            "key": "industry",
            "label": "Industry",
            "type": "selectbox",
            "options": [
                "Technology",
                "Finance",
                "Healthcare",
                "E-commerce",
                "Manufacturing",
                "Retail",
                "Education",
                "Other"
            ],
            "help": "Primary industry sector"
        },
        {
            "key": "location",
            "label": "Location",
            "type": "text",
            "placeholder": "City, State/Country",
            "help": "Company headquarters location"
        },
        {
            "key": "email",
            "label": "Contact Email",
            "type": "text",
            "placeholder": "contact@example.com",
            "help": "Primary contact email"
        },
        {
            "key": "phone",
            "label": "Phone Number",
            "type": "text",
            "placeholder": "+1 (555) 123-4567",
            "help": "Primary contact phone"
        }
    ]
}

# Styling configuration
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        padding: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Score badge styling */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .score-a { background-color: #28a745; color: white; }
    .score-b { background-color: #17a2b8; color: white; }
    .score-c { background-color: #ffc107; color: black; }
    .score-d { background-color: #fd7e14; color: white; }
    .score-f { background-color: #dc3545; color: white; }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1557a0;
    }
    
    /* Success/Error messages */
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
        margin: 1rem 0;
    }
    
    .error-message {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
        margin: 1rem 0;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    .dataframe th {
        background-color: #f0f2f6;
        font-weight: 600;
        text-align: left;
        padding: 0.75rem;
    }
    
    .dataframe td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
"""

# Help text and tooltips
HELP_TEXT = {
    "lead_scoring": """
    **Lead Scoring** uses machine learning to evaluate leads based on:
    - Company size and growth indicators
    - Industry fit with your target market
    - Technology stack compatibility
    - Engagement signals and intent data
    - Market timing and opportunity size
    
    Scores range from 0-100, with letter grades:
    - **A (80-100)**: Hot leads - immediate action recommended
    - **B (65-79)**: Qualified leads - high potential
    - **C (50-64)**: Nurture leads - needs development
    - **D (35-49)**: Low priority - minimal engagement
    - **F (0-34)**: Poor fit - not recommended
    """,
    
    "enrichment": """
    **Data Enrichment** automatically gathers additional information:
    - Company details from website scraping
    - Technology stack detection
    - Social media profiles
    - Contact information extraction
    - Industry classification
    - Growth indicators and funding data
    
    Data is gathered from multiple sources and validated for accuracy.
    """,
    
    "batch_processing": """
    **Batch Processing** allows you to:
    - Upload CSV files with multiple leads
    - Process up to 1000 leads at once
    - Export enriched data in multiple formats
    - Track processing progress in real-time
    
    Required CSV columns: company_name
    Optional columns: website, employee_count, industry, location
    """
}