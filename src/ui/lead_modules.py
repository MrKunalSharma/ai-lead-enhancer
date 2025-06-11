# src/ui/lead_modules.py
# Unified module containing all necessary components for the Streamlit app

import random
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Any, Optional

# Configuration
MODEL_CONFIG = {
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10
    },
    'neural_network': {
        'hidden_layers': [64, 32],
        'activation': 'relu'
    }
}

SCORING_CONFIG = {
    'weights': {
        'industry_score': 0.2,
        'company_size_score': 0.15,
        'engagement_score': 0.25,
        'budget_score': 0.2,
        'timing_score': 0.2
    },
    'thresholds': {
        'high': 80,
        'medium': 50,
        'low': 20
    }
}

# Lead Scorer Class
class LeadScorer:
    def __init__(self):
        self.weights = SCORING_CONFIG['weights']
        
    def score_lead(self, lead_data: Dict[str, Any]) -> float:
        """Score a lead based on various factors"""
        scores = {}
        
        # Industry score
        industry = lead_data.get('industry', 'Unknown')
        industry_scores = {
            'Technology': 90, 'Software': 85, 'AI/ML': 95,
            'Finance': 80, 'Healthcare': 75, 'Retail': 70,
            'Manufacturing': 65, 'Unknown': 50
        }
        scores['industry_score'] = industry_scores.get(industry, 50)
        
        # Company size score (if available)
        employee_count = lead_data.get('employee_count', 0)
        if employee_count > 1000:
            scores['company_size_score'] = 90
        elif employee_count > 100:
            scores['company_size_score'] = 70
        elif employee_count > 10:
            scores['company_size_score'] = 50
        else:
            scores['company_size_score'] = 30
            
        # Engagement score (simulated)
        scores['engagement_score'] = random.randint(40, 90)
        
        # Budget score (simulated)
        scores['budget_score'] = random.randint(30, 85)
        
        # Timing score (simulated)
        scores['timing_score'] = random.randint(50, 95)
        
        # Calculate weighted average
        total_score = sum(scores[key] * self.weights.get(key, 0) 
                         for key in scores)
        
        return min(100, max(0, total_score))

# Text Analyzer Class
class TextAnalyzer:
    def __init__(self):
        self.positive_words = ['excellent', 'great', 'innovative', 'leading', 
                              'successful', 'growing', 'expand', 'opportunity']
        self.negative_words = ['challenge', 'difficult', 'problem', 'issue', 
                              'concern', 'risk', 'decline']
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        if not text:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Calculate scores
        total_words = len(text.split())
        positive_score = positive_count / max(total_words, 1)
        negative_score = negative_count / max(total_words, 1)
        
        # Compound score
        compound = (positive_score - negative_score)
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': 1 - (positive_score + negative_score),
            'compound': compound
        }
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text"""
        if not text:
            return []
        
        # Simple keyword extraction
        keywords = ['technology', 'software', 'cloud', 'ai', 'machine learning',
                   'data', 'analytics', 'digital', 'transformation', 'innovation',
                   'automation', 'enterprise', 'solution', 'platform', 'service']
        
        text_lower = text.lower()
        found_topics = [kw for kw in keywords if kw in text_lower]
        
        return found_topics[:5]  # Return top 5 topics

# Data Enricher Class
class DataEnricher:
    def __init__(self):
        self.tech_stacks = [
            'AWS, Python, React',
            'Azure, .NET, Angular',
            'GCP, Java, Vue.js',
            'AWS, Node.js, React',
            'Kubernetes, Docker, Jenkins'
        ]
        
    def enrich_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich lead data with additional information"""
        enriched = {}
        
        # Simulate LinkedIn URL
        company_name = lead_data.get('company_name', 'company').lower().replace(' ', '-')
        enriched['linkedin_url'] = f"https://linkedin.com/company/{company_name}"
        
        # Simulate Twitter URL
        enriched['twitter_url'] = f"https://twitter.com/{company_name}"
        
        # Random technology stack
        enriched['technologies'] = random.choice(self.tech_stacks)
        
        # Company size (if not present)
        if 'employee_count' not in lead_data:
            enriched['employee_count'] = random.randint(10, 5000)
        
        # Founded year
        enriched['founded_year'] = random.randint(1990, 2023)
        
        return enriched