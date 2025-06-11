"""
Lead Scoring Model for AI Lead Enhancer

This module implements the machine learning model for scoring leads based on
various features including company size, industry, technology stack, and engagement signals.
"""

import logging
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Add these constants after the imports
MODEL_CONFIG = {
    "scoring": {
        "model_path": "models/lead_scorer.pkl",
        "feature_columns": [
            "employee_count_normalized",
            "industry_score", 
            "technology_score",
            "growth_score",
            "engagement_score"
        ],
        "thresholds": {
            "A": 85,
            "B": 70,
            "C": 55,
            "D": 40,
            "F": 0
        }
    }
}

FEATURE_WEIGHTS = {
    "industry": {
        "Technology": 1.0,
        "Financial Services": 0.9,
        "Healthcare": 0.85,
        "Retail": 0.7,
        "Manufacturing": 0.7,
        "Education": 0.6,
        "Other": 0.5
    }
}

SCORING_RUBRIC = {
    "hot": {
        "range": [85, 100],
        "actions": [
            "Schedule immediate outreach",
            "Assign to senior sales rep",
            "Prepare custom demo",
            "Fast-track through sales process"
        ]
    },
    "warm": {
        "range": [70, 84],
        "actions": [
            "Add to priority nurture campaign",
            "Schedule discovery call within 48 hours",
            "Send relevant case studies"
        ]
    },
    "cool": {
        "range": [40, 69],
        "actions": [
            "Add to standard nurture sequence",
            "Monitor for engagement signals",
            "Send educational content"
        ]
    },
    "cold": {
        "range": [0, 39],
        "actions": [
            "Add to long-term nurture",
            "Review in 6 months",
            "Consider different approach"
        ]
    }
}


logger = logging.getLogger(__name__)


class ScoreBreakdown:
    """Represents the breakdown of a lead score"""
    
    def __init__(self):
        self.company_size_score = 0.0
        self.industry_score = 0.0
        self.technology_score = 0.0
        self.engagement_score = 0.0
        self.growth_score = 0.0
        
    def to_dict(self) -> Dict[str, float]:
        """Convert breakdown to dictionary"""
        return {
            "company_size": round(self.company_size_score, 2),
            "industry_fit": round(self.industry_score, 2),
            "technology_stack": round(self.technology_score, 2),
            "engagement_signals": round(self.engagement_score, 2),
            "growth_indicators": round(self.growth_score, 2)
        }
    
    def total_score(self) -> float:
        """Calculate total score"""
        return sum([
            self.company_size_score,
            self.industry_score,
            self.technology_score,
            self.engagement_score,
            self.growth_score
        ])


class ScoringModel:
    """Machine learning model for lead scoring"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or MODEL_CONFIG["scoring"]["model_path"]
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = MODEL_CONFIG["scoring"]["feature_columns"]
        self._load_or_create_model()
        
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        model_file = Path(self.model_path)
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.scaler = saved_data['scaler']
                logger.info("Loaded existing model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self._create_new_model()
        else:
            self._create_new_model()
            
    def _create_new_model(self):
        """Create and train a new model with synthetic data"""
        logger.info("Creating new model with synthetic training data")
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features
        company_size = np.random.choice([10, 50, 150, 300, 1000], n_samples)
        industry_score = np.random.uniform(0.3, 1.0, n_samples)
        tech_score = np.random.uniform(0.2, 1.0, n_samples)
        growth_score = np.random.uniform(0.1, 1.0, n_samples)
        engagement_score = np.random.uniform(0.0, 1.0, n_samples)
        
        # Create target scores with some noise
        target = (
            0.25 * np.log1p(company_size) / np.log1p(1000) * 100 +
            0.20 * industry_score * 100 +
            0.20 * tech_score * 100 +
            0.20 * growth_score * 100 +
            0.15 * engagement_score * 100 +
            np.random.normal(0, 5, n_samples)
        )
        
        # Clip to valid range
        target = np.clip(target, 0, 100)
        
        # Create DataFrame
        X = pd.DataFrame({
            'employee_count_normalized': np.log1p(company_size) / np.log1p(1000),
            'industry_score': industry_score,
            'technology_score': tech_score,
            'growth_score': growth_score,
            'engagement_score': engagement_score
        })
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, target)
        
        # Save model
        self._save_model()
        
    def _save_model(self):
        """Save model to disk"""
        try:
            model_dir = Path(self.model_path).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler
                }, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions on features"""
        X_scaled = self.scaler.transform(features)
        return self.model.predict(X_scaled)


class LeadScorer:
    """Main class for scoring leads"""
    
    def __init__(self):
        self.model = ScoringModel()
        self.weights = FEATURE_WEIGHTS
        self.thresholds = MODEL_CONFIG["scoring"]["thresholds"]
        
    def score_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single lead
        
        Args:
            lead_data: Dictionary containing lead information
            
        Returns:
            Dictionary with score, grade, breakdown, and confidence
        """
        try:
            # Calculate individual scores
            breakdown = self._calculate_breakdown(lead_data)
            
            # Prepare features for ML model
            features = self._prepare_features(lead_data, breakdown)
            
            # Get ML prediction
            ml_score = self.model.predict(features)[0]
            
            # Combine rule-based and ML scores
            rule_score = breakdown.total_score()
            final_score = 0.6 * ml_score + 0.4 * rule_score
            final_score = max(0, min(100, final_score))
            
            # Calculate confidence
            confidence = self._calculate_confidence(lead_data)
            
            # Determine grade
            grade = self._score_to_grade(final_score)
            
            return {
                "score": int(final_score),
                "grade": grade,
                "breakdown": breakdown.to_dict(),
                "confidence": round(confidence, 2),
                "ml_score": round(ml_score, 2),
                "rule_score": round(rule_score, 2)
            }
            
        except Exception as e:
            logger.error(f"Error scoring lead: {e}")
            return {
                "score": 0,
                "grade": "F",
                "breakdown": ScoreBreakdown().to_dict(),
                "confidence": 0.0,
                "error": str(e)
            }
            
    def _calculate_breakdown(self, lead_data: Dict[str, Any]) -> ScoreBreakdown:
        """Calculate score breakdown for each category"""
        breakdown = ScoreBreakdown()
        
        # Company size score (25% weight)
        employee_count = lead_data.get("employee_count", 0)
        breakdown.company_size_score = self._score_company_size(employee_count) * 25
        
        # Industry score (20% weight)
        industry = lead_data.get("industry", "Other")
        breakdown.industry_score = self._score_industry(industry) * 20
        
        # Technology score (20% weight)
        tech_stack = lead_data.get("technology_stack", [])
        breakdown.technology_score = self._score_technology(tech_stack) * 20
        
        # Engagement score (15% weight)
        engagement_data = lead_data.get("engagement", {})
        breakdown.engagement_score = self._score_engagement(engagement_data) * 15
        
        # Growth score (20% weight)
        growth_data = lead_data.get("growth_indicators", {})
        breakdown.growth_score = self._score_growth(growth_data) * 20
        
        return breakdown
        
    def _score_company_size(self, employee_count: int) -> float:
        """Score based on company size"""
        if employee_count <= 10:
            return 0.3
        elif employee_count <= 50:
            return 0.5
        elif employee_count <= 200:
            return 0.8
        elif employee_count <= 500:
            return 1.0
        else:
            return 0.9
            
    def _score_industry(self, industry: str) -> float:
        """Score based on industry fit"""
        return self.weights["industry"].get(industry, 0.5)
        
    def _score_technology(self, tech_stack: List[str]) -> float:
        """Score based on technology stack"""
        if not tech_stack:
            return 0.5
            
        # Check for modern technologies
        modern_tech = ["cloud", "kubernetes", "microservices", "api", "saas"]
        matches = sum(1 for tech in tech_stack if any(m in tech.lower() for m in modern_tech))
        
        return min(1.0, 0.5 + (matches * 0.1))
        
    def _score_engagement(self, engagement_data: Dict[str, Any]) -> float:
        """Score based on engagement signals"""
        score = 0.5  # Base score
        
        if engagement_data.get("website_visits", 0) > 5:
            score += 0.2
        if engagement_data.get("content_downloads", 0) > 0:
            score += 0.2
        if engagement_data.get("demo_requests", 0) > 0:
            score += 0.1
            
        return min(1.0, score)
        
    def _score_growth(self, growth_data: Dict[str, Any]) -> float:
        """Score based on growth indicators"""
        growth_rate = growth_data.get("revenue_growth", 0)
        funding = growth_data.get("recent_funding", False)
        
        if growth_rate > 50:
            base_score = 1.0
        elif growth_rate > 20:
            base_score = 0.8
        elif growth_rate > 0:
            base_score = 0.6
        else:
            base_score = 0.4
            
        if funding:
            base_score = min(1.0, base_score + 0.2)
            
        return base_score
        
    def _prepare_features(self, lead_data: Dict[str, Any], breakdown: ScoreBreakdown) -> pd.DataFrame:
        """Prepare features for ML model"""
        employee_count = lead_data.get("employee_count", 1)
        
        features = pd.DataFrame([{
            'employee_count_normalized': np.log1p(employee_count) / np.log1p(1000),
            'industry_score': breakdown.industry_score / 20,
            'technology_score': breakdown.technology_score / 20,
            'growth_score': breakdown.growth_score / 20,
            'engagement_score': breakdown.engagement_score / 15
        }])
        
        return features
        
    def _calculate_confidence(self, lead_data: Dict[str, Any]) -> float:
        """Calculate confidence level based on data completeness"""
        required_fields = ["company_name", "employee_count", "industry", "website"]
        optional_fields = ["technology_stack", "growth_indicators", "engagement"]
        
        # Check required fields
        required_present = sum(1 for field in required_fields if lead_data.get(field))
        required_score = required_present / len(required_fields)
        
        # Check optional fields
        optional_present = sum(1 for field in optional_fields if lead_data.get(field))
        optional_score = optional_present / len(optional_fields)
        
        # Weighted confidence
        confidence = 0.7 * required_score + 0.3 * optional_score
        
        return confidence
        
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        for grade, threshold in self.thresholds.items():
            if score >= threshold:
                return grade
        return "F"
        
    def generate_recommendations(self, score_result: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on score"""
        recommendations = []
        score = score_result["score"]
        breakdown = score_result["breakdown"]
        
        # Find scoring category
        for category, rubric in SCORING_RUBRIC.items():
            if rubric["range"][0] <= score <= rubric["range"][1]:
                recommendations.extend(rubric["actions"])
                break
                
        # Add specific recommendations based on breakdown
        if breakdown["company_size"] < 15:
            recommendations.append("Consider company size fit for your solution")
        if breakdown["technology_stack"] < 10:
            recommendations.append("May need technology modernization discussion")
        if breakdown["engagement_signals"] > 12:
            recommendations.append("High engagement - strike while iron is hot")
            
        return recommendations[:5]  # Return top 5 recommendations