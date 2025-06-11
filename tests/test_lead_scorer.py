# tests/test_lead_scorer.py
"""
Unit tests for the Lead Scorer module

Tests the AI-powered lead scoring functionality including:
- Model initialization
- Score calculation
- Feature extraction
- Edge cases and error handling
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.lead_scorer import LeadScorer, LeadScoringModel


class TestLeadScorer(unittest.TestCase):
    """Test cases for the LeadScorer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scorer = LeadScorer()
        
        # Create sample lead data
        self.sample_lead = pd.Series({
            'name': 'John Doe',
            'email': 'john.doe@techcorp.com',
            'company': 'TechCorp Inc.',
            'title': 'Chief Technology Officer',
            'phone': '+1-555-123-4567',
            'location': 'San Francisco, CA',
            'industry': 'Technology',
            'company_size': '1000-5000',
            'revenue': '$100M-$500M',
            'technologies': 'Python, AWS, Machine Learning',
            'recent_funding': 'Series B - $50M',
            'website': 'https://techcorp.com',
            'linkedin': 'https://linkedin.com/company/techcorp',
            'description': 'Leading innovative technology solutions company focused on AI'
        })
        
        # Create sample DataFrame
        self.sample_df = pd.DataFrame([
            {
                'name': 'Jane Smith',
                'email': 'jane@startup.com',
                'company': 'AI Startup',
                'title': 'CEO',
                'industry': 'Technology',
                'company_size': '10-50',
                'revenue': '$1M-$10M'
            },
            {
                'name': 'Bob Johnson',
                'email': 'bob@enterprise.com',
                'company': 'Enterprise Corp',
                'title': 'Sales Manager',
                'industry': 'Manufacturing',
                'company_size': '10000+',
                'revenue': '$1B+'
            }
        ])
    
    def test_initialization(self):
        """Test LeadScorer initialization"""
        self.assertIsInstance(self.scorer, LeadScorer)
        self.assertIsNotNone(self.scorer.model)
        self.assertIsInstance(self.scorer.model, LeadScoringModel)
    
    def test_score_single_lead(self):
        """Test scoring a single lead"""
        score = self.scorer.score_lead(self.sample_lead)
        
        # Check score is in valid range
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        
        # Check score is reasonable for a good lead
        self.assertGreater(score, 50)  # CTO at tech company should score well
    
    def test_score_batch(self):
        """Test batch scoring of multiple leads"""
        scores = self.scorer.score_batch(self.sample_df)
        
        # Check return type
        self.assertIsInstance(scores, pd.Series)
        self.assertEqual(len(scores), len(self.sample_df))
        
        # Check all scores are valid
        self.assertTrue(all(0 <= score <= 100 for score in scores))
    
    def test_feature_extraction(self):
        """Test feature extraction from lead data"""
        features = self.scorer._extract_features(self.sample_lead)
        
        # Check features were extracted
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check specific features
        self.assertIn('has_email', features)
        self.assertIn('title_seniority', features)
        self.assertIn('company_size_score', features)
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        incomplete_lead = pd.Series({
            'name': 'Test User',
            'email': None,
            'company': '',
            'title': None
        })
        
        # Should still produce a valid score
        score = self.scorer.score_lead(incomplete_lead)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        
        # Score should be lower due to missing data
        self.assertLess(score, 50)
    
    def test_email_validation(self):
        """Test email validation logic"""
        # Valid corporate email
        corporate_lead = self.sample_lead.copy()
        corporate_lead['email'] = 'john@company.com'
        corporate_score = self.scorer.score_lead(corporate_lead)
        
        # Personal email
        personal_lead = self.sample_lead.copy()
        personal_lead['email'] = 'john@gmail.com'
        personal_score = self.scorer.score_lead(personal_lead)
        
        # Corporate email should score higher
        self.assertGreater(corporate_score, personal_score)
    
    def test_title_seniority_scoring(self):
        """Test title-based scoring"""
        # C-level title
        c_level_lead = self.sample_lead.copy()
        c_level_lead['title'] = 'Chief Executive Officer'
        c_level_score = self.scorer.score_lead(c_level_lead)
        
        # Junior title
        junior_lead = self.sample_lead.copy()
        junior_lead['title'] = 'Junior Analyst'
        junior_score = self.scorer.score_lead(junior_lead)
        
        # C-level should score higher
        self.assertGreater(c_level_score, junior_score)
    
    def test_company_size_scoring(self):
        """Test company size impact on scoring"""
        # Large company
        large_company_lead = self.sample_lead.copy()
        large_company_lead['company_size'] = '10000+'
        large_score = self.scorer.score_lead(large_company_lead)
        
        # Small company
        small_company_lead = self.sample_lead.copy()
        small_company_lead['company_size'] = '1-10'
        small_score = self.scorer.score_lead(small_company_lead)
        
        # Both should be valid scores
        self.assertGreaterEqual(large_score, 0)
        self.assertGreaterEqual(small_score, 0)
    
    def test_get_score_explanation(self):
        """Test score explanation generation"""
        score = self.scorer.score_lead(self.sample_lead)
        explanation = self.scorer.get_score_explanation(self.sample_lead)
        
        # Check explanation structure
        self.assertIsInstance(explanation, dict)
        self.assertIn('score', explanation)
        self.assertIn('factors', explanation)
        self.assertIn('recommendations', explanation)
        
        # Check score matches
        self.assertEqual(explanation['score'], score)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty lead
        empty_lead = pd.Series({})
        score = self.scorer.score_lead(empty_lead)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        
        # Lead with all None values
        none_lead = pd.Series({
            'name': None,
            'email': None,
            'company': None,
            'title': None
        })
        score = self.scorer.score_lead(none_lead)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_scoring_consistency(self):
        """Test that scoring is consistent"""
        # Score the same lead multiple times
        scores = [self.scorer.score_lead(self.sample_lead) for _ in range(5)]
        
        # All scores should be identical
        self.assertEqual(len(set(scores)), 1)
    
    def test_model_persistence(self):
        """Test model save and load functionality"""
        import tempfile
        
        # Score a lead
        original_score = self.scorer.score_lead(self.sample_lead)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            self.scorer.save_model(tmp.name)
            temp_path = tmp.name
        
        try:
            # Create new scorer and load model
            new_scorer = LeadScorer()
            new_scorer.load_model(temp_path)
            
            # Score should be the same
            new_score = new_scorer.score_lead(self.sample_lead)
            self.assertEqual(original_score, new_score)
        finally:
            # Clean up
            os.unlink(temp_path)


class TestLeadScoringModel(unittest.TestCase):
    """Test cases for the LeadScoringModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = LeadScoringModel()
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, LeadScoringModel)
        self.assertTrue(hasattr(self.model, 'predict'))
    
    def test_feature_vector_creation(self):
        """Test feature vector creation"""
        features = {
            'has_email': 1,
            'email_quality': 0.8,
            'title_seniority': 0.9,
            'company_size_score': 0.7,
            'has_phone': 1,
            'data_completeness': 0.85
        }
        
        vector = self.model._create_feature_vector(features)
        
        # Check vector properties
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), len(features))
    
    def test_prediction_bounds(self):
        """Test that predictions are within valid bounds"""
        # Generate random feature vectors
        for _ in range(10):
            features = {
                'has_email': np.random.choice([0, 1]),
                'email_quality': np.random.random(),
                'title_seniority': np.random.random(),
                'company_size_score': np.random.random(),
                'has_phone': np.random.choice([0, 1]),
                'data_completeness': np.random.random()
            }
            
            score = self.model.predict(features)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)


if __name__ == '__main__':
    unittest.main()