"""
NLP Processing Module for AI Lead Enhancer

This module handles natural language processing tasks including text analysis,
entity extraction, sentiment analysis, and insight generation for lead enrichment.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from datetime import datetime
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup


# Define MODEL_CONFIG locally to avoid circular imports
MODEL_CONFIG = {
    "nlp": {
        "sentiment_threshold": 0.5,
        "max_summary_length": 200,
        "min_keyword_relevance": 0.3
    }
}

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    logger.warning("Could not download NLTK data")


class TextAnalyzer:
    """Analyzes text content for insights and entities"""
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.config = MODEL_CONFIG["nlp"]
        
    def _load_spacy_model(self):
        """Load spaCy model"""
        try:
            return spacy.load("en_core_web_sm")
        except:
            logger.error("SpaCy model not found. Please install: python -m spacy download en_core_web_sm")
            return None
            
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for various NLP features
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.nlp or not text:
            return {}
            
        doc = self.nlp(text[:self.config["max_tokens"] * 4])  # Rough char to token conversion
        
        return {
            "entities": self._extract_entities(doc),
            "keywords": self._extract_keywords(doc),
            "sentiment": self._analyze_sentiment(text),
            "topics": self._extract_topics(doc),
            "summary": self._generate_summary(text)
        }
        
    def _extract_entities(self, doc) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ in self.config["entity_types"]:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
                
        # Deduplicate
        for label in entities:
            entities[label] = list(set(entities[label]))
            
        return entities
        
    def _extract_keywords(self, doc) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF-like scoring"""
        # Filter tokens
        tokens = [
            token.lemma_.lower() for token in doc
            if not token.is_stop 
            and not token.is_punct 
            and len(token.text) >= self.config["keyword_extraction"]["min_word_length"]
            and token.pos_ in self.config["keyword_extraction"]["pos_tags"]
        ]
        
        # Calculate frequency
        word_freq = Counter(tokens)
        
        # Score based on frequency and position
        keyword_scores = {}
        for word, freq in word_freq.items():
            # Give higher score to words appearing earlier
            first_occurrence = next(i for i, token in enumerate(doc) if token.lemma_.lower() == word)
            position_score = 1 / (1 + first_occurrence * 0.01)
            
            keyword_scores[word] = freq * position_score
            
        # Get top keywords
        top_keywords = sorted(
            keyword_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.config["keyword_extraction"]["max_keywords"]]
        
        return top_keywords
        
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        try:
            blob = TextBlob(text)
            
            return {
                "polarity": round(blob.sentiment.polarity, 3),
                "subjectivity": round(blob.sentiment.subjectivity, 3),
                "classification": self._classify_sentiment(blob.sentiment.polarity)
            }
        except:
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "classification": "neutral"
            }
            
    def _classify_sentiment(self, polarity: float) -> str:
        """Classify sentiment based on polarity score"""
        if polarity > self.config["sentiment_threshold"]:
            return "positive"
        elif polarity < -self.config["sentiment_threshold"]:
            return "negative"
        else:
            return "neutral"
            
    def _extract_topics(self, doc) -> List[str]:
        """Extract main topics from text"""
        # Simple topic extraction using noun phrases
        topics = []
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Keep short phrases
                topics.append(chunk.text.lower())
                
        # Deduplicate and return top topics
        topic_counts = Counter(topics)
        return [topic for topic, _ in topic_counts.most_common(5)]
        
    def _generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a brief summary of the text"""
        sentences = text.split('.')[:max_sentences]
        summary = '. '.join(s.strip() for s in sentences if s.strip())
        
        if summary and not summary.endswith('.'):
            summary += '.'
            
        return summary


class InsightGenerator:
    """Generates business insights from analyzed data"""
    
    def __init__(self):
        self.industry_insights = {
            "Technology": [
                "Strong demand for digital transformation solutions",
                "Focus on cloud migration and modernization",
                "AI/ML adoption is a key priority"
            ],
            "Finance": [
                "Regulatory compliance is critical",
                "Security and data protection are top concerns",
                "Digital banking transformation ongoing"
            ],
            "Healthcare": [
                "HIPAA compliance is mandatory",
                "Telemedicine adoption accelerating",
                "Data interoperability challenges"
            ],
            "E-commerce": [
                "Customer experience optimization crucial",
                "Omnichannel strategy implementation",
                "Supply chain optimization needs"
            ]
        }
        
    def generate_insights(self, company_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate business insights based on company data and analysis
        
        Args:
            company_data: Company information
            analysis_results: Results from text analysis
            
        Returns:
            Dictionary containing various insights
        """
        insights = {
            "industry_trends": self._get_industry_insights(company_data.get("industry", "Other")),
            "technology_recommendations": self._get_tech_recommendations(company_data),
            "outreach_strategy": self._get_outreach_strategy(company_data, analysis_results),
            "pain_points": self._identify_pain_points(company_data, analysis_results),
            "value_propositions": self._generate_value_props(company_data)
        }
        
        return insights
        
    def _get_industry_insights(self, industry: str) -> List[str]:
        """Get industry-specific insights"""
        return self.industry_insights.get(industry, [
            "Digital transformation opportunities exist",
            "Process automation can improve efficiency",
            "Data-driven decision making is becoming crucial"
        ])
        
    def _get_tech_recommendations(self, company_data: Dict[str, Any]) -> List[str]:
        """Generate technology recommendations"""
        recommendations = []
        tech_stack = company_data.get("technology_stack", [])
        
        # Check for missing modern technologies
        if not any("cloud" in tech.lower() for tech in tech_stack):
            recommendations.append("Consider cloud migration for scalability")
            
        if not any("api" in tech.lower() for tech in tech_stack):
            recommendations.append("API-first architecture could improve integration")
            
        if len(tech_stack) < 5:
            recommendations.append("Technology stack appears limited - modernization opportunity")
            
        return recommendations
        
    def _get_outreach_strategy(self, company_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate outreach strategy recommendations"""
        strategy = {
            "approach": "consultative",
            "tone": "professional",
            "key_points": [],
            "avoid": []
        }
        
        # Adjust based on company size
        employee_count = company_data.get("employee_count", 0)
        if employee_count > 500:
            strategy["approach"] = "enterprise"
            strategy["key_points"].append("ROI and scalability focus")
        elif employee_count < 50:
            strategy["approach"] = "SMB-friendly"
            strategy["key_points"].append("Quick implementation and value")
            
        # Adjust based on sentiment
        sentiment = analysis_results.get("sentiment", {})
        if sentiment.get("classification") == "negative":
            strategy["tone"] = "empathetic"
            strategy["key_points"].append("Problem-solving focus")
            
        return strategy
        
    def _identify_pain_points(self, company_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> List[str]:
        """Identify potential pain points"""
        pain_points = []
        
        # Based on company size and industry
        if company_data.get("employee_count", 0) > 200:
            pain_points.append("Scaling challenges with current systems")
            
        if company_data.get("industry") == "Technology":
            pain_points.append("Technical debt and legacy system maintenance")
            
        # Based on keywords
        keywords = [kw[0] for kw in analysis_results.get("keywords", [])]
        if any(word in keywords for word in ["slow", "problem", "issue", "challenge"]):
            pain_points.append("Performance or reliability concerns")
            
        return pain_points
        
    def _generate_value_props(self, company_data: Dict[str, Any]) -> List[str]:
        """Generate relevant value propositions"""
        value_props = []
        
        # Generic value props
        value_props.extend([
            "Reduce operational costs by up to 30%",
            "Improve team productivity and efficiency",
            "Scale seamlessly as your business grows"
        ])
        
        # Industry-specific
        industry = company_data.get("industry", "Other")
        if industry == "Technology":
            value_props.append("Accelerate development cycles and time-to-market")
        elif industry == "Finance":
            value_props.append("Ensure compliance while improving agility")
        elif industry == "Healthcare":
            value_props.append("Enhance patient care through better data management")
            
        return value_props[:3]


class NLPProcessor:
    """Main NLP processor combining analysis and insight generation"""
    
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.insight_generator = InsightGenerator()
        self._cache = {}
        
    def process_company_content(self, company_name: str, content: str) -> Dict[str, Any]:
        """
        Process company content and generate insights
        
        Args:
            company_name: Name of the company
            content: Text content to analyze (e.g., from website)
            
        Returns:
            Dictionary containing analysis and insights
        """
        # Check cache
        cache_key = f"{company_name}:{hash(content[:100])}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Analyze content
        analysis = self.analyzer.analyze_text(content)
        
        # Generate insights
        company_data = {"company_name": company_name}
        insights = self.insight_generator.generate_insights(company_data, analysis)
        
        result = {
            "company": company_name,
            "analysis": analysis,
            "insights": insights,
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
        
    def extract_company_description(self, html_content: str) -> str:
        """Extract company description from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Look for common description areas
            description = ""
            
            # Check meta description
            meta_desc = soup.find("meta", {"name": "description"})
            if meta_desc and meta_desc.get("content"):
                description = meta_desc["content"]
                
            # Check common content areas if no meta description
            if not description:
                for selector in ["div.about", "section.about", "div#about", "main", "article"]:
                    element = soup.select_one(selector)
                    if element:
                        description = element.get_text(strip=True)[:500]
                        break
                        
            #            # If still no description, get first substantial paragraph
            if not description:
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 100:
                        description = text
                        break
                        
            return description[:1000]  # Limit length
            
        except Exception as e:
            logger.error(f"Error extracting description: {e}")
            return ""
            
    def generate_insights(self, company_name: str, company_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate quick insights for a company
        
        Args:
            company_name: Name of the company
            company_data: Optional additional company data
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Basic insights
        insights.append(f"{company_name} shows potential for digital transformation")
        
        if company_data:
            # Size-based insights
            employee_count = company_data.get("employee_count", 0)
            if employee_count > 100:
                insights.append("Enterprise-scale solution would be appropriate")
            else:
                insights.append("SMB-focused approach recommended")
                
            # Industry insights
            industry = company_data.get("industry", "")
            if industry:
                industry_insights = self.insight_generator._get_industry_insights(industry)
                insights.extend(industry_insights[:2])
                
        return insights
        
    def analyze_website_content(self, url: str) -> Dict[str, Any]:
        """
        Analyze content from a website URL
        
        Args:
            url: Website URL to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Fetch website content
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Extract text content
            description = self.extract_company_description(response.text)
            
            # Analyze content
            if description:
                analysis = self.analyzer.analyze_text(description)
                return {
                    "url": url,
                    "content_analyzed": True,
                    "analysis": analysis,
                    "description": description[:200] + "..." if len(description) > 200 else description
                }
            else:
                return {
                    "url": url,
                    "content_analyzed": False,
                    "error": "No substantial content found"
                }
                
        except Exception as e:
            logger.error(f"Error analyzing website {url}: {e}")
            return {
                "url": url,
                "content_analyzed": False,
                "error": str(e)
            }
            
    def get_personalization_suggestions(self, lead_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalization suggestions for outreach
        
        Args:
            lead_data: Lead information
            analysis: NLP analysis results
            
        Returns:
            Dictionary with personalization suggestions
        """
        suggestions = {
            "greeting": self._generate_greeting(lead_data),
            "opening_line": self._generate_opening(lead_data, analysis),
            "value_prop": self._select_value_prop(lead_data, analysis),
            "call_to_action": self._generate_cta(lead_data),
            "tone": self._determine_tone(analysis)
        }
        
        return suggestions
        
    def _generate_greeting(self, lead_data: Dict[str, Any]) -> str:
        """Generate personalized greeting"""
        contact_name = lead_data.get("contact_name", "")
        if contact_name:
            return f"Hi {contact_name.split()[0]}"
        else:
            return "Hello"
            
    def _generate_opening(self, lead_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate personalized opening line"""
        company = lead_data.get("company_name", "your company")
        
        # Check for recent news or growth
        if analysis.get("entities", {}).get("MONEY"):
            return f"Congratulations on {company}'s recent funding round!"
        elif analysis.get("keywords"):
            keyword = analysis["keywords"][0][0]
            return f"I noticed {company}'s focus on {keyword}."
        else:
            return f"I came across {company} and was impressed by your work."
            
    def _select_value_prop(self, lead_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Select most relevant value proposition"""
        industry = lead_data.get("industry", "Other")
        sentiment = analysis.get("sentiment", {}).get("classification", "neutral")
        
        if sentiment == "negative":
            return "We help companies like yours overcome technical challenges and improve efficiency."
        elif industry == "Technology":
            return "We accelerate development cycles and reduce technical debt for growing tech companies."
        else:
            return "We help businesses modernize their operations and scale efficiently."
            
    def _generate_cta(self, lead_data: Dict[str, Any]) -> str:
        """Generate appropriate call-to-action"""
        employee_count = lead_data.get("employee_count", 0)
        
        if employee_count > 500:
            return "Would you be open to a strategic discussion about your technology roadmap?"
        elif employee_count > 100:
            return "I'd love to show you how we've helped similar companies. Do you have 15 minutes this week?"
        else:
            return "Would you be interested in a quick demo tailored to your needs?"
            
    def _determine_tone(self, analysis: Dict[str, Any]) -> str:
        """Determine appropriate communication tone"""
        formality = analysis.get("sentiment", {}).get("subjectivity", 0.5)
        
        if formality > 0.7:
            return "casual"
        elif formality < 0.3:
            return "formal"
        else:
            return "professional"