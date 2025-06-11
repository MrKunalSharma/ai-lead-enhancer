"""
Lead Enricher - Main enrichment orchestrator

Coordinates multiple data sources to enrich lead information with additional
context and business intelligence.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass
import json

# Define constants locally to avoid circular imports
ENRICHMENT_CONFIG = {
    "api_timeout": 30,
    "max_retries": 3,
    "cache_ttl": 86400,  # 24 hours
    "batch_size": 100,
    "rate_limit": 60  # requests per minute
}

QUALITY_RULES = {
    "email": {
        "required": True,
        "validator": "email_regex",
        "score_weight": 0.2
    },
    "phone": {
        "required": False,
        "validator": "phone_regex", 
        "score_weight": 0.1
    },
    "company": {
        "required": True,
        "validator": "not_empty",
        "score_weight": 0.15
    },
    "title": {
        "required": True,
        "validator": "not_empty",
        "score_weight": 0.15
    }
}

FIELD_MAPPINGS = {
    "company_name": ["company", "organization", "employer"],
    "contact_email": ["email", "email_address", "contact"],
    "job_title": ["title", "position", "role"],
    "phone_number": ["phone", "telephone", "mobile"],
    "location": ["location", "city", "address", "region"]
}

from .data_sources import DataSourceManager

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    """Container for enrichment results"""
    original_data: Dict[str, Any]
    enriched_data: Dict[str, Any]
    sources_used: List[str]
    confidence_score: float
    enrichment_time: float
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_data": self.original_data,
            "enriched_data": self.enriched_data,
            "sources_used": self.sources_used,
            "confidence_score": self.confidence_score,
            "enrichment_time": self.enrichment_time,
            "errors": self.errors,
            "timestamp": datetime.utcnow().isoformat()
        }


class LeadEnricher:
    """Main class for enriching lead data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ENRICHMENT_CONFIG
        self.data_source_manager = DataSourceManager(self.config)
        self.quality_rules = QUALITY_RULES
        self._cache = {}
        
    def enrich_lead(self, lead_data: Dict[str, Any]) -> EnrichmentResult:
        """
        Synchronously enrich a single lead
        
        Args:
            lead_data: Dictionary containing lead information
            
        Returns:
            EnrichmentResult with enriched data
        """
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.enrich_lead_async(lead_data))
        finally:
            loop.close()
            
    async def enrich_lead_async(self, lead_data: Dict[str, Any]) -> EnrichmentResult:
        """
        Asynchronously enrich a single lead
        
        Args:
            lead_data: Dictionary containing lead information
            
        Returns:
            EnrichmentResult with enriched data
        """
        start_time = datetime.utcnow()
        errors = []
        sources_used = []
        
        # Check cache first
        cache_key = self._generate_cache_key(lead_data)
        if cache_key in self._cache:
            logger.info(f"Cache hit for {lead_data.get('company_name', 'unknown')}")
            return self._cache[cache_key]
            
        # Validate required fields
        if not self._validate_required_fields(lead_data):
            return EnrichmentResult(
                original_data=lead_data,
                enriched_data=lead_data,
                sources_used=[],
                confidence_score=0.0,
                enrichment_time=0.0,
                errors=["Missing required fields"]
            )
            
        # Create a copy for enrichment
        enriched_data = lead_data.copy()
        
        # Normalize input data
        enriched_data = self._normalize_data(enriched_data)
        
        # Collect enrichment tasks
        tasks = []
        
        # Web scraping enrichment
        if self.config["sources"]["web_scraping"]["enabled"] and enriched_data.get("website"):
            tasks.append(self._enrich_from_website(enriched_data))
            
        # API enrichment
        if self.config["sources"]["apis"]["enabled"]:
            tasks.append(self._enrich_from_apis(enriched_data))
            
        # Database enrichment
        if self.config["sources"]["database"]["enabled"]:
            tasks.append(self._enrich_from_database(enriched_data))
            
        # Execute all enrichment tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                elif isinstance(result, dict):
                    # Merge enrichment results
                    for source, data in result.items():
                        sources_used.append(source)
                        enriched_data = self._merge_data(enriched_data, data)
                        
        # Post-process enriched data
        enriched_data = self._post_process(enriched_data)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(enriched_data, lead_data)
        
        # Calculate enrichment time
        enrichment_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Create result
        result = EnrichmentResult(
            original_data=lead_data,
            enriched_data=enriched_data,
            sources_used=list(set(sources_used)),
            confidence_score=confidence_score,
            enrichment_time=enrichment_time,
            errors=errors
        )
        
        # Cache result
        if confidence_score > self.config["quality"]["min_confidence"]:
            self._cache[cache_key] = result
            
        return result
        
    def _validate_required_fields(self, lead_data: Dict[str, Any]) -> bool:
        """Validate that required fields are present"""
        required = self.config["fields"]["required"]
        return all(lead_data.get(field) for field in required)
        
    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data formats"""
        normalized = data.copy()
        
        # Normalize website URL
        if "website" in normalized and normalized["website"]:
            url = normalized["website"]
            if not url.startswith(("http://", "https://")):
                normalized["website"] = f"https://{url}"
                
        # Normalize employee count
        if "employee_count" in normalized:
            emp_count = normalized["employee_count"]
            if isinstance(emp_count, str):
                # Handle ranges like "50-100"
                if "-" in emp_count:
                    parts = emp_count.split("-")
                    normalized["employee_count"] = int(parts[0])
                else:
                    # Extract numbers from string
                    import re
                    numbers = re.findall(r'\d+', emp_count)
                    if numbers:
                        normalized["employee_count"] = int(numbers[0])
                        
        # Normalize phone numbers
        if "phone" in normalized and normalized["phone"]:
            phone = normalized["phone"]
            # Remove non-numeric characters
            normalized["phone"] = ''.join(c for c in phone if c.isdigit() or c == '+')
            
        return normalized
        
    async def _enrich_from_website(self, lead_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Enrich data by scraping company website"""
        try:
            website_data = await self.data_source_manager.scrape_website(lead_data["website"])
            
            enriched = {}
            
            # Extract technology stack
            if "html_content" in website_data:
                tech_stack = self._detect_technologies(website_data["html_content"])
                if tech_stack:
                    enriched["technology_stack"] = tech_stack
                    
            # Extract meta information
            if "meta_data" in website_data:
                meta = website_data["meta_data"]
                if meta.get("description"):
                    enriched["description"] = meta["description"]
                if meta.get("keywords"):
                    enriched["keywords"] = meta["keywords"]
                    
            # Extract social profiles
            if "social_links" in website_data:
                enriched["social_profiles"] = website_data["social_links"]
                
            return {"web_scraping": enriched}
            
        except Exception as e:
            logger.error(f"Error in website enrichment: {e}")
            return {}
            
    async def _enrich_from_apis(self, lead_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Enrich data from external APIs"""
        api_results = {}
        
        # Example: Clearbit-style enrichment (mock implementation)
        if lead_data.get("website"):
            try:
                # In real implementation, this would call actual APIs
                mock_api_data = {
                    "industry": "Technology",
                    "employee_count": 150,
                    "founded_year": 2015,
                    "revenue_range": "$10M - $50M",
                    "location": {
                        "city": "San Francisco",
                        "state": "CA",
                        "country": "USA"
                    }
                }
                api_results["clearbit"] = mock_api_data
            except Exception as e:
                logger.error(f"API enrichment error: {e}")
                
        return api_results
        
    async def _enrich_from_database(self, lead_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Enrich data from internal database"""
        # Mock implementation - would query actual database
        return {"database": {}}
        
    def _detect_technologies(self, html_content: str) -> List[str]:
        """Detect technologies used on website"""
        from . import TECH_PATTERNS
        detected = []
        
        html_lower = html_content.lower()
        
        for category, patterns in TECH_PATTERNS.items():
            for tech, indicators in patterns.items():
                if any(indicator in html_lower for indicator in indicators):
                    detected.append(tech)
                    
        return list(set(detected))
        
    def _merge_data(self, existing: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently merge new data with existing data"""
        merged = existing.copy()
        
        for key, value in new_data.items():
            if key not in merged or not merged[key]:
                # Add new field
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # Merge dictionaries
                merged[key].update(value)
            elif key == "employee_count":
                # Take the average for employee count
                merged[key] = int((merged[key] + value) / 2)
            # For other conflicts, keep existing value
                
        return merged
        
    def _post_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process enriched data"""
        processed = data.copy()
        
        # Ensure data quality
        if "employee_count" in processed:
            rules = self.quality_rules["employee_count"]
            count = processed["employee_count"]
            if count < rules["min"] or count > rules["max"]:
                processed["employee_count"] = None
                
        # Categorize company size
        if processed.get("employee_count"):
            count = processed["employee_count"]
            if count <= 10:
                processed["company_size_category"] = "Startup"
            elif count <= 50:
                processed["company_size_category"] = "Small"
            elif count <= 200:
                processed["company_size_category"] = "Medium"
            elif count <= 1000:
                processed["company_size_category"] = "Large"
            else:
                processed["company_size_category"] = "Enterprise"
                
        # Add enrichment metadata
        processed["enrichment_date"] = datetime.utcnow().isoformat()
        processed["enrichment_version"] = "1.0"
        
        return processed
        
    def _calculate_confidence(self, enriched: Dict[str, Any], original: Dict[str, Any]) -> float:
        """Calculate confidence score for enrichment"""
        # Count how many fields were enriched
        enrichable_fields = self.config["fields"]["enrichable"]
        
        enriched_count = 0
        total_fields = len(enrichable_fields)
        
        for field in enrichable_fields:
            if field in enriched and enriched[field] and field not in original:
                enriched_count += 1
                
        # Base confidence on enrichment ratio
        confidence = enriched_count / total_fields if total_fields > 0 else 0
        
        # Boost confidence if key fields are present
        key_fields = ["industry", "employee_count", "technology_stack"]
        key_present = sum(1 for field in key_fields if field in enriched and enriched[field])
        confidence += (key_present / len(key_fields)) * 0.3
        
        # Cap at 1.0
        confidence = min(1.0, confidence)
        
        return round(confidence, 2)
        
    def _generate_cache_key(self, lead_data: Dict[str, Any]) -> str:
        """Generate cache key for lead data"""
        # Use company name and website as cache key
        company = lead_data.get("company_name", "").lower().replace(" ", "_")
        website = lead_data.get("website", "").replace("https://", "").replace("http://", "")
        return f"{company}:{website}"
        
    def get_sources_used(self) -> List[str]:
        """Get list of data sources that were used"""
        sources = []
        if self.config["sources"]["web_scraping"]["enabled"]:
            sources.append("web_scraping")
        if self.config["sources"]["apis"]["enabled"]:
            sources.extend(self.config["sources"]["apis"]["providers"])
        if self.config["sources"]["database"]["enabled"]:
            sources.append("database")
        return sources
        
    def batch_enrich(self, leads: List[Dict[str, Any]], max_concurrent: int = 10) -> List[EnrichmentResult]:
        """
        Enrich multiple leads in batch
        
        Args:
            leads: List of lead dictionaries
            max_concurrent: Maximum concurrent enrichments
            
        Returns:
            List of EnrichmentResult objects
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.batch_enrich_async(leads, max_concurrent))
        finally:
            loop.close()
            
    async def batch_enrich_async(self, leads: List[Dict[str, Any]], max_concurrent: int = 10) -> List[EnrichmentResult]:
        """
        Asynchronously enrich multiple leads in batch
        
        Args:
            leads: List of lead dictionaries
            max_concurrent: Maximum concurrent enrichments
            
        Returns:
            List of EnrichmentResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def enrich_with_semaphore(lead):
            async with semaphore:
                return await self.enrich_lead_async(lead)
                
        tasks = [enrich_with_semaphore(lead) for lead in leads]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                final_results.append(EnrichmentResult(
                    original_data=leads[i],
                    enriched_data=leads[i],
                    sources_used=[],
                    confidence_score=0.0,
                    enrichment_time=0.0,
                    errors=[str(result)]
                ))
            else:
                final_results.append(result)
                
        return final_results
        
    def export_enriched_data(self, results: List[EnrichmentResult], format: str = "csv") -> str:
        """
        Export enriched data in various formats
        
        Args:
            results: List of EnrichmentResult objects
            format: Export format (csv, json, excel)
            
        Returns:
            File path of exported data
        """
        import pandas as pd
        from datetime import datetime
        
        # Convert results to list of dictionaries
        data = []
        for result in results:
            row = result.enriched_data.copy()
            row["confidence_score"] = result.confidence_score
            row["sources_used"] = ", ".join(result.sources_used)
            row["enrichment_time"] = result.enrichment_time
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            filename = f"enriched_leads_{timestamp}.csv"
            df.to_csv(filename, index=False)
        elif format == "json":
            filename = f"enriched_leads_{timestamp}.json"
            df.to_json(filename, orient="records", indent=2)
        elif format == "excel":
            filename = f"enriched_leads_{timestamp}.xlsx"
            df.to_excel(filename, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return filename
        
    def get_enrichment_stats(self, results: List[EnrichmentResult]) -> Dict[str, Any]:
        """
        Calculate statistics from enrichment results
        
        Args:
            results: List of EnrichmentResult objects
            
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {}
            
        total = len(results)
        successful = sum(1 for r in results if r.confidence_score > 0.5)
        
        avg_confidence = sum(r.confidence_score for r in results) / total
        avg_time = sum(r.enrichment_time for r in results) / total
        
        # Count fields enriched
        field_counts = {}
        for result in results:
            for field in result.enriched_data:
                if field not in result.original_data:
                    field_counts[field] = field_counts.get(field, 0) + 1
                    
        # Most common errors
        error_counts = {}
        for result in results:
            for error in result.errors:
                error_counts[error] = error_counts.get(error, 0) + 1
                
        return {
            "total_processed": total,
            "successful": successful,
            "success_rate": round(successful / total * 100, 1),
            "average_confidence": round(avg_confidence, 2),
            "average_time": round(avg_time, 2),
            "most_enriched_fields": sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "common_errors": sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }