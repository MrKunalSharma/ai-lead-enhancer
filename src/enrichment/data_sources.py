"""
Data Sources Module for Lead Enrichment

Manages various data sources including web scraping, APIs, and databases
for enriching lead information.
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import re
from bs4 import BeautifulSoup
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    async def fetch_data(self, identifier: str) -> Dict[str, Any]:
        """Fetch data from the source"""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available"""
        pass


class CacheManager:
    """Manages caching for data sources"""
    
    def __init__(self, ttl: int = 86400, max_size: int = 10000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}
        self.timestamps = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Check if expired
            if datetime.utcnow() - self.timestamps[key] > timedelta(seconds=self.ttl):
                del self.cache[key]
                del self.timestamps[key]
                return None
            return self.cache[key]
        return None
        
    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Implement LRU if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
            
        self.cache[key] = value
        self.timestamps[key] = datetime.utcnow()
        
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.timestamps.clear()


class WebScraperSource(DataSource):
    """Web scraping data source"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.cache = CacheManager()
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 10))
            connector = aiohttp.TCPConnector(limit=100)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': self.config.get("user_agent", "Mozilla/5.0")
                }
            )
        return self.session
        
    async def fetch_data(self, url: str) -> Dict[str, Any]:
        """Fetch and parse website data"""
        # Check cache
        cached = self.cache.get(url)
        if cached:
            return cached
            
        try:
            session = await self._get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Got status {response.status} for {url}")
                    return {}
                    
                html = await response.text()
                
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            data = {
                "url": url,
                "title": self._extract_title(soup),
                "meta_data": self._extract_meta_data(soup),
                "social_links": self._extract_social_links(soup, url),
                "contact_info": self._extract_contact_info(soup),
                "technology_hints": self._extract_tech_hints(html),
                "html_content": html[:5000]  # First 5000 chars for tech detection
            }
            
            # Cache result
            self.cache.set(url, data)
            
            return data
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return {}
            
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title = soup.find('title')
        return title.text.strip() if title else ""
        
    def _extract_meta_data(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract meta tags"""
        meta_data = {}
        
        # Standard meta tags
        for tag in soup.find_all('meta'):
            name = tag.get('name', tag.get('property', ''))
            content = tag.get('content', '')
            
            if name and content:
                if name in ['description', 'keywords', 'author']:
                    meta_data[name] = content
                elif name.startswith('og:'):
                    meta_data[name] = content
                    
        return meta_data
        
    def _extract_social_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
        """Extract social media links"""
        social_patterns = {
            'linkedin': r'linkedin\.com/company/([^/\s]+)',
            'twitter': r'twitter\.com/([^/\s]+)',
            'facebook': r'facebook\.com/([^/\s]+)',
            'github': r'github\.com/([^/\s]+)',
            'youtube': r'youtube\.com/(?:c/|channel/|user/)([^/\s]+)'
        }
        
        social_links = {}
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            for platform, pattern in social_patterns.items():
                match = re.search(pattern, href)
                if match and platform not in social_links:
                    social_links[platform] = urljoin(base_url, href)
                    
        return social_links
        
    def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract contact information"""
        contact = {}
        
        # Email patterns
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, soup.get_text())
        if emails:
            # Filter out common non-contact emails
            contact_emails = [e for e in emails if not any(x in e for x in ['example.', 'test.', '@2x.'])]
            if contact_emails:
                contact['emails'] = list(set(contact_emails))[:3]  # Limit to 3
                
        # Phone patterns (US format)
        phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
        phones = re.findall(phone_pattern, soup.get_text())
        if phones:
            contact['phones'] = list(set(phones))[:3]
            
        # Address - look for common patterns
        address_keywords = ['address', 'location', 'headquarters', 'office']
        for keyword in address_keywords:
            elements = soup.find_all(text=re.compile(keyword, re.I))
            for element in elements:
                parent = element.parent
                if parent:
                    address_text = parent.get_text(strip=True)
                    if len(address_text) > 10 and len(address_text) < 200:
                        contact['address'] = address_text
                        break
                        
        return contact
        
    def _extract_tech_hints(self, html: str) -> List[str]:
        """Extract technology hints from HTML"""
        tech_hints = []
        
        # Common technology indicators
        tech_patterns = {
            'WordPress': ['wp-content', 'wp-includes', 'wordpress'],
            'Shopify': ['cdn.shopify', 'shopify.com'],
            'React': ['react', 'reactdom', '_jsx'],
            'Angular': ['ng-version', 'angular', 'ng-app'],
            'Vue.js': ['vue.js', 'vuejs'],
            'jQuery': ['jquery.min.js', 'jquery-'],
            'Bootstrap': ['bootstrap.min.css', 'bootstrap.js'],
            'Google Analytics': ['google-analytics.com', 'gtag.js'],
            'Google Tag Manager': ['googletagmanager.com'],
            'Cloudflare': ['cloudflare.com', 'cf-ray']
        }
        
        html_lower = html.lower()
        
        for tech, patterns in tech_patterns.items():
            if any(pattern in html_lower for pattern in patterns):
                tech_hints.append(tech)
                
        return tech_hints
        
    def is_available(self) -> bool:
        """Check if web scraping is available"""
        return self.config.get("enabled", True)
        
    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()


class APIDataSource(DataSource):
    """External API data source"""
    
    def __init__(self, provider: str, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.base_urls = {
            "clearbit": "https://company.clearbit.com/v2/companies/domain/",
            "hunter": "https://api.hunter.io/v2/domain-search",
            "builtwith": "https://api.builtwith.com/v14/api.json"
        }
        self.cache = CacheManager()
        
    async def fetch_data(self, identifier: str) -> Dict[str, Any]:
        """Fetch data from API"""
        # Check cache
        cache_key = f"{self.provider}:{identifier}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        # Mock implementation - in production, would make actual API calls
        if self.provider == "clearbit":
            data = await self._fetch_clearbit_data(identifier)
        elif self.provider == "hunter":
            data = await self._fetch_hunter_data(identifier)
        elif self.provider == "builtwith":
            data = await self._fetch_builtwith_data(identifier)
        else:
            data = {}
            
        # Cache result
        if data:
            self.cache.set(cache_key, data)
            
        return data
        
    async def _fetch_clearbit_data(self, domain: str) -> Dict[str, Any]:
        """Mock Clearbit API response"""
        # In production, would make actual API call
        return {
            "name": "Example Company",
            "domain": domain,
            "industry": "Technology",
            "employeeCount": 150,
            "foundedYear": 2015,
            "location": {
                "city": "San Francisco",
                "state": "CA",
                "country": "US"
            },
            "techStack": ["AWS", "React", "Node.js"],
            "funding": {
                "total": 10000000,
                "lastRound": "Series A"
            }
        }
        
    async def _fetch_hunter_data(self, domain: str) -> Dict[str, Any]:
        """Mock Hunter.io API response"""
        return {
            "domain": domain,
            "emails": [
                {
                    "email": f"contact@{domain}",
                    "type": "generic",
                    "confidence": 90
                }
            ],
            "pattern": "{first}.{last}@{domain}"
        }
        
    async def _fetch_builtwith_data(self, domain: str) -> Dict[str, Any]:
        """Mock BuiltWith API response"""
        return {
            "domain": domain,
            "technologies": [
                {"name": "Google Analytics", "category": "Analytics"},
                {"name": "WordPress", "category": "CMS"},
                {"name": "MySQL", "category": "Database"}
            ]
        }
        
    def is_available(self) -> bool:
        """Check if API is available"""
        return bool(self.api_key) if self.provider != "mock" else True


class DatabaseSource(DataSource):
    """Internal database data source"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string
        self.cache = CacheManager(ttl=300)  # 5 minute cache
        
    async def fetch_data(self, identifier: str) -> Dict[str, Any]:
        """Fetch data from database"""
        # Mock implementation
        cached = self.cache.get(identifier)
        if cached:
            return cached
            
        # In production, would query actual database
        data = {
            "previous_interactions": [
                {
                    "date": "2024-01-15",
                    "type": "email",
                    "outcome": "no_response"
                }
            ],
            "internal_score": 75,
            "assigned_to": "sales_team_a"
        }
        
        self.cache.set(identifier, data)
        return data
        
    def is_available(self) -> bool:
        """Check if database is available"""
        # In production, would check actual database connection
        return True


class DataSourceManager:
    """Manages multiple data sources for enrichment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.web_scraper = WebScraperSource(config["sources"]["web_scraping"])
        self.api_sources = self._initialize_api_sources()
        self.database = DatabaseSource()
        
    def _initialize_api_sources(self) -> Dict[str, APIDataSource]:
        """Initialize API data sources"""
        sources = {}
        api_config = self.config["sources"]["apis"]
        
        if api_config["enabled"]:
            for provider in api_config["providers"]:
                # In production, would load actual API keys from config
                sources[provider] = APIDataSource(provider, api_key="mock_key")
                
        return sources
        
    async def scrape_website(self, url: str) -> Dict[str, Any]:
        """Scrape website for data"""
        if not self.web_scraper.is_available():
            return {}
            
        return await self.web_scraper.fetch_data(url)
        
    async def fetch_from_apis(self, domain: str) -> Dict[str, Any]:
        """Fetch data from all available APIs"""
        results = {}
        
        for provider, source in self.api_sources.items():
            if source.is_available():
                try:
                    data = await source.fetch_data(domain)
                    if data:
                        results[provider] = data
                except Exception as e:
                    logger.error(f"Error fetching from {provider}: {e}")
                    
        return results
        
    async def fetch_from_database(self, company_name: str) -> Dict[str, Any]:
        """Fetch data from database"""
        if not self.database.is_available():
            return {}
            
        return await self.database.fetch_data(company_name)
        
    async def enrich_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate enrichment from all sources"""
        enriched = {}
        tasks = []
        
        # Web scraping
        if lead_data.get("website"):
            tasks.append(self.scrape_website(lead_data["website"]))
            
        # API enrichment
        if lead_data.get("website"):
            domain = urlparse(lead_data["website"]).netloc
            tasks.append(self.fetch_from_apis(domain))
            
        # Database enrichment
        if lead_data.get("company_name"):
            tasks.append(self.fetch_from_database(lead_data["company_name"]))
            
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, dict):
                enriched.update(result)
                
        return enriched
        
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
        
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        # Remove non-numeric characters
        cleaned = re.sub(r'\D', '', phone)
        
        # Check if it's a valid length (10-15 digits)
        return 10 <= len(cleaned) <= 15
        
    def validate_website(self, url: str) -> bool:
        """Validate website URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
            
    def normalize_industry(self, industry: str) -> str:
        """Normalize industry names"""
        industry_mapping = {
            "tech": "Technology",
            "software": "Technology",
            "it": "Technology",
            "fintech": "Finance",
            "financial services": "Finance",
            "banking": "Finance",
            "health": "Healthcare",
            "medical": "Healthcare",
            "pharma": "Healthcare",
            "ecommerce": "E-commerce",
            "retail": "E-commerce",
            "online retail": "E-commerce"
        }
        
        industry_lower = industry.lower().strip()
        return industry_mapping.get(industry_lower, industry.title())
        
    def estimate_company_size(self, employee_count: Optional[int]) -> str:
        """Estimate company size category"""
        if not employee_count:
            return "Unknown"
            
        if employee_count <= 10:
            return "Micro"
        elif employee_count <= 50:
            return "Small"
        elif employee_count <= 250:
            return "Medium"
        elif employee_count <= 1000:
            return "Large"
        else:
            return "Enterprise"
            
    async def close(self):
        """Clean up resources"""
        await self.web_scraper.close()