"""
Main FastAPI application for AI Lead Enhancer
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio
from functools import lru_cache

from ..models.lead_scorer import LeadScorer
from ..models.nlp_processor import NLPProcessor
from ..enrichment.enricher import LeadEnricher
from ..enrichment.data_sources import DataSourceManager
from . import API_VERSION, API_TITLE, API_DESCRIPTION, TAGS_METADATA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    openapi_tags=TAGS_METADATA,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class LeadInput(BaseModel):
    """Input model for lead data"""
    company_name: str = Field(..., description="Company name")
    website: Optional[HttpUrl] = Field(None, description="Company website URL")
    email: Optional[str] = Field(None, description="Contact email")
    phone: Optional[str] = Field(None, description="Contact phone")
    employee_count: Optional[int] = Field(None, description="Number of employees")
    industry: Optional[str] = Field(None, description="Industry sector")
    location: Optional[str] = Field(None, description="Company location")
    linkedin_url: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "TechCorp Inc",
                "website": "https://techcorp.com",
                "email": "contact@techcorp.com",
                "employee_count": 150,
                "industry": "Software",
                "location": "San Francisco, CA"
            }
        }

class LeadScore(BaseModel):
    """Output model for lead scoring"""
    lead_id: str
    score: int = Field(..., ge=0, le=100, description="Lead score (0-100)")
    grade: str = Field(..., description="Lead grade (A-F)")
    breakdown: Dict[str, float] = Field(..., description="Score breakdown by category")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level")
    recommendations: List[str] = Field(..., description="Action recommendations")
    enriched_data: Optional[Dict[str, Any]] = Field(None, description="Additional enriched data")
    processed_at: datetime

class BatchLeadRequest(BaseModel):
    """Request model for batch processing"""
    leads: List[LeadInput]
    enrich: bool = Field(True, description="Whether to enrich leads")
    async_processing: bool = Field(False, description="Process asynchronously")

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    version: str
    uptime: float
    services: Dict[str, str]

# Dependency injection
@lru_cache()
def get_lead_scorer() -> LeadScorer:
    """Get or create LeadScorer instance"""
    return LeadScorer()

@lru_cache()
def get_enricher() -> LeadEnricher:
    """Get or create LeadEnricher instance"""
    return LeadEnricher()

@lru_cache()
def get_nlp_processor() -> NLPProcessor:
    """Get or create NLPProcessor instance"""
    return NLPProcessor()

# API Endpoints
@app.get("/", tags=["health"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Lead Enhancer API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Check API health status"""
    return HealthCheck(
        status="healthy",
        version=API_VERSION,
        uptime=100.0,  # In production, calculate actual uptime
        services={
            "api": "operational",
            "scorer": "operational",
            "enricher": "operational",
            "nlp": "operational"
        }
    )

@app.post("/api/v1/score", response_model=LeadScore, tags=["scoring"])
async def score_lead(
    lead: LeadInput,
    enrich: bool = True,
    scorer: LeadScorer = Depends(get_lead_scorer),
    enricher: LeadEnricher = Depends(get_enricher)
):
    """
    Score a single lead with optional enrichment
    
    - **lead**: Lead information to score
    - **enrich**: Whether to enrich lead data before scoring
    """
    try:
        # Convert to dict for processing
        lead_data = lead.dict()
        
        # Enrich if requested
        if enrich and lead.website:
            logger.info(f"Enriching lead: {lead.company_name}")
            enriched = await enricher.enrich_lead_async(lead_data)
            lead_data.update(enriched)
        
        # Score the lead
        logger.info(f"Scoring lead: {lead.company_name}")
        score_result = scorer.score_lead(lead_data)
        
        # Generate recommendations
        recommendations = scorer.generate_recommendations(score_result)
        
        # Create response
        return LeadScore(
            lead_id=f"lead_{hash(lead.company_name)}",
            score=score_result["score"],
            grade=score_result["grade"],
            breakdown=score_result["breakdown"],
            confidence=score_result["confidence"],
            recommendations=recommendations,
            enriched_data=lead_data if enrich else None,
            processed_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error scoring lead: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/score/batch", tags=["scoring"])
async def score_batch(
    request: BatchLeadRequest,
    background_tasks: BackgroundTasks,
    scorer: LeadScorer = Depends(get_lead_scorer),
    enricher: LeadEnricher = Depends(get_enricher)
):
    """
    Score multiple leads in batch
    
    - **leads**: List of leads to process
    - **enrich**: Whether to enrich leads
    - **async_processing**: Process in background
    """
    try:
        if request.async_processing:
            # Process in background
            task_id = f"batch_{datetime.utcnow().timestamp()}"
            background_tasks.add_task(
                process_batch_async,
                request.leads,
                request.enrich,
                scorer,
                enricher
            )
            return {
                "status": "processing",
                "task_id": task_id,
                "message": f"Processing {len(request.leads)} leads in background"
            }
        else:
            # Process synchronously
            results = []
            for lead in request.leads:
                lead_data = lead.dict()
                
                if request.enrich and lead.website:
                    enriched = await enricher.enrich_lead_async(lead_data)
                    lead_data.update(enriched)
                
                score_result = scorer.score_lead(lead_data)
                results.append({
                    "company_name": lead.company_name,
                    "score": score_result["score"],
                    "grade": score_result["grade"]
                })
            
            return {
                "status": "completed",
                "processed": len(results),
                "results": results
            }
            
    except Exception as e:
        logger.error(f"Error in batch scoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/enrich", tags=["enrichment"])
async def enrich_lead(
    lead: LeadInput,
    enricher: LeadEnricher = Depends(get_enricher)
):
    """
    Enrich a lead with additional data
    
    - **lead**: Lead information to enrich
    """
    try:
        if not lead.website:
            raise HTTPException(
                status_code=400,
                detail="Website URL is required for enrichment"
            )
        
        lead_data = lead.dict()
        enriched = await enricher.enrich_lead_async(lead_data)
        
        return {
            "status": "success",
            "original": lead_data,
            "enriched": enriched,
            "sources": enricher.get_sources_used()
        }
        
    except Exception as e:
        logger.error(f"Error enriching lead: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/insights/{company_name}", tags=["enrichment"])
async def get_insights(
    company_name: str,
    nlp: NLPProcessor = Depends(get_nlp_processor)
):
    """
    Get AI-generated insights for a company
    
    - **company_name**: Name of the company
    """
    try:
        insights = nlp.generate_insights(company_name)
        return {
            "company": company_name,
            "insights": insights,
            "generated_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def process_batch_async(leads, enrich, scorer, enricher):
    """Process batch of leads asynchronously"""
    # Implementation for async batch processing
    pass

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("AI Lead Enhancer API starting up...")
    # Initialize services
    get_lead_scorer()
    get_enricher()
    get_nlp_processor()
    logger.info("All services initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("AI Lead Enhancer API shutting down...")
    # Cleanup resources

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)