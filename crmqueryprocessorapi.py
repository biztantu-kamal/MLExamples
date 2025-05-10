from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import uvicorn
from typing import Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crm_insights.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class InsightRequest(BaseModel):
    query: str

class InsightResponse(BaseModel):
    insights: str
    timestamp: str

# Initialize FastAPI app
app = FastAPI(title="CRM Insights API")

# Import the CRMQueryProcessor
from crm_query_processor import CRMQueryProcessor

# Initialize the processor
processor = CRMQueryProcessor()

@app.post("/insights", response_model=InsightResponse)
async def get_insights(request: InsightRequest):
    try:
        logger.info(f"Received query: {request.query}")
        
        # Process the query
        result = processor.process_query(request.query)
        
        # Log the SQL query and results
        if "sql" in result:
            logger.info(f"Generated SQL: {result['sql']}")
        
        if "results" in result:
            logger.info(f"Query returned {len(result['results'])} results")
        
        if "error" in result:
            logger.error(f"Error processing query: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        return InsightResponse(
            insights=result.get("insights", "No insights generated"),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
