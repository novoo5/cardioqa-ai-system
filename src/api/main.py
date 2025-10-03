"""
CardioQA FastAPI Backend - FIXED VERSION
Production-ready API for cardiac diagnostic assistant
Author: Novonil Basak
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import logging
import os
from pathlib import Path
from typing import List, Optional
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../../.env")  # Load from root directory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)
    include_metadata: bool = Field(default=True)

class QueryResponse(BaseModel):
    response: str
    safety_score: int
    confidence: str
    knowledge_sources: int
    top_similarity: float
    warnings: List[str]
    response_time: float

# Initialize FastAPI app
app = FastAPI(
    title="CardioQA API",
    description="AI-powered cardiac diagnostic assistant with RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global variables
collection = None
embedding_model = None
gemini_model = None
safety_validator = None

class MedicalSafetyValidator:
    """Medical safety validation - FIXED VERSION"""
    
    def __init__(self):
        self.emergency_keywords = [
            'heart attack', 'chest pain', 'shortness of breath', 'stroke',
            'severe pain', 'bleeding', 'unconscious', 'emergency', 'crushing pain'
        ]
    
    def validate_response(self, response_text: str, user_query: str) -> dict:
        """Validate medical safety of AI response - FIXED SCORING"""
        safety_score = 85  # Start at 85 instead of 100
        warnings = []
        
        # Check for emergency situations
        if any(keyword in user_query.lower() for keyword in self.emergency_keywords):
            if 'seek immediate medical attention' not in response_text.lower():
                warnings.append("CRITICAL: Emergency situation detected")
                safety_score -= 20
            else:
                safety_score += 10  # Bonus for including emergency advice
        
        # Check for professional consultation recommendation
        consult_phrases = ['consult', 'doctor', 'physician', 'healthcare provider']
        if any(phrase in response_text.lower() for phrase in consult_phrases):
            safety_score += 10  # Bonus for recommending consultation
        else:
            warnings.append("Added professional consultation recommendation")
            safety_score -= 15
        
        # Check response quality and detail
        if len(response_text) > 200:
            safety_score += 5  # Bonus for detailed responses
        
        # Check for dangerous absolute statements
        dangerous_phrases = ['you definitely have', 'this is certainly', 'never see a doctor']
        if any(phrase in response_text.lower() for phrase in dangerous_phrases):
            warnings.append("Contains potentially dangerous medical statements")
            safety_score -= 25
        
        # Ensure score stays in reasonable range
        safety_score = min(100, max(50, safety_score))
        
        return {
            'safety_score': safety_score,
            'warnings': warnings,
            'is_safe': safety_score >= 70
        }
    
    def add_safety_disclaimers(self, response_text: str, safety_check: dict) -> str:
        """Add medical disclaimers to response"""
        disclaimers = "\n\n⚠️ **MEDICAL DISCLAIMER**: Educational purposes only.\n👨‍⚕️ **RECOMMENDATION**: Consult healthcare professionals."
        
        if safety_check['safety_score'] < 80:
            disclaimers += "\n🚨 **IMPORTANT**: For severe symptoms, seek immediate medical attention."
        
        return response_text + disclaimers

@app.on_event("startup")
async def startup_event():
    """Initialize models and database"""
    global collection, embedding_model, gemini_model, safety_validator
    
    logger.info("🫀 Starting CardioQA API...")
    
    try:
        # Load vector database from root directory
        db_path = "../../chroma_db"
        logger.info(f"Looking for database at: {Path(db_path).absolute()}")
        
        if not Path(db_path).exists():
            logger.error(f"❌ Database not found at {db_path}")
            raise Exception("ChromaDB not found - run the notebooks first!")
        
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name="cardiac_knowledge")
        logger.info(f"✅ Loaded vector database: {collection.count()} documents")
        
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Loaded embedding model")
        
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY not found in environment")
            # Try hardcoded for testing
            api_key = "AIzaSyDG8JAqtiA6wqeypFoyNziduq7DYKWGu78"
            logger.info("Using hardcoded API key for testing")
        
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Test Gemini connection
        test_response = gemini_model.generate_content("Say 'API ready!'")
        logger.info(f"✅ Gemini test: {test_response.text}")
        
        # Initialize safety validator
        safety_validator = MedicalSafetyValidator()
        logger.info("✅ Safety validator ready")
        
        logger.info("🎉 CardioQA API fully initialized!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_count = collection.count() if collection else 0
        model_status = "ready" if gemini_model else "not loaded"
        
        return {
            "status": "healthy",
            "database_count": db_count,
            "model_status": model_status,
            "api_version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_cardioqa(request: QueryRequest):
    """Main CardioQA query endpoint - FIXED CONFIDENCE CALCULATION"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Search knowledge base
        results = collection.query(
            query_texts=[request.query],
            n_results=3
        )
        
        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        # Format context
        knowledge_context = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            knowledge_context.append({
                'question': metadata['question'],
                'answer': metadata['answer'],
                'similarity': 1 - distance
            })
        
        # Create enhanced prompt
        context_text = f"Medical Evidence:\nQ: {knowledge_context[0]['question']}\nA: {knowledge_context[0]['answer']}"
        
        prompt = f"""You are CardioQA, a cardiac health assistant.

RULES:
- Never provide definitive diagnoses
- Always recommend consulting doctors
- Use **bold** for important points
- Be educational and helpful
- Format with clear headings using **

Question: {request.query}
{context_text}

Provide helpful, evidence-based information with proper **bold** formatting for key points:"""
        
        # Generate response
        response = gemini_model.generate_content(prompt)
        ai_response = response.text
        
        # Apply safety validation
        safety_check = safety_validator.validate_response(ai_response, request.query)
        safe_response = safety_validator.add_safety_disclaimers(ai_response, safety_check)
        
        # FIXED CONFIDENCE CALCULATION
        similarity = knowledge_context[0]['similarity']
        if similarity > 0.6:
            confidence = 'High'
        elif similarity > 0.4:
            confidence = 'Medium' 
        elif similarity > 0.2:
            confidence = 'Low'
        else:
            confidence = 'Very Low'
        
        # Return response
        response_time = time.time() - start_time
        
        return QueryResponse(
            response=safe_response,
            safety_score=safety_check['safety_score'],
            confidence=confidence,
            knowledge_sources=len(knowledge_context),
            top_similarity=knowledge_context[0]['similarity'],
            warnings=safety_check['warnings'],
            response_time=round(response_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
