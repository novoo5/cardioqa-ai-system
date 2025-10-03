"""
CardioQA FastAPI Backend - PRODUCTION RENDER VERSION
AI-powered cardiac diagnostic assistant with RAG
Author: Novonil Basak
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
collection = None
embedding_model = None
gemini_model = None
safety_validator = None

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

class MedicalSafetyValidator:
    """Medical safety validation system"""
    
    def __init__(self):
        self.emergency_keywords = [
            'heart attack', 'chest pain', 'shortness of breath', 'stroke',
            'severe pain', 'bleeding', 'unconscious', 'emergency', 'crushing pain'
        ]
    
    def validate_response(self, response_text: str, user_query: str) -> dict:
        """Validate medical safety of AI response"""
        safety_score = 85  # Start at 85
        warnings = []
        
        # Check for emergency situations
        if any(keyword in user_query.lower() for keyword in self.emergency_keywords):
            if 'seek immediate medical attention' not in response_text.lower():
                warnings.append("CRITICAL: Emergency situation detected")
                safety_score -= 20
            else:
                safety_score += 10
        
        # Check for professional consultation recommendation
        consult_phrases = ['consult', 'doctor', 'physician', 'healthcare provider']
        if any(phrase in response_text.lower() for phrase in consult_phrases):
            safety_score += 10
        else:
            warnings.append("Added professional consultation recommendation")
            safety_score -= 15
        
        # Check response quality
        if len(response_text) > 200:
            safety_score += 5
        
        # Check for dangerous statements
        dangerous_phrases = ['you definitely have', 'this is certainly', 'never see a doctor']
        if any(phrase in response_text.lower() for phrase in dangerous_phrases):
            warnings.append("Contains potentially dangerous medical statements")
            safety_score -= 25
        
        safety_score = min(100, max(50, safety_score))
        
        return {
            'safety_score': safety_score,
            'warnings': warnings,
            'is_safe': safety_score >= 70
        }
    
    def add_safety_disclaimers(self, response_text: str, safety_check: dict) -> str:
        """Add medical disclaimers"""
        disclaimers = "\n\n⚠️ **MEDICAL DISCLAIMER**: Educational purposes only.\n👨‍⚕️ **RECOMMENDATION**: Consult healthcare professionals."
        
        if safety_check['safety_score'] < 80:
            disclaimers += "\n🚨 **IMPORTANT**: For severe symptoms, seek immediate medical attention."
        
        return response_text + disclaimers

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global collection, embedding_model, gemini_model, safety_validator
    
    logger.info("🫀 Starting CardioQA API...")
    
    try:
        # Find ChromaDB in deployment environment
        possible_paths = [
            "./chroma_db",
            "chroma_db", 
            "/opt/render/project/src/chroma_db",
            Path.cwd() / "chroma_db",
            Path(__file__).parent.parent.parent / "chroma_db"
        ]
        
        db_path = None
        for path in possible_paths:
            path_obj = Path(path)
            logger.info(f"🔍 Checking: {path_obj.absolute()}")
            if path_obj.exists() and path_obj.is_dir():
                db_path = str(path_obj)
                logger.info(f"✅ Found ChromaDB at: {db_path}")
                break
        
        if not db_path:
            # Debug current environment
            current_dir = Path.cwd()
            logger.info(f"📂 Current working directory: {current_dir}")
            logger.info(f"📋 Files in current directory: {[f.name for f in current_dir.iterdir()]}")
            
            # Search recursively for chroma_db
            for chroma_path in current_dir.rglob("chroma_db"):
                if chroma_path.is_dir():
                    logger.info(f"🔍 Found chroma_db via search: {chroma_path}")
                    db_path = str(chroma_path)
                    break
            
            if not db_path:
                raise Exception(f"ChromaDB not found. Searched in: {[str(p) for p in possible_paths]}")
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name="cardiac_knowledge")
        logger.info(f"✅ Loaded vector database: {collection.count()} documents")
        
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Loaded embedding model")
        
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise Exception("❌ GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Test Gemini connection
        test_response = gemini_model.generate_content("Say 'CardioQA API ready!'")
        logger.info(f"✅ Gemini test: {test_response.text}")
        
        # Initialize safety validator
        safety_validator = MedicalSafetyValidator()
        logger.info("✅ Safety validator ready")
        
        logger.info("🎉 CardioQA API fully initialized!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    # Cleanup (if needed)
    logger.info("🔄 Shutting down CardioQA API...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="CardioQA API",
    description="AI-powered cardiac diagnostic assistant with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "CardioQA API - AI-Powered Cardiac Diagnostic Assistant",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "docs": "/docs",
            "stats": "/stats"
        }
    }

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
            "api_version": "1.0.0",
            "deployment": "production"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_cardioqa(request: QueryRequest):
    """Main CardioQA query endpoint"""
    start_time = time.time()
    
    try:
        if not collection or not gemini_model or not safety_validator:
            raise HTTPException(status_code=503, detail="System not fully initialized")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Search knowledge base
        results = collection.query(
            query_texts=[request.query],
            n_results=3
        )
        
        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail="No relevant cardiac information found")
        
        # Format knowledge context
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
        
        # Create medical prompt
        context_text = f"Medical Evidence:\nQ: {knowledge_context[0]['question']}\nA: {knowledge_context[0]['answer']}"
        
        prompt = f"""You are CardioQA, a specialized cardiac health assistant.

MEDICAL RESPONSE RULES:
- Never provide definitive diagnoses
- Always recommend consulting healthcare professionals
- Use **bold** for important medical points
- Be educational and evidence-based
- Include appropriate medical caution

USER QUESTION: {request.query}

{context_text}

Provide a helpful, evidence-based response with proper **bold** formatting:"""
        
        # Generate AI response
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,
                'max_output_tokens': 800,
            }
        )
        ai_response = response.text
        
        # Apply safety validation
        safety_check = safety_validator.validate_response(ai_response, request.query)
        safe_response = safety_validator.add_safety_disclaimers(ai_response, safety_check)
        
        # Calculate confidence level
        similarity = knowledge_context[0]['similarity']
        if similarity > 0.6:
            confidence = 'High'
        elif similarity > 0.4:
            confidence = 'Medium'
        elif similarity > 0.2:
            confidence = 'Low'
        else:
            confidence = 'Very Low'
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """System statistics endpoint"""
    try:
        return {
            "total_documents": collection.count() if collection else 0,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gemini-2.0-flash",
            "specialty": "cardiology",
            "safety_features": [
                "emergency_detection",
                "professional_consultation", 
                "medical_disclaimers",
                "confidence_scoring"
            ],
            "deployment": "render-production"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
