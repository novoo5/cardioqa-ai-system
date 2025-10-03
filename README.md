 # 🫀 CardioQA - AI-Powered Cardiac Diagnostic Assistant

A specialized RAG (Retrieval-Augmented Generation) system for cardiac health education and information, built with ChromaDB and Google Gemini 2.0.

## 🎯 Project Overview

CardioQA is a production-ready healthcare AI system that provides evidence-based cardiac health information using:
- **364 curated cardiac Q&A pairs** from MedQuAD medical dataset
- **ChromaDB vector database** with semantic search capabilities
- **Google Gemini 2.0 Flash** for intelligent response generation
- **Medical safety validation** with emergency detection
- **Professional medical disclaimers** and consultation recommendations

## 🏗️ Technical Architecture

### RAG Pipeline
1. **Data Collection**: MedQuAD dataset → Cardiac filtering → 364 Q&As
2. **Vector Database**: SentenceTransformers → ChromaDB → Semantic search
3. **Response Generation**: Gemini 2.0 → Safety validation → Medical disclaimers
4. **Interactive Interface**: Chat-based system with confidence scoring

### Technology Stack
- **Vector Database**: ChromaDB (persistent storage)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini 2.0 Flash
- **Data Processing**: Pandas, NumPy
- **Development**: Jupyter Notebooks, Python 3.11

## 📊 Performance Metrics

- **Knowledge Base**: 364 cardiac-specific Q&A pairs
- **Response Quality**: 38-55% similarity scores for relevant queries
- **Safety Score**: 70-100/100 with emergency detection
- **Response Time**: 2-3 seconds average
- **Confidence Levels**: High/Medium/Low based on similarity

## 🛡️ Safety Features

- Emergency keyword detection
- Professional consultation recommendations
- Medical disclaimer automation
- Dangerous advice prevention
- Confidence-based response scaling

## 🚀 Getting Started

1. Install requirements: `pip install -r requirements.txt`
2. Run notebook 1: Data collection and filtering
3. Run notebook 2: Build RAG system and vector database
4. Run notebook 3: Complete system with Gemini integration

## 📈 Business Impact

- **Educational Tool**: Evidence-based cardiac health information
- **Time Savings**: Instant access to verified medical knowledge
- **Safety-First Design**: Production-ready medical AI with safety validation
- **Scalable Architecture**: Ready for additional medical specialties

## 👨‍💻 Author

**Novonil Basak** - B.Sc. Biotechnology Student
- Specialized in Healthcare AI and Machine Learning
- Focus: Medical data analysis and diagnostic prediction models

