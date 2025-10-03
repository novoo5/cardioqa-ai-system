"""
CardioQA Data Collection Module
Collects and processes medical Q&A data from MedQuAD dataset
Author: Novonil Basak
Date: October 2, 2025
"""

import os
import pandas as pd
import requests
from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDataCollector:
    """Collect and process medical datasets for CardioQA RAG system"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_medquad_dataset(self):
        """Collect MedQuAD dataset from HuggingFace"""
        logger.info("Starting MedQuAD dataset collection...")
        
        try:
            # Load MedQuAD dataset from HuggingFace
            logger.info("Loading MedQuAD from HuggingFace...")
            dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset['train'])
            logger.info(f"Loaded {len(df)} medical Q&A pairs")
            
            # Basic data inspection
            logger.info("Dataset columns: " + str(df.columns.tolist()))
            logger.info("Dataset shape: " + str(df.shape))
            
            # Save raw dataset
            raw_file_path = self.data_dir / "medquad_raw.csv"
            df.to_csv(raw_file_path, index=False)
            logger.info(f"Saved raw MedQuAD to {raw_file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting MedQuAD dataset: {str(e)}")
            return None
            
    def filter_cardiac_data(self, df):
        """Filter dataset for cardiology-related content"""
        logger.info("Filtering for cardiology-related content...")
        
        # Cardiac-related keywords
        cardiac_keywords = [
            'heart', 'cardiac', 'cardiology', 'cardiovascular', 'coronary',
            'arrhythmia', 'hypertension', 'blood pressure', 'chest pain',
            'heart attack', 'myocardial', 'atrial', 'ventricular', 'valve',
            'pacemaker', 'ECG', 'EKG', 'angina', 'stroke', 'circulation'
        ]
        
        # Create cardiac filter mask
        cardiac_mask = df.apply(
            lambda row: any(
                keyword.lower() in str(row).lower() 
                for keyword in cardiac_keywords
            ), axis=1
        )
        
        cardiac_df = df[cardiac_mask].copy()
        logger.info(f"Found {len(cardiac_df)} cardiac-related Q&A pairs")
        
        # Save filtered cardiac data
        cardiac_file_path = self.data_dir / "medquad_cardiac.csv"
        cardiac_df.to_csv(cardiac_file_path, index=False)
        logger.info(f"Saved cardiac data to {cardiac_file_path}")
        
        return cardiac_df
        
    def display_sample_data(self, df, n_samples=3):
        """Display sample Q&A pairs"""
        logger.info(f"Sample {n_samples} Q&A pairs:")
        print("\n" + "="*80)
        
        for i, row in df.head(n_samples).iterrows():
            print(f"Q{i+1}: {row.iloc[0] if len(row) > 0 else 'No question'}")
            print(f"A{i+1}: {row.iloc[1] if len(row) > 1 else 'No answer'}")
            print("-" * 60)
        
    def get_dataset_statistics(self, df):
        """Generate basic statistics about the dataset"""
        stats = {
            'total_pairs': len(df),
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Save statistics
        stats_file = self.data_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        logger.info("Dataset Statistics:")
        logger.info(f"- Total Q&A pairs: {stats['total_pairs']}")
        logger.info(f"- Columns: {stats['columns']}")
        logger.info(f"- Statistics saved to {stats_file}")
        
        return stats

def main():
    """Main execution function"""
    print("🫀 CardioQA Data Collection Pipeline")
    print("=" * 50)
    
    # Initialize collector
    collector = MedicalDataCollector()
    
    # Step 1: Collect MedQuAD dataset
    print("\n📊 Step 1: Collecting MedQuAD Dataset...")
    medquad_df = collector.collect_medquad_dataset()
    
    if medquad_df is not None:
        # Step 2: Generate statistics
        print("\n📈 Step 2: Analyzing Dataset...")
        stats = collector.get_dataset_statistics(medquad_df)
        
        # Step 3: Display samples
        print("\n👀 Step 3: Sample Data Preview...")
        collector.display_sample_data(medquad_df, n_samples=3)
        
        # Step 4: Filter cardiac data
        print("\n🫀 Step 4: Filtering Cardiac Data...")
        cardiac_df = collector.filter_cardiac_data(medquad_df)
        
        # Step 5: Display cardiac samples
        if len(cardiac_df) > 0:
            print("\n💓 Step 5: Cardiac Data Preview...")
            collector.display_sample_data(cardiac_df, n_samples=2)
        
        print("\n✅ Data collection completed successfully!")
        print(f"📁 Files saved in: {collector.data_dir}")
        
    else:
        print("❌ Data collection failed!")

if __name__ == "__main__":
    main()
