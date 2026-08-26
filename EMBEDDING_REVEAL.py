import os 
import vec2text
import openai
import torch
import requests
import urllib3
import pickle
import hashlib
import numpy as np
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from Security_Methods import AESHelper, Encryptor, Decryptor
from Cryptodome.Cipher import AES
import base64
import logging
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
from rouge import Rouge

import random

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Embedding Security Evaluation Experiment')
parser.add_argument('--run-dir', type=str, default="./",
                    help='Project root directory path')
parser.add_argument('--type-of-dataset', type=str, default="HealthCareMagic",
                    help='Dataset type (HealthCareMagic, ErionEmails, billsum)')
parser.add_argument('--api-base', type=str, default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                    help='OpenAI API base URL')
parser.add_argument('--api-key', type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                    help='OpenAI API key')
parser.add_argument('--embedding-model', type=str, default="text-embedding-ada-002",
                    help='Embedding model name')
parser.add_argument('--gpu', type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                    help='GPU number to use')
parser.add_argument('--num-samples', type=int, default=5,
                    help='Number of samples to randomly select from the dataset')
parser.add_argument("--rag-method", type=str, choices=["A", "B"], 
                    default="A", help="RAG method to use (A or B)")
parser.add_argument('--output', type=str, default=None,
                    help='Output file name for results')
args = parser.parse_args()

# Set environment variables
os.environ["OPENAI_API_BASE"] = args.api_base
os.environ["OPENAI_API_KEY"] = args.api_key
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from API_chat import Chat, Embedding



# Encryption and conversion functions
def text_to_fake_embedding(text, dim=1536):
    """Encrypt text and convert to fake embedding vector"""
    # Generate key and IV using AESHelper
    key = AESHelper.generate_key(16)
    iv = AESHelper.generate_iv()
    
    # Create encryptor and encrypt text
    cbc_encryptor = Encryptor(key, mode=AES.MODE_CBC, iv=iv)
    # Convert text to bytes
    text_bytes = text.encode('utf-8') if isinstance(text, str) else text
    _, ciphertext = cbc_encryptor.encrypt(text_bytes)
    
    # Combine key, IV and ciphertext to ensure decryption is possible
    combined_data = {
        'key': key,
        'iv': iv,
        'ciphertext': ciphertext
    }
    
    # Use pickle to serialize encryption result to byte stream
    pickled_data = pickle.dumps(combined_data)
    
    # Convert byte stream to numeric array
    byte_array = np.frombuffer(pickled_data, dtype=np.uint8)
    
    # Extend/truncate to required dimension
    if len(byte_array) >= dim:
        # If array is too long, truncate
        byte_array = byte_array[:dim]
    else:
        # If array is too short, extend by repeating
        repeats = dim // len(byte_array) + 1
        byte_array = np.tile(byte_array, repeats)[:dim]
    
    # Convert to floating point and normalize
    fake_embedding = byte_array.astype(np.float32)
    
    # Normalize vector
    norm = np.linalg.norm(fake_embedding)
    if norm > 0:
        fake_embedding = fake_embedding / norm
    
    return fake_embedding

# Calculate ROUGE scores
def calculate_rouge(original_text, recovered_text):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(recovered_text, original_text)
        return scores[0]
    except Exception as e:
        logger.error(f"Error calculating ROUGE scores: {e}")
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}

# Load test samples from file
def load_test_samples(run_dir, dataset_type, num_samples=5):
    """Load test samples from JSON file"""
    file_path = os.path.join(run_dir, f"Evaluation_Materials/Private_data_{dataset_type.lower()}_Method{args.rag_method}.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        
        logger.info(f"Successfully loaded {len(all_samples)} samples")
        
        # If there are enough samples, randomly select the specified number
        if len(all_samples) > num_samples:
            return random.sample(all_samples, num_samples)
        return all_samples
    except Exception as e:
        logger.error(f"Error loading test samples: {e}")
        # If loading fails, return default samples
        return [
            "Jack Morris is a PhD student at Cornell Tech in New York City",
            "It was the best of times, it was the worst of times, it was the age of wisdom",
            "The quick brown fox jumps over the lazy dog",
            "Artificial intelligence is intelligence demonstrated by machines",
            "Machine learning is a method of data analysis that automates analytical model building"
        ]

def save_results(results, run_dir, dataset_type, output_name=None):
    """Save experiment results to JSON file"""
    result_dir = os.path.join(run_dir, "Results")
    os.makedirs(result_dir, exist_ok=True)
    
    if output_name:
        filename = output_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embedding_security_{dataset_type.lower()}_{timestamp}.json"
    
    output_path = os.path.join(result_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_path}")
    return output_path

def main():
    # Print initial information
    logger.info(f"\n{'='*80}")
    logger.info(f"Embedding Security Experiment - Dataset: {args.type_of_dataset}")
    logger.info(f"{'='*80}\n")
    
    # Load test samples
    test_samples = load_test_samples(args.run_dir, args.type_of_dataset, args.num_samples)
    
    # Create result dictionary
    results = {
        "metadata": {
            "dataset": args.type_of_dataset,
            "embedding_model": args.embedding_model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(test_samples)
        },
        "samples": [],
        "summary": {}
    }
    
    try:
        # Load corrector model
        corrector = vec2text.load_pretrained_corrector(args.embedding_model)
        logger.info("Model loaded successfully!")
        
        # Conduct two experimental comparisons
        logger.info("\nExperimental Results Comparison:")
        logger.info("-" * 80)
        logger.info("Experiment 1: Real Embedding Inversion")
        logger.info("-" * 80)
        
        # Get real embeddings
        embed_client = Embedding(api_base=args.api_base, 
                               api_key=args.api_key,
                               model=args.embedding_model)
        
        real_embeddings = embed_client.encode(test_samples)
        real_embeddings = torch.tensor(real_embeddings)
        
        # Invert real embeddings
        real_results = vec2text.invert_embeddings(
            embeddings=real_embeddings.cuda(),
            corrector=corrector
        )
        
        # Print real embedding results and evaluation
        real_rouge_scores = []
        for i, (original, result) in enumerate(zip(test_samples, real_results)):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Original text: {original}")
            logger.info(f"Recovered text: {result}")
            scores = calculate_rouge(original, result)
            real_rouge_scores.append(scores)
            logger.info(f"ROUGE-1: {scores['rouge-1']['f']:.4f}, ROUGE-2: {scores['rouge-2']['f']:.4f}, ROUGE-L: {scores['rouge-l']['f']:.4f}")
        
        logger.info("\n" + "-" * 80)
        logger.info("Experiment 2: Encrypted Fake Embedding Inversion")
        logger.info("-" * 80)
        
        # Generate fake embeddings
        fake_embeddings = []
        for text in test_samples:
            fake_embeddings.append(text_to_fake_embedding(text))
        fake_embeddings = torch.tensor(fake_embeddings)
        
        # Check if dimensions match
        logger.info(f"Fake embedding dimensions: {fake_embeddings.shape}")
        
        # Invert fake embeddings
        fake_results = vec2text.invert_embeddings(
            embeddings=fake_embeddings.cuda(),
            corrector=corrector
        )
        
        # Print fake embedding results and evaluation
        fake_rouge_scores = []
        for i, (original, result) in enumerate(zip(test_samples, fake_results)):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Original text: {original}")
            logger.info(f"Recovered text: {result}")
            scores = calculate_rouge(original, result)
            fake_rouge_scores.append(scores)
            logger.info(f"ROUGE-1: {scores['rouge-1']['f']:.4f}, ROUGE-2: {scores['rouge-2']['f']:.4f}, ROUGE-L: {scores['rouge-l']['f']:.4f}")
        
        # Calculate embedding vector similarity
        original_embeddings = embed_client.encode(test_samples)
        real_result_embeddings = embed_client.encode(real_results)
        fake_result_embeddings = embed_client.encode(fake_results)
        
        real_embed_sims = []
        fake_embed_sims = []
        
        for i in range(len(test_samples)):
            # Calculate similarity
            orig_vec = np.array(original_embeddings[i])
            real_vec = np.array(real_result_embeddings[i])
            fake_vec = np.array(fake_result_embeddings[i])
            
            real_sim = np.dot(orig_vec, real_vec) / (np.linalg.norm(orig_vec) * np.linalg.norm(real_vec))
            fake_sim = np.dot(orig_vec, fake_vec) / (np.linalg.norm(orig_vec) * np.linalg.norm(fake_vec))
            
            real_embed_sims.append(real_sim)
            fake_embed_sims.append(fake_sim)
            
            # Add results to the results dictionary
            sample_result = {
                "original_text": test_samples[i],
                "real_recovery": {
                    "text": real_results[i],
                    "rouge_1": real_rouge_scores[i]['rouge-1']['f'],
                    "rouge_2": real_rouge_scores[i]['rouge-2']['f'],
                    "rouge_l": real_rouge_scores[i]['rouge-l']['f'],
                    "embedding_similarity": float(real_sim)
                },
                "fake_recovery": {
                    "text": fake_results[i],
                    "rouge_1": fake_rouge_scores[i]['rouge-1']['f'],
                    "rouge_2": fake_rouge_scores[i]['rouge-2']['f'],
                    "rouge_l": fake_rouge_scores[i]['rouge-l']['f'],
                    "embedding_similarity": float(fake_sim)
                }
            }
            results["samples"].append(sample_result)
        
        # Calculate averages
        real_avg_rouge1 = sum(s['rouge-1']['f'] for s in real_rouge_scores) / len(real_rouge_scores)
        real_avg_rouge2 = sum(s['rouge-2']['f'] for s in real_rouge_scores) / len(real_rouge_scores)
        real_avg_rougeL = sum(s['rouge-l']['f'] for s in real_rouge_scores) / len(real_rouge_scores)
        
        fake_avg_rouge1 = sum(s['rouge-1']['f'] for s in fake_rouge_scores) / len(fake_rouge_scores)
        fake_avg_rouge2 = sum(s['rouge-2']['f'] for s in fake_rouge_scores) / len(fake_rouge_scores)
        fake_avg_rougeL = sum(s['rouge-l']['f'] for s in fake_rouge_scores) / len(fake_rouge_scores)
        
        real_avg_embed_sim = sum(real_embed_sims) / len(real_embed_sims)
        fake_avg_embed_sim = sum(fake_embed_sims) / len(fake_embed_sims)
        
        # Calculate percentage differences
        diff_rouge1 = (real_avg_rouge1 - fake_avg_rouge1) / real_avg_rouge1 * 100 if real_avg_rouge1 > 0 else 0
        diff_rouge2 = (real_avg_rouge2 - fake_avg_rouge2) / real_avg_rouge2 * 100 if real_avg_rouge2 > 0 else 0
        diff_rougeL = (real_avg_rougeL - fake_avg_rougeL) / real_avg_rougeL * 100 if real_avg_rougeL > 0 else 0
        diff_embed_sim = (real_avg_embed_sim - fake_avg_embed_sim) / real_avg_embed_sim * 100 if real_avg_embed_sim > 0 else 0
        
        # Save summary results
        results["summary"] = {
            "real_recovery": {
                "avg_rouge_1": float(real_avg_rouge1),
                "avg_rouge_2": float(real_avg_rouge2),
                "avg_rouge_l": float(real_avg_rougeL),
                "avg_embedding_similarity": float(real_avg_embed_sim)
            },
            "fake_recovery": {
                "avg_rouge_1": float(fake_avg_rouge1),
                "avg_rouge_2": float(fake_avg_rouge2),
                "avg_rouge_l": float(fake_avg_rougeL),
                "avg_embedding_similarity": float(fake_avg_embed_sim)
            },
            "difference_percentage": {
                "rouge_1": float(diff_rouge1),
                "rouge_2": float(diff_rouge2),
                "rouge_l": float(diff_rougeL),
                "embedding_similarity": float(diff_embed_sim)
            }
        }
        
        # Print summary
        logger.info("\n" + "-" * 80)
        logger.info("Experiment Results Summary")
        logger.info("-" * 80)
        logger.info(f"Real embedding average scores - ROUGE-1: {real_avg_rouge1:.4f}, ROUGE-2: {real_avg_rouge2:.4f}, ROUGE-L: {real_avg_rougeL:.4f}")
        logger.info(f"Fake embedding average scores - ROUGE-1: {fake_avg_rouge1:.4f}, ROUGE-2: {fake_avg_rouge2:.4f}, ROUGE-L: {fake_avg_rougeL:.4f}")
        logger.info(f"Percentage difference - ROUGE-1: {diff_rouge1:.2f}%, ROUGE-2: {diff_rouge2:.2f}%, ROUGE-L: {diff_rougeL:.2f}%")
        logger.info("\nEmbedding similarity averages:")
        logger.info(f"Real recovered text: {real_avg_embed_sim:.4f}")
        logger.info(f"Fake recovered text: {fake_avg_embed_sim:.4f}")
        logger.info(f"Embedding similarity percentage difference: {diff_embed_sim:.2f}%")
        
        # Save results to file
        saved_path = save_results(results, args.run_dir, args.type_of_dataset, args.output)
        logger.info(f"\nComplete results saved to: {saved_path}")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error occurred during experiment: {e}")
        logger.error(error_details)
        
        # Try to save any existing results even if error occurs
        results["error"] = {
            "message": str(e),
            "traceback": error_details
        }
        save_results(results, args.run_dir, args.type_of_dataset, args.output)

if __name__ == "__main__":
    main()
