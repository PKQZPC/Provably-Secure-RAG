import os
import json
import glob
import argparse
from rouge import Rouge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging


parser = argparse.ArgumentParser(description="Evaluate the similarity between stolen data results and private data")
parser.add_argument("--results-dir", type=str, default="./Results", help="Directory of result files")
parser.add_argument("--output-dir", type=str, default="./Results", help="Output directory")
parser.add_argument("--specific-file", type=str, default=None, help="Specify a single result file to evaluate")
parser.add_argument("--run-dir", type=str, help="Running directory")
args = parser.parse_args()
os.chdir(args.run_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name%s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def load_json_file(file_path):
    """Load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def calculate_rouge_scores(reference_text, candidate_text):
    """Calculate ROUGE scores"""
    # Ensure input text is not empty and is string type
    if not reference_text or not candidate_text:
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    
    reference_text = str(reference_text).strip()
    candidate_text = str(candidate_text).strip()
    
    if not reference_text or not candidate_text:
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate_text, reference_text)[0]
        # Ensure returned scores have the expected structure
        if isinstance(scores["rouge-l"], dict) and "f" in scores["rouge-l"]:
            return scores
        else:
            # If structure is incorrect, construct a standard format return value
            return {
                "rouge-1": {"f": scores["rouge-1"]["f"] if isinstance(scores["rouge-1"], dict) and "f" in scores["rouge-1"] else 0.0},
                "rouge-2": {"f": scores["rouge-2"]["f"] if isinstance(scores["rouge-2"], dict) and "f" in scores["rouge-2"] else 0.0},
                "rouge-l": {"f": scores["rouge-l"]["f"] if isinstance(scores["rouge-l"], dict) and "f" in scores["rouge-l"] else 0.0}
            }
    except Exception as e:
        logger.warning(f"Error calculating ROUGE: {e}")
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}

def find_result_files(results_dir="./Results"):
    """Find all result files"""
    pattern = os.path.join(results_dir, "stolen_data_*.json")
    return glob.glob(pattern)

def parse_filename(file_path):
    """Parse filename to extract dataset type and RAG method"""
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    if len(parts) >= 5:
        dataset_type = parts[2]
        rag_method = parts[3]
        thief_method = parts[4].split('.')[0]
        return dataset_type, rag_method, thief_method
    return None, None, None

def load_private_data(dataset_type, rag_method):
    """Load corresponding private data"""
    file_path = f"./Evaluation_Materials/Private_data_{dataset_type.lower()}_Method{rag_method}.json"
    return load_json_file(file_path)

def evaluate_result_file(file_path):
    """Evaluate a single result file"""
    dataset_type, rag_method, thief_method = parse_filename(file_path)
    if not dataset_type or not rag_method:
        logger.error(f"Could not parse filename: {file_path}")
        return None
    
    # Load stolen data chunks
    result_data = load_json_file(file_path)
    if not result_data or "stolen_chunks" not in result_data:
        logger.error(f"Invalid result file or missing 'stolen_chunks': {file_path}")
        return None
    stolen_chunks = result_data["stolen_chunks"]
    
    # Load private data
    private_data = load_private_data(dataset_type, rag_method)
    if not private_data:
        logger.error(f"Could not load private data for {dataset_type}, {rag_method}")
        return None
        
    # Calculate similarity scores
    results = []
    for i, chunk in enumerate(tqdm(stolen_chunks, desc=f"Processing {dataset_type}_{thief_method}")):
        chunk_results = []
        for j, private_item in enumerate(private_data):
            scores = calculate_rouge_scores(private_item, chunk)
            try:
                rouge1 = scores["rouge-1"]["f"] if isinstance(scores["rouge-1"], dict) and "f" in scores["rouge-1"] else 0.0
                rouge2 = scores["rouge-2"]["f"] if isinstance(scores["rouge-2"], dict) and "f" in scores["rouge-2"] else 0.0
                rougeL = scores["rouge-l"]["f"] if isinstance(scores["rouge-l"], dict) and "f" in scores["rouge-l"] else 0.0
                
                chunk_results.append({
                    "chunk_id": i,
                    "private_id": j,
                    "rouge-1": rouge1,
                    "rouge-2": rouge2,
                    "rouge-l": rougeL,
                    "chunk": chunk[:100] + "..." if len(chunk) > 100 else chunk,  # Truncate display
                    "private_text": private_item[:100] + "..." if isinstance(private_item, str) and len(private_item) > 100 else private_item  # Truncate display
                })
            except Exception as e:
                logger.warning(f"Error processing scores for chunk {i}, private_item {j}: {e}")
                # Add a result with default values
                chunk_results.append({
                    "chunk_id": i,
                    "private_id": j,
                    "rouge-1": 0.0,
                    "rouge-2": 0.0,
                    "rouge-l": 0.0,
                    "chunk": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    "private_text": private_item[:100] + "..." if isinstance(private_item, str) and len(private_item) > 100 else private_item
                })
                
        # Find the most similar private data for each chunk
        if chunk_results:
            try:
                max_score_item = max(chunk_results, key=lambda x: x["rouge-l"])
                results.append(max_score_item)
            except Exception as e:
                logger.warning(f"Error finding max score item for chunk {i}: {e}")
                if chunk_results:
                    results.append(chunk_results[0])  # Add the first result as a fallback
    
    # Organize results
    results_summary = {
        "dataset": dataset_type,
        "rag_method": rag_method,
        "thief_method": thief_method,
        "num_chunks": len(stolen_chunks),
        "avg_rouge1": np.mean([r["rouge-1"] for r in results]) if results else 0,
        "avg_rouge2": np.mean([r["rouge-2"] for r in results]) if results else 0,
        "avg_rougeL": np.mean([r["rouge-l"] for r in results]) if results else 0,
        "max_rouge1": max([r["rouge-1"] for r in results]) if results else 0,
        "max_rouge2": max([r["rouge-2"] for r in results]) if results else 0,
        "max_rougeL": max([r["rouge-l"] for r in results]) if results else 0,
        "detailed_results": results
    }
    
    return results_summary

def save_results(results, output_dir="./Evaluation_Results"):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the current dataset and RAG method being evaluated
    datasets = set(result["dataset"] for result in results)
    rag_methods = set(result["rag_method"] for result in results)
    
    # Build the base part of the filename
    if len(datasets) == 1 and len(rag_methods) == 1:
        # If there is only one dataset and RAG method
        dataset_type = list(datasets)[0]
        rag_method = list(rag_methods)[0]
        file_base = f"{dataset_type}_method_{rag_method}"
    else:
        # If there are multiple datasets or RAG methods, use "combined"
        file_base = "combined"
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"theft_evaluation_detailed_{file_base}.json")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # Save summary results
    summary_data = []
    for result in results:
        summary_data.append({
            "dataset": result["dataset"],
            "rag_method": result["rag_method"],
            "thief_method": result["thief_method"],
            "num_chunks": result["num_chunks"],
            "avg_rouge1": result["avg_rouge1"],
            "avg_rouge2": result["avg_rouge2"],
            "avg_rougeL": result["avg_rougeL"],
            "max_rouge1": result["max_rouge1"],
            "max_rouge2": result["max_rouge2"],
            "max_rougeL": result["max_rougeL"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"theft_evaluation_summary_{file_base}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Generate evaluation charts
    generate_plots(summary_df, output_dir, file_base)
    
    return detailed_file, summary_file

def generate_plots(summary_df, output_dir, file_base):
    """Generate visualization charts for evaluation results"""
    # Set better chart style
    plt.style.use('ggplot')
    
    # Set font size
    plt.rcParams.update({'font.size': 12})
    
    # Average ROUGE-L scores grouped by attack method
    plt.figure(figsize=(14, 10))
    ax = pd.pivot_table(summary_df, values='avg_rougeL', index=['dataset', 'rag_method'], 
                    columns='thief_method').plot(kind='bar', rot=15)
    plt.title('Average ROUGE-L Score by Dataset and Attack Method', fontsize=16)
    plt.ylabel('ROUGE-L Score', fontsize=14)
    plt.xlabel('Dataset and RAG Method', fontsize=14)
    
    # Add data labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Add legend, placed outside the plot in the upper right
    plt.legend(title='Attack Method', title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"avg_rougeL_by_{file_base}.png"), dpi=300)
    
    # Maximum ROUGE-L scores chart
    plt.figure(figsize=(14, 10))
    ax = pd.pivot_table(summary_df, values='max_rougeL', index=['dataset', 'rag_method'], 
                    columns='thief_method').plot(kind='bar', rot=15)
    plt.title('Maximum ROUGE-L Score by Dataset and Attack Method', fontsize=16)
    plt.ylabel('Maximum ROUGE-L Score', fontsize=14)
    plt.xlabel('Dataset and RAG Method', fontsize=14)
    
    # Add data labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    plt.legend(title='Attack Method', title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"max_rougeL_by_{file_base}.png"), dpi=300)
    
    # Number of extracted data chunks
    plt.figure(figsize=(14, 10))
    ax = pd.pivot_table(summary_df, values='num_chunks', index=['dataset', 'rag_method'], 
                    columns='thief_method').plot(kind='bar', rot=15)
    plt.title('Number of Extracted Chunks by Dataset and Attack Method', fontsize=16)
    plt.ylabel('Number of Chunks', fontsize=14)
    plt.xlabel('Dataset and RAG Method', fontsize=14)
    
    # Add data labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=10)
    
    plt.legend(title='Attack Method', title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"num_chunks_by_{file_base}.png"), dpi=300)

def main():
    """Main function"""

    logger.info("Starting data theft results evaluation")
    
    if args.specific_file:
        # Evaluate a specified single file
        result_file = args.specific_file
        if not os.path.exists(result_file):
            logger.error(f"Specified file not found: {result_file}")
            return
        
        results = [evaluate_result_file(result_file)]
    else:
        # Evaluate all result files
        result_files = find_result_files(args.results_dir)
        logger.info(f"Found {len(result_files)} result files")
        
        results = []
        for file_path in result_files:
            result = evaluate_result_file(file_path)
            if result:
                results.append(result)
    
    # Save evaluation results
    if results:
        detailed_file, summary_file = save_results(results, args.output_dir)
        logger.info(f"Detailed evaluation results saved to: {detailed_file}")
        logger.info(f"Summary evaluation results saved to: {summary_file}")
    else:
        logger.warning("No evaluation results were generated")
    
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()
