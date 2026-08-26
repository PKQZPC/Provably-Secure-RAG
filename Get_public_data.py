import json
import os
import torch
from sentence_transformers import SentenceTransformer
import tqdm
import logging
from RAG_Methods import generate_node,split_text_into_chunks
from API_chat import Embedding
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir",type=str)
parser.add_argument("--chunk-size",type=int,default=512)
parser.add_argument("--overlap",type=int,default=20)
parser.add_argument("--is-method-A",type=str,default="True",help="Whether to use MethodA or MethodB.")
parser.add_argument("--api-base",type=str)
parser.add_argument("--api-key",type=str)
parser.add_argument("--embedding-model",type=str,default="text-embedding-ada-002")
parser.add_argument("--type-of-dataset",type=str,default="HealthCareMagic",help="The type of dataset to process.")
args = parser.parse_args()

os.chdir(args.run_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Start generate public data:")
chunk_size=args.chunk_size
overlap=args.overlap
# device = torch.device("cuda:0")
# model = SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder="./Embeddings", trust_remote_code=True).to(device)
model=Embedding(api_base=args.api_base, api_key=args.api_key,model=args.embedding_model)
if args.type_of_dataset.lower() == "healthcaremagic":
    input_file="./Datasets/HealthCareMagic-100k-Cleaned.json"
elif args.type_of_dataset.lower() == "erionemails":
    input_file="./Datasets/ErionEmails-500k-Cleaned.json"
elif args.type_of_dataset.lower() == "billsum":
    input_file="./Datasets/BillSum-Cleaned.json"
elif args.type_of_dataset.lower() == "fnspid":
    input_file = "./Datasets/FNSPID-Cleaned.json"
else:
    raise ValueError("type_of_dataset must be HealthCareMagic,ErionEmails or billsum.")
with open(input_file,"r") as infile:
    dataset=json.load(infile)
chunks=[split_text_into_chunks(data["matter"],chunk_size=chunk_size,overlap=overlap) for data in tqdm.tqdm(dataset,desc="Split_Sentences") if data["is_private"]==False]
chunks=[c for chunk in chunks for c in chunk]
public_data=[generate_node(chunk,model=model) for chunk in tqdm.tqdm(chunks,desc="Generate_nodes")]
if args.is_method_A.lower() == "true":
    output_path="./Storage/Matter_vectors/Public_database_A.json"
elif args.is_method_A.lower() == "false":
    output_path="./Storage/Matter_vectors/Public_database_B.json"
else:
    raise ValueError("is_method_A must be True or False.")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path,"w") as outfile:
    json.dump(public_data,outfile,indent=4)
logger.info("Over! Write into %s", output_path)
public_data=[data["matter"] for data in dataset if data["is_private"]==False]
os.makedirs("./Evaluation_Materials", exist_ok=True)
if args.is_method_A.lower() == "true":
    with open(f"./Evaluation_Materials/Public_data_{args.type_of_dataset.lower()}_MethodA.json","w") as outfile:
        json.dump(public_data,outfile,indent=4)
elif args.is_method_A.lower() == "false":
    with open(f"./Evaluation_Materials/Public_data_{args.type_of_dataset.lower()}_MethodB.json","w") as outfile:
        json.dump(public_data,outfile,indent=4)
else:
    raise ValueError("is_method_A must be True or False.")
