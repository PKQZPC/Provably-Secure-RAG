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
parser.add_argument("--user-id",type=str,default="Mr.Brown",help="User ID for the private client")
parser.add_argument("--is-method-A",type=str,default="True",help="Whether to use MethodA or MethodB")
parser.add_argument("--type-of-dataset",type=str,default="HealthCareMagic",help="The type of dataset to process")
parser.add_argument("--api-base",type=str)
parser.add_argument("--api-key",type=str)
parser.add_argument("--embedding-model",type=str,default="text-embedding-ada-002")
args = parser.parse_args()

os.chdir(args.run_dir)
os.environ["OPENAI_API_BASE"] = args.api_base or "https://api.openai.com/v1"
os.environ["OPENAI_API_KEY"] = args.api_key or ""
os.environ["EMBEDDING_MODEL"] = args.embedding_model
os.makedirs("./Evaluation_Materials", exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

user_id = args.user_id
is_method_A = True if args.is_method_A.lower() == "true" else False
type_of_dataset = args.type_of_dataset
if type_of_dataset.lower() == "healthcaremagic":
    input_file = "./Datasets/HealthCareMagic-100k-Cleaned.json"
elif type_of_dataset.lower() == "erionemails":
    input_file = "./Datasets/ErionEmails-500k-Cleaned.json"
elif type_of_dataset.lower() == "billsum":
    input_file = "./Datasets/BillSum-Cleaned.json"
elif type_of_dataset.lower() == "fnspid":
    input_file = "./Datasets/FNSPID-Cleaned.json"
else:
    raise ValueError("type_of_dataset must be HealthCareMagic,ErionEmails or billsum.")
with open(input_file,"r") as infile:
    dataset=json.load(infile)
private_data=[data["matter"] for data in dataset if data["is_private"]==True]
if is_method_A:
    from MethodA_Client import *
    from MethodA_Server import *
    private_server=PrivateServer()
    private_client=PrivateClient(user_id=user_id)
    server_public_key=private_server.register(user_id=private_client.user_id)
    enc_session_key=private_client._negotiate_key(server_public_key=server_public_key)
    private_server._get_session_key(enc_session_key=enc_session_key)
    for idx,data in enumerate(tqdm.tqdm(private_data,desc="Process private dataset")):
        p_node=private_client.p_nodes_addition(data)
        private_server.add_nodes(p_node)
    with open(f"./Evaluation_Materials/Knowledge_base_{type_of_dataset.lower()}_MethodA.json","w") as outfile:
        json.dump(private_server.public_data,outfile,indent=4)
    logger.info("Over! Write into %s", f"./Evaluation_Materials/Knowledge_base_{type_of_dataset.lower()}_MethodA.json")
    with open(f"./Evaluation_Materials/Private_data_{type_of_dataset.lower()}_MethodA.json","w") as outfile:
        json.dump(private_data,outfile,indent=4)
    logger.info("Over! Private data write into %s", f"./Evaluation_Materials/Private_data_{type_of_dataset.lower()}_MethodA.json")
else:
    from MethodB_Client import *
    from MethodB_Server import *
    private_server=PrivateServer()
    private_client=PrivateClient(user_id=user_id)
    public_key=private_server.register(user_id=user_id)
    enc_session_key=private_client._negotiate_key(server_public_key=public_key)
    private_server._get_session_key(enc_session_key)
    for idx,data in enumerate(tqdm.tqdm(private_data,desc="Process private dataset")):
        private_nodes,Authdoor=private_client.submit_private_nodes(text=data)
        iv,enc_client_salt=private_client.sent_key()
        private_server.process_private_nodes(private_nodes,user_id=private_client.user_id,trapdoor=Authdoor,enc_client_salt=enc_client_salt,iv=iv)
    with open(f"./Evaluation_Materials/Knowledge_base_{type_of_dataset.lower()}_MethodB.json","w") as outfile:
        json.dump(private_server.public_data,outfile,indent=4)
    logger.info("Over! Write into %s", f"./Evaluation_Materials/Knowledge_base_{type_of_dataset.lower()}_MethodB.json")
    with open(f"./Evaluation_Materials/Private_data_{type_of_dataset.lower()}_MethodB.json","w") as outfile:
        json.dump(private_data,outfile,indent=4)
    logger.info("Over! Private data write into %s", f"./Evaluation_Materials/Private_data_{type_of_dataset.lower()}_MethodB.json")
