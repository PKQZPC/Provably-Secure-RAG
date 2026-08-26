from Security_Methods import *
import os
import base64
import pickle
import re
import json
import logging
from RAG_Methods import search_node,RAG
from API_chat import Chat
from sentence_transformers import SentenceTransformer
from API_chat import Embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
api_key = os.environ.get("OPENAI_API_KEY", "")
class PrivateServer:
    def __init__(self,model_name:str="gpt-3.5-turbo",embedding_use_api:bool=True):
        self.user_ids=[]
        self.file_path="./Storage/Matter_vectors/Public_database_A.json"
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        try:
            with open(self.file_path,"r") as infile:
                self.public_data=json.load(infile)
        except FileNotFoundError:
            self.public_data=[]
        key_pair = RSA.generate(2048)
        if embedding_use_api:
            self.embedding_model=Embedding(
                api_base=api_base,
                api_key=api_key,
                model=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
            )
        else:
            self.embedding_model=SentenceTransformer(
                'BAAI/bge-small-en-v1.5', cache_folder="./Embeddings", trust_remote_code=True
            ).to("cuda:0")
        self.public_key=key_pair.publickey()
        self.private_key=key_pair
        self.session_key=""
        try:
            self.chat=Chat(api_key=api_key,api_base=api_base,model=model_name,temperature=0.2)
        except Exception as e:
            logging.error(f"Something wrong when initializing chat: {e}")

    def register(self,user_id:str):
        if user_id not in self.user_ids:
            self.user_ids.append(user_id)
            return self.public_key
        else:
            raise ValueError(f"{user_id} already exited.")
    
    def _get_session_key(self,enc_session_key):
        cipher_rsa = PKCS1_OAEP.new(self.private_key)
        self.session_key=cipher_rsa.decrypt(enc_session_key)
        
    def _save_to_file(self):
        try:
            with open(self.file_path, "w") as outfile:
                json.dump(self.public_data, outfile, indent=4)
        except IOError as e:
            logging.error(f"Something wrong when saving flie: {e}")
            raise

    def add_nodes(self,nodes:list)->bool:
        try:
            # temp=[node for node in nodes if node["doc_hash"] not in set([node["doc_hash"] for node in self.public_data])]
            temp=[node for node in nodes]
            self.public_data.extend(temp)
            if temp == []:
                logging.info("No new nodes to add.")
                return False
            else:
                self._save_to_file()
            return True
        except Exception as e:
            logging.error(f"Something wrong when adding node: {e}")
            return False
    
    def _reload(self)->None:
        try:
            with open("./Storage/Matter_vectors/Public_database_A.json","r") as infile:
                self.public_data=json.load(infile)
        except Exception as e:
            logging.error(f"Something wrong when reload file: {e}")
    def get_private_nodes(self,user_information:tuple[list,str])->list:
        if user_information[1] not in self.user_ids:
            logging.error(f"{user_information[0]} not legal member.")
        primary_keys=user_information[0]
        return search_node(primary_keys,self.public_data)

    def get_private_information(self,private_nodes:list,enc_enc_key:str,iv:bytes):
        def get_key(enc_enc_key,iv):
            cbc_decryptor = Decryptor(self.session_key, mode=AES.MODE_CBC, iv=iv)
            enc_key = cbc_decryptor.decrypt(enc_enc_key)
            return enc_key
        enc_key=get_key(enc_enc_key,iv)
        restore_nodes=[]
        for private_node in private_nodes:
            iv=base64.b64decode(private_node["metadata"]["iv"].encode())
            cipher_chunk=private_node["chunk"]
            cipher_embedding=private_node["embedding"]
            cbc_decryptor = Decryptor(enc_key, mode=AES.MODE_CBC, iv=iv)
            node={
                "primary_key":private_node["primary_key"],
                "chunk":cbc_decryptor.decrypt(cipher_chunk).decode("utf-8"),
                "embedding":pickle.loads(cbc_decryptor.decrypt(cipher_embedding)).tolist(),
                "metadata":private_node["metadata"]
            }
            restore_nodes.append(node)
        restore_nodes.extend(self.public_data)
        return restore_nodes
    
    def chat_llm(self,query:str,top_k_similarity:int=5,similarity_threshold:float=0.6)->str:
        reference_node=RAG(query,self.public_data,self.embedding_model,top_k_similarity,similarity_threshold)
        relevant_documents="\n".join(["<relevance>"+node["chunk"]+"</relevance>" for node in reference_node])
        prompt=f"""
        The {len(reference_node)} relevant documents retrieved are as follows:
        {relevant_documents}
        Request:
        1 - The tag like <relevance></relevance> wraps the retrieved related files. If there are no related files, there will be no such tag.
        2 - You need to recover in response to the user's inquiry, referring to the relevant documents within the <relevance></relevance> tag, and the user's inquiry is enclosed within the <query></query> tag.
        3 - Your output result is placed in the <output></output> tag.
        4 - Do not output your relevant document just read them.
        <query>{query}</query>
        """
        answer=self.chat.ask(prompt)
        matches = re.findall(r'<output>(.*?)</output>', answer,flags=re.DOTALL)
        return matches[-1]
    
    def private_chat_llm(self,query:str,restore_nodes:list,top_k_similarity:int=5,similarity_threshold:float=0.6)->str:
        reference_nodes=RAG(query,restore_nodes,self.embedding_model,top_k_similarity,similarity_threshold)
        relevant_documents="\n".join(["<relevance>"+node["chunk"]+"</relevance>" for node in reference_nodes])
        prompt=f"""
        The {len(reference_nodes)} relevant documents retrieved are as follows:
        {relevant_documents}
        Request:
        1 - The tag like <relevance></relevance> wraps the retrieved related files. If there are no related files, there will be no such tag.
        2 - You need to recover in response to the user's inquiry, referring to the relevant documents within the <relevance></relevance> tag, and the user's inquiry is enclosed within the <query></query> tag.
        3 - Your output result is placed in the <output></output> tag.
        4 - Do not output your relevant document just read them.
        <query>{query}</query>
        """
        answer=self.chat.ask(prompt)
        matches = re.findall(r'<output>(.*?)</output>', answer,flags=re.DOTALL)
        return matches[-1]
    
    def _save_config(self):
        config = {
            "user_ids": self.user_ids,
            "public_key": self.public_key.export_key(),
            "key_pair": self.private_key.export_key(),
            "session_key": self.session_key,
        }
        outfile_path = "./Storage/Private_Config/MethodA/Server.pkl"
        os.makedirs(os.path.dirname(outfile_path), exist_ok=True)  # 确保目录存在
        with open(outfile_path, "wb") as outfile:
            pickle.dump(config, outfile)
        logger.info("MethodA Server's config are wrote into %s", outfile_path)
    
