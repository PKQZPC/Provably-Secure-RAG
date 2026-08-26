from Security_Methods import *
import os
import base64
import pickle
import torch
from sentence_transformers import SentenceTransformer
import base64
from Cryptodome.Hash import SHA256
from RAG_Methods import split_text_into_chunks,generate_node
from API_chat import Embedding
import logging
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

chunk_size=512
class PrivateClient:
    def __init__(self,user_id:str,embedding_use_api:bool=True):
        self.user_id=user_id
        self.primary_key=[]
        self.enc_key=AESHelper.generate_key()
        self.session_key=AESHelper.generate_key()
        if embedding_use_api:
            self.embedding_model=Embedding(
                api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
            )
        else:
            self.embedding_model=SentenceTransformer('BAAI/bge-small-en-v1.5',cache_folder="./Private_RAG/Embeddings", trust_remote_code=True).to("cuda:0")

    def _negotiate_key(self,server_public_key):
        cipher_rsa = PKCS1_OAEP.new(server_public_key)
        enc_session_key = cipher_rsa.encrypt(self.session_key)
        return enc_session_key

    def p_nodes_addition(self,text:str)->list:
        chunks=split_text_into_chunks(text, chunk_size)
        embeddings=self.embedding_model.encode(sentences=chunks)
        nodes=[]
        for chunk,embedding in zip(chunks,embeddings):
            iv=AESHelper.generate_iv()
            cbc_encryptor=Encryptor(self.enc_key,mode=AES.MODE_CBC,iv=iv)
            node={}
            _, cipher_chunk=cbc_encryptor.encrypt(chunk.encode())
            _,cipher_embedding=cbc_encryptor.encrypt(pickle.dumps(torch.tensor(embedding)))
            node["primary_key"]=generate_collision_resistant_uuid()
            self.primary_key.append(node["primary_key"])
            node["chunk"]=cipher_chunk
            node["embedding"]=cipher_embedding
            node["doc_hash"]=SHA256.new(chunk.encode()).hexdigest()
            node["metadata"]={
                "iv":base64.b64encode(iv).decode()
            }
            nodes.append(node)
        return nodes
    
    def p_nodes_addition_multi(self,texts:list)->list:
        result_list=[]
        for text in texts:
            result_list.extend(self.p_nodes_addition(text))
        return result_list

    def node_addition(self,text:str)->list:
        chunks=split_text_into_chunks(text,chunk_size)
        return [generate_node(chunk,self.embedding_model) for chunk in chunks]
    
    def nodes_addition_multi(self,texts:list)->list:
        result_list=[]
        for text in texts:
            result_list.extend(self.nodes_addition(text))
        return result_list
    
    def request_private_information(self)->tuple[list,str]:
        return self.primary_key,self.user_id
    
    def sent_key(self)->tuple[bytes,str]:
        iv=AESHelper.generate_iv()
        cbc_encryptor = Encryptor(self.session_key, mode=AES.MODE_CBC, iv=iv)
        _, enc_enc_key = cbc_encryptor.encrypt(self.enc_key)
        return iv,enc_enc_key

    def _save_config(self):
        config = {
            "user_id": self.user_id,
            "primary_key": self.primary_key,
            "enc_key": self.enc_key,
            "session_key": self.session_key
        }
        outfile_path = f"./Storage/Private_Config/MethodA/Client_({self.user_id}).pkl"
        os.makedirs(os.path.dirname(outfile_path), exist_ok=True) 
        with open(outfile_path, "wb") as outfile:
            pickle.dump(config, outfile)
        logger.info("MethodA Client's config are wrote into %s", outfile_path)
