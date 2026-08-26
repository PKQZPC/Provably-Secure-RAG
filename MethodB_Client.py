from Security_Methods import *
import os
import pickle
import torch
from sentence_transformers import SentenceTransformer
import base64
import logging
from RAG_Methods import split_text_into_chunks,generate_node
from API_chat import Embedding
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
chunk_size=512
class PrivateClient:
    def __init__(self,user_id:str,embedding_use_api:bool=True):
        self.master_key=get_random_bytes(16)
        self.salt=get_random_bytes(16)
        self.session_key=AESHelper.generate_key()
        self.user_id=user_id
        self.current_key=""
        self.is_first=True
        self.gen_flag=True
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

    def _generate_trapdoor(self,user_id: str, first_addr:str) -> str:
        init_key = HKDF(self.master_key, 16, salt=user_id.encode(), hashmod=SHA256, context=b"InitKey")
        h_enc = HMAC.new(key=self.salt, digestmod=SHA256)
        user_id_pad = pad(user_id.encode(), 16)  # 块大小 16，填充到 16 字节
        h_enc.update(user_id_pad)
        current_hash = h_enc.digest()  # 32 字节
        first_addr_pad = pad(first_addr.encode(), 16)  # 关键修正：块大小 16 → 填充到 48 字节
        data_block = user_id_pad + init_key + first_addr_pad  # 16+16+48=80 字节
        repeat_times = (len(data_block) // len(current_hash)) + 1
        extended_hash = (current_hash * repeat_times)[:len(data_block)]
        trapdoor = bytes([a ^ b for a, b in zip(extended_hash, data_block)])
        return trapdoor.hex()
    
    #只能输入一个chunk，必须先分chunk再使用，提交第一个node自动返回trapdoor存储在验证数据库
    # def get_private_node(self,chunk:str,user_id:str)->tuple[dict,str]:
    def _get_private_node(self,chunk:str)->dict:
        if self.gen_flag:
            init_key = HKDF(self.master_key, 16, salt=self.user_id.encode(), 
                        hashmod=SHA256, context=b"InitKey")
            self.gen_flag=False
        else:
            init_key=self.current_key
        iv=AESHelper.generate_iv()
        cbc_encryptor=Encryptor(init_key,mode=AES.MODE_CBC,iv=iv)
        next_key=AESHelper.generate_key(16)
        self.current_key=next_key
        embedding=self.embedding_model.encode(sentences=chunk)
        hash_before=SHA256.new(init_key).hexdigest()
        _,cipher_chunk=cbc_encryptor.encrypt(chunk.encode())
        _,cipher_embedding=cbc_encryptor.encrypt(pickle.dumps(torch.tensor(embedding)))
        _,cipher_next_key=cbc_encryptor.encrypt(next_key)
        node={
            "primary_key":generate_collision_resistant_uuid(),
            "chunk":cipher_chunk,
            "embedding":cipher_embedding,
            "doc_hash":SHA256.new(chunk.encode()).hexdigest(),
            "metadata":{
                    "iv":base64.b64encode(iv).decode(),
                    "next_node":"",
                    "hash_before":hash_before,
                    "cipher_next_key":cipher_next_key
                }
            }
        # if self.is_first:
        #     self.current_address=node["primary_key"]
        #     trapdoor=self._generate_trapdoor(user_id,first_addr=node["primary_key"])
        #     self.is_first=False
        #     return node,trapdoor
        # else:
        #     return node,""
        return node

    def sent_key(self)->tuple[bytes,str]:
        iv=AESHelper.generate_iv()
        cbc_encryptor=Encryptor(self.session_key,mode=AES.MODE_CBC,iv=iv)
        _,enc_client_salt=cbc_encryptor.encrypt(self.salt)
        return iv,enc_client_salt

    def _get_private_nodes(self,text:str)->list:
        chunks=split_text_into_chunks(text, chunk_size)
        private_nodes=[self._get_private_node(chunk) for chunk in chunks]
        # for i in range(len(private_nodes)):
        #     private_nodes[i]["metadata"]["next_node"]=private_nodes[i+1]["primary_key"] if (i+1)<len(chunks) else None
        return private_nodes
    
    def submit_private_nodes(self,text:str)->tuple[list,str]:
        private_nodes=self._get_private_nodes(text)
        if self.is_first:
            trapdoor=self._generate_trapdoor(self.user_id,first_addr=private_nodes[0]["primary_key"])
            self.is_first=False
            return private_nodes,trapdoor
        else:
            # 不是第一次提交
            return private_nodes,""
    
    # def submit_next_node(self,chunk:str,user_id:str)->dict:
        
    # def submit_private_nodes_multi(self,texts:list)->list:
    #     private_nodes=[]
    #     for text in texts:
    #         private_node,trapdoor=self.submit_private_nodes(text)
    #         private_nodes+=private_node
    #     return private_nodes
    def _save_config(self):
        config={
            "user_id":self.user_id,
            "master_key":self.master_key,
            "salt":self.salt,
            "session_key":self.session_key,
            "current_key":self.current_key,
            "is_first":self.is_first,
            "gen_flag":self.gen_flag
        }
        outfile_path = f"./Storage/Private_Config/MethodB/Client_({self.user_id}).pkl"
        os.makedirs(os.path.dirname(outfile_path), exist_ok=True) 
        with open(outfile_path, "wb") as outfile:
            pickle.dump(config, outfile)
        logger.info("MethodB Client's config are wrote into %s", outfile_path)
