from Security_Methods import *
import os
import base64
import pickle
import re
import json
import torch
import logging
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP
from RAG_Methods import search_node,RAG
from API_chat import Chat
from sentence_transformers import SentenceTransformer
from API_chat import Embedding
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
api_key = os.environ.get("OPENAI_API_KEY", "")
chunk_size=512
class PrivateServer:
	def __init__(self,model_name:str="gpt-3.5-turbo",embedding_use_api:bool=True):
		self.users=[]
		self.file_path="./Storage/Matter_vectors/Public_database_B.json"
		os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
		try:
			with open(self.file_path,"r") as infile:
				self.public_data=json.load(infile)
		except FileNotFoundError:
			self.public_data=[]
		if embedding_use_api:
			self.embedding_model=Embedding(
				api_base=api_base,
				api_key=api_key,
				model=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
			)
		else:
			self.embedding_model=SentenceTransformer('BAAI/bge-small-en-v1.5',cache_folder="./Private_RAG/Embeddings", trust_remote_code=True).to("cuda:0")
		key_pair = RSA.generate(2048)
		self.public_key=key_pair.publickey()
		self.private_key=key_pair
		self.session_key=""
		try:
			self.chat=Chat(api_key=api_key,api_base=api_base,model=model_name,temperature=0.2)
		except Exception as e:
			logging.error(f"Something wrong when initializing chat: {e}")
	
	def _get_session_key(self,enc_session_key):
		cipher_rsa = PKCS1_OAEP.new(self.private_key)
		self.session_key=cipher_rsa.decrypt(enc_session_key)

	def register(self,user_id:str):
		if user_id not in [user["user_id"] for user in self.users]:
			self.users.append({
				"user_id":user_id,
				"trapdoor":"",
			})
			return self.public_key
		else:
			raise ValueError(f"{user_id} already exited.")
		
	def _save_to_file(self):
		try:
			with open(self.file_path, "w") as outfile:
				json.dump(self.public_data, outfile, indent=4)
		except IOError as e:
			logging.error(f"Something wrong when saving flie: {e}")
			raise

	def _search_node(self,addr:str)->dict:
		for data in self.public_data:
			if data["primary_key"]==addr:
				return data
		return {
			"primary_key":"",
			"metadata":{
				"next_node":""
			}
		}
		
	def _get_node_idx(self,addr:str)->int:
		for idx,data in enumerate(self.public_data):
			if data["primary_key"]==addr:
				return idx
		return -1
	
#用户使用数字信封传输saltkey，服务器执行trapdoor认证并取出第一个private_node     
	def _verify_trapdoor(self,trapdoor: str, user_id: str, client_salt: bytes) -> tuple[bool, str, str]:
		h_dec = HMAC.new(key=client_salt, digestmod=SHA256)
		user_id_pad = pad(user_id.encode(), 16)
		h_dec.update(user_id_pad)
		decrypt_hash = h_dec.digest()  # 32 bytes     
		data_block_len = 16 + 16 + 48  # 80 bytes
		repeat_times = (data_block_len // len(decrypt_hash)) + 1
		extended_decrypt_hash = (decrypt_hash * repeat_times)[:data_block_len]
		trapdoor_bytes = bytes.fromhex(trapdoor)
		decoded = bytes([a ^ b for a, b in zip(extended_decrypt_hash, trapdoor_bytes)])
		decoded_user_id = unpad(decoded[:16], 16)  # 16
		decoded_init_key = decoded[16:32]
		decoded_first_addr = unpad(decoded[32:], 16)  
		if decoded_user_id != user_id.encode():
			return False,"",""
		return True, decoded_init_key.hex(), decoded_first_addr.decode()
	
	def _trace_last_node(self,user_id:str,client_salt:bytes)->dict:
		if user_id not in [user["user_id"] for user in self.users]:
			raise ValueError(f"{user_id} please register.")
		else:
			for user in self.users:
				if user["user_id"]==user_id:
					trapdoor=user["trapdoor"]
			if trapdoor=="":
				return {}
			flag,_,first_addr=self._verify_trapdoor(trapdoor,user_id=user_id,client_salt=client_salt)
			if flag==False:
				raise ValueError("ID not match,trapdoor is already changed!")
			current_node=self._search_node(first_addr)
			if current_node=={}:
				logger.info("The first address is wrong!")
				return current_node
			# current_key=bytes.fromhex(first_key)
			while(current_node["metadata"]["next_node"]!=""):
				# iv=current_node["metadata"]["iv"]
				# if  current_node["metadata"]["hash_before"] != SHA256.new(current_key).hexdigest():
				# 	raise ValueError("hash_before not match,trapdoor is already changed!")
				# cbc_decryptor=Decryptor(current_key,mode=AES.MODE_CBC,iv=iv)
				# current_key=cbc_decryptor.decrypt(current_node["metadata"]["cipher_next_key"])
				next_addr=current_node["metadata"]["next_node"]
				current_temp=self._search_node(next_addr)
				if(current_temp["metadata"]["next_node"]==""):break
				current_node=current_temp
			return current_node

	def process_private_nodes(self,private_nodes:list,user_id:str,trapdoor:str,enc_client_salt:str,iv:bytes)->bool:
		def get_key(enc_client_salt,iv):
			cbc_decryptor = Decryptor(self.session_key, mode=AES.MODE_CBC, iv=iv)
			client_salt = cbc_decryptor.decrypt(enc_client_salt)
			return client_salt
		if user_id not in [user["user_id"] for user in self.users]:
			raise ValueError(f"{user_id} please register.")
		else:
			client_salt=get_key(enc_client_salt=enc_client_salt,iv=iv)
			# for idx,user in enumerate(self.users):
			# 	if user["user_id"]==user_id:
			# 		user_id_idx=idx
			# if len(self.users[user_id_idx]["trapdoor"]) != 0:
			if len(trapdoor) !=0:
				#表示是第一次提交，需要写入trapdoor
				for idx,user in enumerate(self.users):
					if user["user_id"]==user_id:
						self.users[idx]["trapdoor"]=trapdoor
			else:
				#第n次提交，需要验证trapdoor，然后得到第一个node的信息，顺着链找到最后一个node，将next_addr修改成添加的node 的addr。
				last_node=self._trace_last_node(user_id,client_salt)
				if last_node["primary_key"] == False:
					return False
				last_node_idx=self._get_node_idx(last_node["primary_key"])
				if last_node_idx==-1:
					raise ValueError("last node not found!")
				else:
					self.public_data[last_node_idx]["metadata"]["next_node"]=private_nodes[0]["primary_key"]
		try:
			# if private_node["doc_hash"] not in [data["doc_hash"] for data in self.public_data]
			private_nodes=[private_node for private_node in private_nodes]
			if len(private_nodes) == 0:
				logging.info("The documents is already exist!")
				return False
			for i in range(len(private_nodes)):
				private_nodes[i]["metadata"]["next_node"]=private_nodes[i+1]["primary_key"] if (i+1)<len(private_nodes) else None
			self.public_data.extend(private_nodes)
			if len(private_nodes)==0:
					logging.info("No new nodes to add.")
					return False
			else:
					self._save_to_file()
					return True
		except Exception as e:
			logging.error(f"Something wrong when adding node: {e}")
			return False
	
	def get_private_information(self,user_id:str,enc_client_salt:str,iv:bytes)->list:
		def get_key(enc_client_salt,iv):
			cbc_decryptor = Decryptor(self.session_key, mode=AES.MODE_CBC, iv=iv)
			client_salt = cbc_decryptor.decrypt(enc_client_salt)
			return client_salt
		if user_id not in [user["user_id"] for user in self.users]:
			raise ValueError(f"{user_id} please register.")
		else:
			for user in self.users:
				if user["user_id"]==user_id:
					trapdoor=user["trapdoor"]
		client_salt=get_key(enc_client_salt,iv)
		flag,first_key,first_addr=self._verify_trapdoor(trapdoor,user_id=user_id,client_salt=client_salt)
		if flag==False:
			raise ValueError("ID not match,trapdoor is already changed!")
		restore_nodes=[]
		current_key=bytes.fromhex(first_key)
		current_node=self._search_node(first_addr)
		if current_node=={}:
			return []
		while current_node.get("primary_key"):
			restore_node={}
			logger.info("Processing node :"+current_node["primary_key"])
			iv=base64.b64decode(current_node["metadata"]["iv"].encode())
			cipher_chunk=current_node["chunk"]
			cipher_embedding=current_node["embedding"]
			cbc_decryptor = Decryptor(current_key, mode=AES.MODE_CBC, iv=iv)
			restore_node["chunk"]=cbc_decryptor.decrypt(cipher_chunk).decode()
			restore_node["embedding"]=pickle.loads(cbc_decryptor.decrypt(cipher_embedding)).tolist()
			current_key=cbc_decryptor.decrypt(current_node["metadata"]["cipher_next_key"])
			restore_node["primary_key"]=current_node["primary_key"]
			restore_node["doc_hash"]=current_node["doc_hash"]
			restore_node["metadata"]={
				"iv":base64.b64encode(iv).decode(),
				"next_node":current_node["metadata"]["next_node"],
				"hash_before":current_node["metadata"]["hash_before"],
				"cipher_next_key":current_node["metadata"]["cipher_next_key"]
			}
			restore_nodes.append(restore_node)
			next_addr=current_node["metadata"]["next_node"]
			if not next_addr:
				break
			current_node=self._search_node(next_addr)
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
			"users": self.users,
			"public_key": self.public_key.export_key(),
			"private_key": self.private_key.export_key(),
			"session_key": self.session_key,
		}
		try:
			with open("config.json", "w") as outfile:
				json.dump(config, outfile, indent=4)
		except IOError as e:
			logging.error(f"Something wrong when saving config: {e}")
			raise
