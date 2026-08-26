import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Security_Methods import *
from Cryptodome.Hash import SHA256
from typing import Union, Optional
from API_chat import Embedding

# def split_text_into_chunks(text:str,chunk_size:int)->list:
#     chunks=[]
#     current_chunk=""
#     words=text.split()
#     for word in words:
#         if len(current_chunk)+len(word) +1 <=chunk_size:
#             current_chunk += (" " if current_chunk else "") + word
#         else:
#             chunks.append(current_chunk)
#             current_chunk=word
#     if current_chunk:
#        chunks.append(current_chunk)
#     return chunks

# import re

def split_text_into_chunks(text: str, chunk_size: int, overlap: int = 0) -> list:
    # 检查 overlap 的合法性
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("`overlap` must be >= 0 and < `chunk_size`.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # 下一个块的起始位置，推进 chunk_size - overlap 个字符

    return chunks


def generate_node(chunk:str,model:Union[SentenceTransformer,Embedding])->dict:
    node={}
    node["primary_key"]=generate_collision_resistant_uuid()
    node["chunk"]=chunk
    if isinstance(model,SentenceTransformer):
        node["embedding"]=model.encode(sentences=chunk).tolist()
    else:
        node["embedding"]=model.encode(chunk)
    node["doc_hash"]=SHA256.new(chunk.encode()).hexdigest()
    node["metadata"]={}
    return node

def search_node(primary_keys:list,nodes:list)->list:
    return [node for node in nodes if node["primary_key"] in primary_keys]

def RAG(query:str,nodes:list,embedding_model:Union[SentenceTransformer,Embedding],top_k_similarity:int,similarity_threshold:float)->list:
    query_embedding=embedding_model.encode(sentences=query)
    for restore_node in nodes:
        if isinstance(restore_node["embedding"],list):
            restore_node["similarity"]=cosine_similarity(torch.tensor(restore_node["embedding"]).reshape(1,-1),torch.tensor(query_embedding).reshape(1,-1))[0][0]
        else:
            restore_node["similarity"]=0
            pass
    nodes.sort(key=lambda x:x["similarity"],reverse=True)
    nodes=[restore_node for restore_node in nodes if restore_node["similarity"]>similarity_threshold]
    return nodes[:min(top_k_similarity,len(nodes))]

if __name__=="__main__":
    input_text = "This is a sample text that shows how to split the text into multiple small chunks. The size of each small piece can be adjusted as needed."
    chunk_size = 20
    chunks = split_text_into_chunks(input_text, chunk_size,optimize=False)
    print(chunks)