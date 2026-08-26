import logging
import json
import tqdm
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from API_chat import Chat
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir",type=str)
parser.add_argument("--seed-value",type=int,default=10)
# parser.add_argument("--fraction-of-data",type=float,default=None,help="Fraction of data to use (deprecated)")
parser.add_argument("--num-of-data",type=int,default=100,help="Number of data points to use")
parser.add_argument("--num-of-private",type=int,default=3)
parser.add_argument("--do-shuffle",type=str,default="True")
parser.add_argument("--do-shuffle-private",type=str,default="True")
parser.add_argument("--type-of-dataset",type=str,default="ErionEmails",help="The type of dataset to process.")
args = parser.parse_args()

os.chdir(args.run_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

seed_value = args.seed_value
num_of_data=args.num_of_data
num_of_private=args.num_of_private
do_shuffle=True if args.do_shuffle.lower() == "true" else False
do_shuffle_private=True if args.do_shuffle_private.lower() == "true" else False
random.seed(seed_value)

def process_dataset_ErionEmails(dataset,num_of_private:int,do_shuffle_private:bool=True):
    dataset_cleaned=[]
    for idx in tqdm.tqdm(range(len(dataset)),desc="Process dataset"):
        dataset_temp={}
        dataset_temp["file"]=dataset.loc[idx,"file"]
        dataset_temp["message"]=dataset.loc[idx,"message"]
        dataset_temp["matter"]=f"""file:{dataset.loc[idx,"file"]}\nmessage:{dataset.loc[idx,"message"]}"""
        if num_of_private >0 : 
            dataset_temp["is_private"]=True
            num_of_private-=1
        else:
            dataset_temp["is_private"]=False
        dataset_cleaned.append(dataset_temp)
    if do_shuffle_private:
        random.shuffle(dataset_cleaned)
    return dataset_cleaned

def process_dataset_HealthCareMagic(dataset:list,num_of_private:int,do_shuffle_private:bool=True):
    for idx,data in enumerate(tqdm.tqdm(dataset,desc="Process dataset")):
        dataset[idx]["matter"]=f"""Patient:{data["input"]}\nDoctor:{data["output"]}"""
        if num_of_private >0 : 
            dataset[idx]["is_private"]=True
            num_of_private-=1
        else:
            dataset[idx]["is_private"]=False
    if do_shuffle_private:
        random.shuffle(dataset)
    return dataset

def process_dataset_BillSum(dataset:list, num_of_private:int, do_shuffle_private:bool=True):
    for idx,data in enumerate(tqdm.tqdm(dataset,desc="Process dataset")):
        dataset[idx]["matter"]=f"""bill_id:{data["bill_id"]}\ntext:{data["text"]}"""
        if num_of_private > 0 : 
            dataset[idx]["is_private"]=True
            num_of_private-=1
        else:
            dataset[idx]["is_private"]=False
    if do_shuffle_private:
        random.shuffle(dataset)
    return dataset

def process_dataset_FNSPID(dataset:list, num_of_private:int, do_shuffle_private:bool=True):
    dataset_cleaned = []
    for idx, data in enumerate(tqdm.tqdm(dataset, desc="Process dataset")):
        data_temp = {}
        row = data["row"]
        data_temp["Date"] = row["Date"]
        data_temp["Article_title"] = row["Article_title"]
        data_temp["Stock_symbol"] = row["Stock_symbol"]
        data_temp["Publisher"] = row["Publisher"]
        data_temp["Url"] = row["Url"]
        data_temp["matter"] = f"""Date: {row["Date"]}\nTitle: {row["Article_title"]}\nStock: {row["Stock_symbol"]}\nPublisher: {row["Publisher"]}\nURL: {row["Url"]}"""
        if num_of_private > 0:
            data_temp["is_private"] = True
            num_of_private -= 1
        else:
            data_temp["is_private"] = False
        dataset_cleaned.append(data_temp)
    
    if do_shuffle_private:
        random.shuffle(dataset_cleaned)
    return dataset_cleaned

def process_jsonl_file(file_path):
    data = []
    with open(file_path, "r") as infile:
        for line in infile:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    return data

if args.type_of_dataset.lower() == "healthcaremagic":
    logger.info("Start process dataset HealthCareMagic:")
    with open("./Datasets/HealthCareMagic-100k.json", "r") as infile:
        dataset = json.load(infile)
    if num_of_private > num_of_data:
        raise ValueError("num_of_private must be less than the number of dataset.")
    if do_shuffle:
        random.shuffle(dataset)
    dataset = dataset[:num_of_data]
    dataset_cleaned=process_dataset_HealthCareMagic(dataset=dataset,num_of_private=num_of_private,do_shuffle_private=do_shuffle_private)
    outfile_path="./Datasets/HealthCareMagic-100k-Cleaned.json"

elif args.type_of_dataset.lower() == "erionemails":
    logger.info("Start process dataset ErionEmails:")
    dataset = pd.read_csv("./Datasets/emails.csv")
    if num_of_private > num_of_data:
        raise ValueError("num_of_private must be less than the number of dataset.")
    if do_shuffle:
        dataset = dataset.sample(frac=1,random_state=seed_value).reset_index(drop=True).head(num_of_data)
    else:
        dataset = dataset.head(num_of_data)
    dataset_cleaned=process_dataset_ErionEmails(dataset,num_of_private=num_of_private,do_shuffle_private=do_shuffle_private)
    outfile_path="./Datasets/ErionEmails-500k-Cleaned.json"

elif args.type_of_dataset.lower() == "billsum":
    logger.info("Start process dataset BillSum:")
    dataset = process_jsonl_file("./Datasets/billsum_v4_1/us_train_data_final_OFFICIAL.jsonl")
    if num_of_private > num_of_data:
        raise ValueError("num_of_private must be less than the number of dataset.")
    if do_shuffle:
        random.shuffle(dataset)
    dataset = dataset[:num_of_data]
    dataset_cleaned=process_dataset_BillSum(dataset=dataset,num_of_private=num_of_private,do_shuffle_private=do_shuffle_private)
    outfile_path="./Datasets/BillSum-Cleaned.json"

elif args.type_of_dataset.lower() == "fnspid":
    logger.info("Start process dataset FNSPID:")
    with open("./Datasets/FNSPID/fnspid_samples.json", "r") as infile:
        dataset_json = json.load(infile)
        dataset = dataset_json["rows"]
    if num_of_private > num_of_data:
        raise ValueError("num_of_private must be less than the number of dataset.")
    if do_shuffle:
        random.shuffle(dataset)
    dataset = dataset[:num_of_data]
    dataset_cleaned=process_dataset_FNSPID(dataset=dataset,num_of_private=num_of_private,do_shuffle_private=do_shuffle_private)
    outfile_path="./Datasets/FNSPID-Cleaned.json"

with open(outfile_path, "w") as outfile:
    json.dump(dataset_cleaned, outfile, indent=4)
logger.info("Over! Write into %s", outfile_path)