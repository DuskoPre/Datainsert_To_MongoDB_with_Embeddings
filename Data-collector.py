#  Copyright (c) 18.04.2024 [D. P.]
#  All rights reserved.

from datasets import load_dataset
from pymongo import MongoClient

issues_dataset = load_dataset("lewtun/github-issues", split="train")
print(issues_dataset)

issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
print(issues_dataset)

columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
print(issues_dataset)


issues_dataset.set_format("pandas")
df = issues_dataset[:]

print(df["comments"][0].tolist())


comments_df = df.explode("comments", ignore_index=True)
print(comments_df.head(4))

from datasets import Dataset

comments_dataset = Dataset.from_pandas(comments_df)
print(comments_dataset)


comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)

comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
print(comments_dataset)


def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }

comments_dataset = comments_dataset.map(concatenate_text)

from transformers import AutoTokenizer, TFAutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)



def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]



def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


embedding = get_embeddings(comments_dataset["text"][0])
print(embedding.shape)



embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).numpy()[0]}
)


embeddings_dataset.add_faiss_index(column="embeddings")

question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).numpy()
print(question_embedding.shape)




scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)


import pandas as pd

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)


for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()


mongo_uri = 'mongodb://172.17.0.2:27017'
db_name = 'new_database18'
collection_name = 'embeddings18'
client = MongoClient(mongo_uri)
db = client[db_name]
embeddings_collection = db[collection_name]
embeddings_collection.insert_many(embeddings_dataset)
