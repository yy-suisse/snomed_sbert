import polars as pl
import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import numpy as np

def embed_and_save(model, data, file_name, type ):
    embeddings = model.encode(data, show_progress_bar=True, device="cuda")
    np.save(PATH_SAVE + file_name + type + ".npy", embeddings)
    print("Embeddings saved successfully!")


# pipeline for embeddings
MODEL_NAME_B = "all-MiniLM-L6-v2"
FILE_NAME_B = "base"
MODEL_NAME_FT = "yyzheng00/snomed_triplet_800k"
FILE_NAME_FT = "ft"


# paths for the data
PATH_LOAD = "D:/finetune_sbert_new/"
PATH_SAVE = "D:/finetune_sbert_new/embeddings/"

# load data
# 1. load all concept data
all_concept = pd.read_csv(PATH_LOAD + "graph_data/concept_snomed_hug_2025-04-01.csv")
all_concept["n.id"] = all_concept["n.id"].astype(str)
all_concept = pl.from_pandas(all_concept).rename({"n.id": "id"})

# 2. load pre_coordation concept data with expression
pre_coordination = pl.read_parquet(PATH_LOAD + "graph_data/pre_with_expression.parquet",
                                   columns=["id" ,"expression"])

# 3. prepare semantic tag patterns
list_sem_tag = all_concept['sem_tag'].unique().to_list() + ["attribute"]
escaped_tags = [re.escape(tag) for tag in list_sem_tag]
pattern = r"\s*\((" + "|".join(escaped_tags) + r")\)"

# 4. preprocess pre and post (all in hugdata), get cleaned label(post) and expression (both) in a dataframe
concept_pre = all_concept.filter(pl.col("concept_type") == "SCT_PRE").join(
    pre_coordination, on="id",how="inner").unique()
concept_pre = (concept_pre
               .with_columns(pl.col("expression")
                             .str.replace_all(r"\d+\|", "|")
                             .str.replace_all(pattern, "")
                             .alias("expression"))
               .unique())

concept_post = (all_concept
                .filter(pl.col("concept_type") == "SCT_POST")
                .with_columns(expression = pl.col("n.label"))
                .unique())
concept_post = (concept_post
                .with_columns(pl.col("expression")
                              .str.replace_all(r"\d+\|", "|")
                              .str.replace_all(pattern, "")
                              .alias("expression"))
                .with_columns(pl.col("expression").alias("n.label"))
                .unique())

# 5. add index to each concept
all_concepts_snomed_hug = pl.concat([concept_pre, concept_post], how="vertical").unique()
index_col = pl.Series("idx", np.arange(all_concepts_snomed_hug.height))
all_concepts_snomed_hug = all_concepts_snomed_hug.insert_column(0,index_col)
all_concepts_snomed_hug.write_parquet(PATH_SAVE + "concepts_all_to_embed.parquet")
print("number of unique concepts that are embedded: ", str(all_concepts_snomed_hug["id"].n_unique()))

# 6. embed them, save into matrix (ft and baseline)
expressions = all_concepts_snomed_hug['expression'].to_list()
labels = all_concepts_snomed_hug['n.label'].to_list()

try:
    model_b = SentenceTransformer(MODEL_NAME_B)
    model_ft = SentenceTransformer(MODEL_NAME_FT)

    print("Model and tokenizer loaded successfully!", MODEL_NAME_B, "and", MODEL_NAME_FT)
except Exception as e:
    print("Error loading model:", e)
    exit(1)

embed_and_save(model_b, expressions, FILE_NAME_B, "_expressions")
embed_and_save(model_b, labels, FILE_NAME_B, "_labels") 
embed_and_save(model_ft, expressions, FILE_NAME_FT, "_expressions")
embed_and_save(model_ft, labels, FILE_NAME_FT, "_labels")

print("all embedded and saved successfully!")
