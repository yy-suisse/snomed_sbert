import polars as pl
import pandas as pd
import re
import sys
import os

# Add the parent directory (repo root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.negative_sampling import *
import pickle
from src.negative_sampling import *

extract_hard = False
extract_medium = False
extract_easy = False

save_concept_info = True

path_load_data = "D:/finetune_sbert_new/graph_data/"
path_save_info = "D:/finetune_sbert_new/concept_info/"
path_save_triplet = "D:/finetune_sbert_new/triplet_sample/"


# import source data : pre-coordination concept info and connectivity
df_concepts = pl.read_parquet(path_load_data + "pre_with_expression.parquet")
df_relations = pd.read_csv(path_load_data + "connectivity_2025-04-01.csv")
df_relations['src.id'] = df_relations['src.id'].astype(str)
df_relations['dst.id'] = df_relations['dst.id'].astype(str)
df_relations = pl.from_pandas(df_relations)

# preprocessing concept info and save it
list_sem_tag = df_concepts['sem_tag'] .unique().to_list() + ["attribute"]
escaped_tags = [re.escape(tag) for tag in list_sem_tag]
pattern = r"\s*\((" + "|".join(escaped_tags) + r")\)"
df_concepts = (((df_concepts.filter(pl.col("concept_type") == "SCT_PRE")
               .filter(pl.col("status") == "defined")
               .with_columns(pl.col("expression").str.replace_all(r"\d+\|", "|").alias("expression_cleaned")) # remove digits
               ).with_columns(pl.col("expression_cleaned").str.replace_all(pattern, "").alias("expression_cleaned_no_semtag")))
               .with_columns(pl.col("n.label").str.replace_all(pattern, "").alias("label_no_semtag"))
               .unique())

if save_concept_info: df_concepts.write_csv(path_save_info + "concept_info_pre_fully_def_all.csv")


# build lookup dictionary id to expression and label/synonyme
id_to_expr = dict(zip(df_concepts["id"], df_concepts["expression_cleaned_no_semtag"]))

df_concept_fsn_syn = df_concepts.group_by("id").agg([
    pl.concat_list([pl.col("label_no_semtag"), pl.col("term")])
      .list.explode()      # Flatten all nested lists into individual elements
      .unique()            # Get unique values             
    .alias("merged_terms")
])
id_to_label_syn = dict(zip(df_concept_fsn_syn["id"], df_concept_fsn_syn["merged_terms"].to_list()))

if save_concept_info:
    ## Save id_to_expr
    with open(path_save_info + "id_to_expr.pkl", "wb") as f:
        pickle.dump(id_to_expr, f)

    ## Save id_to_label_syn
    with open(path_save_info + "id_to_label_syn.pkl", "wb") as f:
        pickle.dump(id_to_label_syn, f)

# preprocess relations in graph for negative sampling
df_relations_is_a = df_relations.filter(pl.col("type(r)") == "IS_A")
df_concepts_pos = df_concepts.select(pl.col("label_no_semtag"), pl.col("expression_cleaned_no_semtag"), pl.col("id") )
df_relations_is_a_def = ((df_relations_is_a
 .select(pl.col("src.id"),pl.col("dst.id"))
 .join(df_concepts_pos, left_on= "src.id", right_on="id", how="left")
 .drop_nulls()
 .rename({"expression_cleaned_no_semtag": "src.expression_cleaned_no_semtag","label_no_semtag": "src.label_no_semtag"}))
 .join(df_concepts_pos, left_on= "dst.id", right_on="id", how="left")
 .drop_nulls()
 .rename({"expression_cleaned_no_semtag": "dst.expression_cleaned_no_semtag","label_no_semtag": "dst.label_no_semtag"})
 ).unique().select(pl.col("src.id"),pl.col("dst.id"))

# extract samples
if extract_hard:
    # extraction of negative samples:
    rows_hard = hard_triplet(df_concepts, df_relations_is_a_def)
    pl.DataFrame(rows_hard).write_csv(path_save_triplet + "hard_samples.csv")

if extract_medium:
    rows_medium = medium_triplet(df_concepts, df_relations_is_a_def)
    df_rows_medium = pl.DataFrame(rows_medium)
    df_rows_medium.write_csv(path_save_triplet + "medium_samples.csv")

if extract_easy:
    rows_easy = easy_triplet(df_concepts)
    df_rows_easy = pd.DataFrame(rows_easy)
    pl.DataFrame(df_rows_easy).write_csv(path_save_triplet + "easy_samples.csv")