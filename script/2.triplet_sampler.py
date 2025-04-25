import polars as pl
import pickle

path_triplet_load = "D:/finetune_sbert_new/triplet_sample/"
path_concept_info_load = "D:/finetune_sbert_new/concept_info/"

num_sample = 800000
ratio = [0.3, 0.3, 0.4]

easy_samples = pl.read_csv(path_triplet_load +"easy_samples.csv").unique()
medium_samples = pl.read_csv(path_triplet_load + "medium_samples.csv").unique()
hard_samples = pl.read_csv(path_triplet_load +"hard_samples.csv").unique()

with open(path_concept_info_load + "id_to_expr.pkl", "rb") as f:
    id_to_expr = pickle.load(f)

with open(path_concept_info_load + "id_to_label_syn.pkl", "rb") as f:
    id_to_label_syn = pickle.load(f)


df_label_syn = pl.DataFrame({
    "id": list(id_to_label_syn.keys()),
    "terms": list(id_to_label_syn.values())
}).explode("terms")


easy_sampled = easy_samples.sample(n=int(num_sample * ratio[0]), with_replacement=False)
medium_sampled = medium_samples.sample(n=int(num_sample * ratio[1]), with_replacement=False)
hard_sampled = hard_samples.sample(n=int(num_sample * ratio[2]), with_replacement=False)

# Concatenate the three sampled DataFrames
all_samples = pl.concat([easy_sampled, medium_sampled, hard_sampled])
# Get expression and label (or synonym)
all_samples = (all_samples.select(pl.col("anchor").cast(pl.String)
                                  ,pl.col("positive").cast(pl.String)
                                  ,pl.col("negative").cast(pl.String)))

# 
all_samples = (all_samples
 .with_columns(pl.col("anchor").replace(id_to_expr).alias("anchor_exp"))
 .join(df_label_syn, left_on="positive", right_on="id")
 .rename({"terms": "positive_label"})
 .join(df_label_syn, left_on="negative", right_on="id")
 .rename({"terms": "negative_label"})).sample(num_sample,shuffle=True, seed=1)

all_samples = all_samples.drop(pl.col("anchor"), pl.col("positive"), pl.col("negative")).rename({
    "anchor_exp":"anchor",
    "positive_label":"positive",
    "negative_label":"negative"
})

dataset_name = "triplet_" + str(num_sample)+ ".parquet"
all_samples.write_parquet(path_triplet_load + dataset_name)