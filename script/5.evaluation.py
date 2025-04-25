import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch



# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def top_k_array_by_batch(id_concept_test, query_matrix, candidate_matrix, batch_size=1000, device=device):

    ranks =[]
    num_concepts = len(id_concept_test)
    num_batches = (num_concepts + batch_size - 1) // batch_size

    query_matrix = torch.tensor(query_matrix, device=device)
    candidate_matrix = torch.tensor(candidate_matrix, device=device)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_concepts)
        batch_indices = id_concept_test[start_idx:end_idx]
        print(f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx})")

        query_batch = query_matrix[batch_indices, :]
        
        # compute the score between the test node and all the candidate nodes
        scores = torch.matmul(query_batch, candidate_matrix.T)

        # Get the ranks for each query in the batch
        sorted_indices = torch.argsort(scores, dim=1, descending=True)  # Updated to use torch.argsort
        for i, test_idx in enumerate(batch_indices):
            if test_idx in sorted_indices[i]:
                rank = torch.where(sorted_indices[i] == test_idx)[0][0].item()
            else:
                rank = -1  # Handle case where test_idx is not found
            ranks.append(rank)

    return np.array(ranks)

def compute_hits_at_k(top_k_array):
    top_k_array_accuracy = []
    num_concept_test = len(top_k_array)
    for i in range(5):
        top_k_array_accuracy.append(sum((top_k_array)<=i)/num_concept_test)
    return np.array(top_k_array_accuracy)

def compute_MMR(top_k_array):
    num_concept_test = len(top_k_array)
    return sum(1/(top_k_array+1))/num_concept_test


def compute_average_rank(ranks):
    return np.mean(ranks) if len(ranks) > 0 else 0

def plot_hits_at_k(top_k_base, top_k_ft, title):
    plt.figure(figsize=(10, 5))
    compute_top_k_accuracy_base = compute_hits_at_k(top_k_base)
    compute_top_k_accuracy_ft = compute_hits_at_k(top_k_ft)
    plt.plot(compute_top_k_accuracy_ft, label="Fine-tuned")
    plt.plot(compute_top_k_accuracy_base, label="Baseline")
    for i, (ft, base) in enumerate(zip(compute_top_k_accuracy_ft, compute_top_k_accuracy_base)):
        plt.text(i, ft - 0.01, f"{ft:.2f}", ha='center', va='bottom', fontsize=9, color='blue')
        plt.text(i, base - 0.01, f"{base:.2f}", ha='center', va='bottom', fontsize=9, color='orange')

    plt.xticks(range(5), ['Top1','Top2','Top3','Top4','Top5'])
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(title + ": Hits@k Comparison")
    plt.show()

def load_model_encoder(model_name):
    try:
        model = SentenceTransformer(model_name)
        print("Model and tokenizer loaded successfully!", model_name)
    except Exception as e:
        print("Error loading model:", e)
        exit(1)
    return model

def top_k_array_with_syn(df_syn, mat_expression, model, batch_size=100):
    top_k = []
    count = 0

    # Move the candidate matrix to GPU
    mat_expression = torch.tensor(mat_expression, device="cuda")

    # Precompute embeddings for all synonyms in batches
    synonym_embeddings = []
    for i in range(0, len(df_syn), batch_size):
        batch_synonyms = df_syn[i:i + batch_size]["term"].to_list()
        batch_embeddings = model.encode(batch_synonyms, convert_to_tensor=True, device="cuda")
        synonym_embeddings.append(batch_embeddings)
    synonym_embeddings = torch.cat(synonym_embeddings, dim=0)

    # Process synonym embeddings in batches
    for i in range(0, len(df_syn), batch_size):
        batch_synonyms = df_syn[i:i + batch_size]
        batch_embeddings = synonym_embeddings[i:i + batch_size]

        # Compute similarity scores for the batch
        scores = torch.matmul(batch_embeddings, mat_expression.T)

        # Get the ranks for each synonym in the batch
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        for j, row in enumerate(batch_synonyms.iter_rows()):
            synonym = row[0]
            label = row[1]
            idx_true = row[2]

            if synonym == label:
                continue

            # Find the rank of the true index
            if idx_true in sorted_indices[j]:
                rank = (sorted_indices[j] == idx_true).nonzero(as_tuple=True)[0].item()
            else:
                rank = -1  # Handle case where idx_true is not found
            top_k.append(rank)

        count += len(batch_synonyms)
        print(f"Processed {count}/{len(df_syn)}")

    return torch.tensor(top_k).cpu().numpy()

def compute_hierarchical_similarity(df_hierarchical_similarity, id2idx, mat_embedding):
    count = 0
    total_rows = len(df_hierarchical_similarity)
    # Create a dictionary to store the similarity scores
    mat_embedding = torch.tensor(mat_embedding, device="cuda")

    accuracy_hierarchical_similarity = []
    
    accuracy = 0
    # Iterate through the DataFrame and compute the similarity scores
    for row in df_hierarchical_similarity.iter_rows():
        count += 1
        if count % 100 == 0:
            print(f"Processing row {count}/{total_rows}...")
        sctid = row[0]
        close_sctid = row[1]
        far_sctid = row[2]

        # Get the indices of the concepts
        idx_sctid = id2idx.get(sctid)
        idx_close_sctid = id2idx.get(close_sctid)
        idx_far_sctid = id2idx.get(far_sctid)

        if idx_sctid is not None and idx_close_sctid is not None and idx_far_sctid is not None:
            # Extract embeddings and reshape to 2D tensors
            emb_sctid = mat_embedding[idx_sctid].unsqueeze(0)  # Shape: [1, embedding_dim]
            emb_close_sctid = mat_embedding[idx_close_sctid].unsqueeze(0)  # Shape: [1, embedding_dim]
            emb_far_sctid = mat_embedding[idx_far_sctid].unsqueeze(0)  # Shape: [1, embedding_dim]

            # Compute the similarity scores
            score_close = torch.cosine_similarity(emb_sctid, emb_close_sctid, dim=-1)
            score_far = torch.cosine_similarity(emb_sctid, emb_far_sctid, dim=-1)

            # Append the comparison result
            accuracy_hierarchical_similarity.append(score_close.item() > score_far.item())

    accuracy = sum(accuracy_hierarchical_similarity) / len(accuracy_hierarchical_similarity) 
    return accuracy

def compute_semantic_composition(df_semantic_composition, id2idx, embeddings_exp_ft, list_idx_all_pre_set):
    top_k_pre = []
    count = 0
    for row in df_semantic_composition.iter_rows():
        print(count, len(df_semantic_composition))
        count+=1
        anchor_id = str(row[0])

        # Skip if anchor_id is not in the index map
        if anchor_id not in id2idx:
            continue

        anchor_idx = id2idx[anchor_id]

        try:
            related_ids = ast.literal_eval(row[1])
        except (ValueError, SyntaxError):
            continue

        related_embedding = np.zeros((embeddings_exp_ft.shape[1],))
        count_valid = 0

        for related_id in related_ids:
            related_id = str(related_id)
            if related_id not in id2idx:
                continue

            related_idx = id2idx[related_id]
            related_embedding += embeddings_exp_ft[related_idx]
            count_valid += 1

        if count_valid == 0:
            continue

        related_embedding /= count_valid

        # Get similarity scores
        similarity_scores = np.dot(related_embedding, embeddings_exp_ft.T)
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Filter sorted indices for fully-defined and pre-defined
        sorted_indices_pre = [idx for idx in sorted_indices if idx in list_idx_all_pre_set]

        # Find rank of anchor in each list (if exists)
        try:
            rank_pre = sorted_indices_pre.index(anchor_idx)
        except ValueError:
            rank_pre = -1  # Not found

        top_k_pre.append(rank_pre)
    return np.array(top_k_pre)

def find_nearest_concept(df_concept_embeddings_all, mat_embedding, list_idx_query, list_idx_dictionary, top_k = 5) :
    # Convert embeddings to torch tensors for faster operations
    mat_embedding_torch = torch.tensor(mat_embedding)
    list_idx_dictionary_torch = torch.tensor(list_idx_dictionary)

    count = 0
    total_rows = len(list_idx_query)
    # Prepare lists for final results
    result_rows = []

    # Process each query index in a vectorized manner
    for idx_query in list_idx_query:
        count += 1
        if count % 100 == 0:
            print(f"Processing row {count}/{total_rows}...")
        # Get the embedding for the query index
        query_embedding = mat_embedding_torch[idx_query]

        # Compute similarity scores with all dictionary embeddings
        dictionary_embeddings = mat_embedding_torch[list_idx_dictionary_torch]
        scores = torch.matmul(query_embedding, dictionary_embeddings.T)

        # Get the top k indices and their corresponding scores
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k)

        # Convert indices back to numpy for filtering in Polars
        top_k_indices_np = list_idx_dictionary_torch[top_k_indices].numpy()

        # Append results to the result list
        anchor_expression = df_concept_embeddings_all.filter(
            pl.col("idx") == idx_query
        )["expression"][0]
        top_concepts = df_concept_embeddings_all.filter(
            pl.col("idx").is_in(top_k_indices_np)
        )["n.label"].to_list()

        # Prepare a row with anchor, top concepts, and their scores
        row = [anchor_expression]
        for concept, score in zip(top_concepts, top_k_scores.numpy()):
            row.extend([concept, score])
        result_rows.append(row)

    # Prepare final column names
    columns = ["anchor"]
    for i in range(1, top_k + 1):
        columns.extend([f"top{i}", f"top{i}_score"])

    # Convert result list to Polars DataFrame
    return pl.DataFrame(result_rows, schema=columns)
    
def find_nearest_concept_label(df_concept_embeddings_all, mat_embedding_exp, mat_embedding_label, list_idx_query, list_idx_dictionary, top_k = 5) :
    # Convert embeddings to torch tensors for faster operations
    mat_embedding_torch_exp = torch.tensor(mat_embedding_exp)
    mat_embedding_torch_label = torch.tensor(mat_embedding_label)
    list_idx_dictionary_torch = torch.tensor(list_idx_dictionary)

    count = 0
    total_rows = len(list_idx_query)
    # Prepare lists for final results
    result_rows = []

    # Process each query index in a vectorized manner
    for idx_query in list_idx_query:
        count += 1
        if count % 100 == 0:
            print(f"Processing row {count}/{total_rows}...")
        # Get the embedding for the query index
        query_embedding = mat_embedding_torch_exp[idx_query]

        # Compute similarity scores with all dictionary embeddings
        dictionary_embeddings = mat_embedding_torch_label[list_idx_dictionary_torch]
        scores = torch.matmul(query_embedding, dictionary_embeddings.T)

        # Get the top k indices and their corresponding scores
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k)

        # Convert indices back to numpy for filtering in Polars
        top_k_indices_np = list_idx_dictionary_torch[top_k_indices].numpy()

        # Append results to the result list
        anchor_expression = df_concept_embeddings_all.filter(
            pl.col("idx") == idx_query
        )["expression"][0]
        top_concepts = df_concept_embeddings_all.filter(
            pl.col("idx").is_in(top_k_indices_np)
        )["n.label"].to_list()

        # Prepare a row with anchor, top concepts, and their scores
        row = [anchor_expression]
        for concept, score in zip(top_concepts, top_k_scores.numpy()):
            row.extend([concept, score])
        result_rows.append(row)

    # Prepare final column names
    columns = ["anchor"]
    for i in range(1, top_k + 1):
        columns.extend([f"top{i}", f"top{i}_score"])

    # Convert result list to Polars DataFrame
    return pl.DataFrame(result_rows, schema=columns)
    



# --------- main script ----------- 
# load the embeddings and concept information 
PATH_EMBEDDINGS = "D:/finetune_sbert_new/embeddings/" 
PATH_LOAD = "D:/finetune_sbert_new/" 

task1_bool = False # retrieve labels given expression
task2_bool = False # retrieve synonyms given expression
task3_bool = False # retrieve labels given synonym 
task4_bool = False # Jamil's hierarchical similarity task (existing benchmark)
task5_bool = True # Jamil's compositional semantic task (existing benchmark)
task6_bool = False # embedding post-coordination and get nearest labels of pre-coordination (using expression of concepts)
task7_bool = False # embedding post-coordination and get nearest expressions of pre-coordination (using label of concepts)

# load concept information 
columns_to_keep = ["id", "expression_cleaned_no_semtag", "label_no_semtag", "term", "term_type"]
df_all_concept_pre_fd = (pl.read_csv(PATH_LOAD + "concept_info/concept_info_pre_fully_def_all.csv", columns = columns_to_keep)
                         .with_columns(pl.col("id").cast(pl.String).alias("id")))
df_concept_embeddings_all = pl.read_parquet(PATH_EMBEDDINGS + "concepts_all_to_embed.parquet")

id2idx = dict(zip(df_concept_embeddings_all["id"], df_concept_embeddings_all["idx"]))

# load embeddings
embeddings_exp_base = np.load(PATH_EMBEDDINGS + "base_expressions.npy")
embeddings_label_base = np.load(PATH_EMBEDDINGS + "base_labels.npy")

embeddings_exp_ft = np.load(PATH_EMBEDDINGS + "ft_expressions.npy")
embeddings_label_ft = np.load(PATH_EMBEDDINGS + "ft_labels.npy")

# load list of concepts that has been in the training set 
df_training = pl.read_parquet(PATH_LOAD +"triplet_sample/triplet_800000.parquet")

# get benchmark Jamil's task 4 and 5
df_hierarchical_similarity = pd.read_table("D:/finetune_sbert_new/benchmark_jamil/hierarchical_similarity_benchmark.tsv")
df_semantic_composition = pd.read_table("D:/finetune_sbert_new/benchmark_jamil/semantic_composition_benchmark.tsv")
df_hierarchical_similarity = pl.from_pandas(df_hierarchical_similarity)
df_semantic_composition = pl.from_pandas(df_semantic_composition)

# get list test concept set
list_id_training_fd = (df_training.join(df_all_concept_pre_fd, 
                                        left_on = "anchor", 
                                        right_on ="expression_cleaned_no_semtag")['id']
                                        .unique()
                                        .to_list())
list_id_all_fd = (df_all_concept_pre_fd['id']
                  .unique()
                  .to_list())
list_id_test_fd = list(set(list_id_all_fd) - set(list_id_training_fd))
list_idx_test_fd = df_concept_embeddings_all.filter(pl.col("id").is_in(list_id_test_fd))['idx'].unique().to_list()

# ----evaluation tasks---- 
# task 1: retrieve labels given expression (find best matching label among all concept labels, using expression of fully defined concepts in test set)
if task1_bool:
    task1 = "retrieve labels given expression"
    top_k_ft_rl = top_k_array_by_batch(list_idx_test_fd, embeddings_exp_ft, embeddings_label_ft, batch_size=1000, device=device)
    top_k_base_rl = top_k_array_by_batch(list_idx_test_fd, embeddings_exp_base, embeddings_label_base, batch_size=1000, device=device)
    plot_hits_at_k(top_k_base_rl, top_k_ft_rl, task1)
    print(task1 + " MMR base: ", compute_MMR(top_k_base_rl))
    print(task1 + " MMR ft: ", compute_MMR(top_k_ft_rl))
    print(task1 + " average rank base: ", compute_average_rank(top_k_base_rl))
    print(task1 + " average rank ft: ", compute_average_rank(top_k_ft_rl))


# task 2: retrieve synonyms given expression (find the best matching synonym among all concept synonyms, using expression of fully defined concepts in test set)
if task2_bool: 
    task2 = "retrieve synonyms given expression"
    from sentence_transformers import SentenceTransformer

    MODEL_NAME_BASE = "all-MiniLM-L6-v2"
    MODEL_NAME_FT = "yyzheng00/snomed_triplet_800k"

    model_base = load_model_encoder(MODEL_NAME_BASE)
    model_ft = load_model_encoder(MODEL_NAME_FT)

    df_synonym = (df_all_concept_pre_fd
                  .filter(pl.col("term_type") == "synonym")
                  .filter(pl.col('id').is_in(list_id_test_fd))
                  .join(df_concept_embeddings_all, left_on = "id", right_on ="id", how="left")
                  .select(pl.col("term"), pl.col("label_no_semtag"), pl.col("idx")))
    top_k_ft_synonym = top_k_array_with_syn(df_synonym, embeddings_exp_ft, model_ft, batch_size=100)
    top_k_base_synonym = top_k_array_with_syn(df_synonym, embeddings_exp_base, model_base, batch_size=100)
    plot_hits_at_k(top_k_base_synonym, top_k_ft_synonym, task2)
    print(task2 + " MMR base: ", compute_MMR(top_k_base_synonym))
    print(task2 + " MMR ft: ", compute_MMR(top_k_ft_synonym))
    print(task2 + " average rank base: ", compute_average_rank(top_k_base_synonym))
    print(task2 + " average rank ft: ", compute_average_rank(top_k_ft_synonym))

# task 3: retrieve labels given synonym (find the best matching label among all concept labels, using synonyms of fully defined concepts in test set)
if task3_bool:
    task3 = "retrieve labels given synonym"
    from sentence_transformers import SentenceTransformer

    MODEL_NAME_BASE = "all-MiniLM-L6-v2"
    MODEL_NAME_FT = "yyzheng00/snomed_triplet_800k"

    model_base = load_model_encoder(MODEL_NAME_BASE)
    model_ft = load_model_encoder(MODEL_NAME_FT)

    df_synonym = (df_all_concept_pre_fd
                  .filter(pl.col("term_type") == "synonym")
                  .filter(pl.col('id').is_in(list_id_test_fd))
                  .join(df_concept_embeddings_all, left_on = "id", right_on ="id", how="left")
                  .select(pl.col("term"), pl.col("label_no_semtag"), pl.col("idx")))
    top_k_ft_synonym_label = top_k_array_with_syn(df_synonym, embeddings_label_ft, model_ft, batch_size=100)
    top_k_base_synonym_label = top_k_array_with_syn(df_synonym, embeddings_label_base, model_base, batch_size=100)
    plot_hits_at_k(top_k_base_synonym_label, top_k_ft_synonym_label, task3)
    print(task3 + " MMR base: ", compute_MMR(top_k_base_synonym_label))
    print(task3 + " MMR ft: ", compute_MMR(top_k_ft_synonym_label))
    print(task3 + " average rank base: ", compute_average_rank(top_k_base_synonym_label))
    print(task3 + " average rank ft: ", compute_average_rank(top_k_ft_synonym_label))

# task 4: Jamil's hierarchical similarity task (existing benchmark)
if task4_bool:
    task4 = "Jamil's hierarchical similarity task"
    df_hierarchical_similarity =( df_hierarchical_similarity.select(pl.col("sctid").cast(pl.String)
                                                                      , pl.col("close_sctid").cast(pl.String)
                                                                      , pl.col("far_sctid").cast(pl.String)))
    hs_exp_b = compute_hierarchical_similarity(df_hierarchical_similarity, id2idx, embeddings_exp_base)
    hs_exp_ft = compute_hierarchical_similarity(df_hierarchical_similarity, id2idx, embeddings_exp_ft)
    hs_label_b = compute_hierarchical_similarity(df_hierarchical_similarity, id2idx, embeddings_label_base)
    hs_label_ft = compute_hierarchical_similarity(df_hierarchical_similarity, id2idx, embeddings_label_ft)
    print(task4 + " accuracy using base expression: ", hs_exp_b)
    print(task4 + " accuracy using ft expression: ", hs_exp_ft)
    print(task4 + " accuracy using base label: ", hs_label_b)
    print(task4 + " accuracy using ft label: ", hs_label_ft)
    
# task 5: Jamil's compositional semantic task (existing benchmark)
if task5_bool:
    import ast
    task5 = "Jamil's compositional semantic task"
    list_idx_all_pre = df_concept_embeddings_all.filter(pl.col("concept_type") == "SCT_PRE")['idx'].unique().to_list()
    list_idx_all_pre_set  = set(list_idx_all_pre)
    df_semantic_composition = (df_semantic_composition
                               .select(pl.col("id_node"), pl.col("parents_ids")))

    mr_semantic_composition_exp_ft = compute_semantic_composition(df_semantic_composition,
                                  id2idx, 
                                  embeddings_exp_ft, 
                                  list_idx_all_pre_set)
    mr_semantic_composition_label_ft = compute_semantic_composition(df_semantic_composition,
                                id2idx, 
                                embeddings_label_ft, 
                                list_idx_all_pre_set)
    mr_semantic_composition_exp_b = compute_semantic_composition(df_semantic_composition,
                                id2idx, 
                                embeddings_exp_base, 
                                list_idx_all_pre_set)
    mr_semantic_composition_label_b = compute_semantic_composition(df_semantic_composition,
                                id2idx, 
                                embeddings_label_base, 
                                list_idx_all_pre_set)
    
    print(task5 + " main rank using ft expression: ", mr_semantic_composition_exp_ft.mean())
    print(task5 + " main rank using base label: ", mr_semantic_composition_label_b.mean())
    print(task5 + " main rank using ft label: ", mr_semantic_composition_label_ft.mean())
    print(task5 + " main rank using base expression: ", mr_semantic_composition_exp_b.mean())
    print(task3 + " MMR base label: ", compute_MMR(mr_semantic_composition_label_b))
    print(task3 + " MMR ft label: ", compute_MMR(mr_semantic_composition_label_ft))    
    print(task3 + " MMR base exp: ", compute_MMR(mr_semantic_composition_exp_b))
    print(task3 + " MMR ft exp: ", compute_MMR(mr_semantic_composition_exp_ft))
# task 6: embedding post-coordination and get nearest labels of pre-coordination (using expression of concepts)
if task6_bool:
    task6 = "embedding post-coordination and get nearest labels of pre-coordination"
    list_idx_all_post = df_concept_embeddings_all.filter(pl.col("concept_type") == "SCT_POST")['idx'].unique().to_list()
    list_idx_all_pre = df_concept_embeddings_all.filter(pl.col("concept_type") == "SCT_PRE")['idx'].unique().to_list()
    list_id_all_pre_fd = (df_all_concept_pre_fd['id']
                  .unique()
                  .to_list())
    list_idx_all_pre_fd = df_concept_embeddings_all.filter(pl.col("concept_type") == "SCT_PRE").filter(pl.col("id").is_in(list_id_all_pre_fd))["idx"].unique().to_list()
    
    df_nearest_pre = find_nearest_concept(df_concept_embeddings_all, embeddings_exp_ft, list_idx_all_post, list_idx_all_pre, 5)
    df_nearest_pre_fd = find_nearest_concept(df_concept_embeddings_all, embeddings_exp_ft, list_idx_all_post, list_idx_all_pre_fd, 5)
    df_nearest_pre.write_csv(PATH_LOAD + "evaluation/nearest_pre_post.csv")
    df_nearest_pre_fd.write_csv(PATH_LOAD + "evaluation/nearest_pre_post_fd.csv")

# task 7: embedding post-coordination and get nearest expressions of pre-coordination (using label of concepts)
if task7_bool:
    task7 = "embedding post-coordination and get nearest expressions of pre-coordination (using label of concepts)"
    list_idx_all_post = df_concept_embeddings_all.filter(pl.col("concept_type") == "SCT_POST")['idx'].unique().to_list()
    list_idx_all_pre = df_concept_embeddings_all.filter(pl.col("concept_type") == "SCT_PRE")['idx'].unique().to_list()
    list_id_all_pre_fd = (df_all_concept_pre_fd['id']
                  .unique()
                  .to_list())
    list_idx_all_pre_fd = df_concept_embeddings_all.filter(pl.col("concept_type") == "SCT_PRE").filter(pl.col("id").is_in(list_id_all_pre_fd))["idx"].unique().to_list()
    
    df_nearest_pre = find_nearest_concept_label(df_concept_embeddings_all, embeddings_exp_ft, embeddings_label_ft, list_idx_all_post, list_idx_all_pre, top_k = 5)
    df_nearest_pre_fd = find_nearest_concept_label(df_concept_embeddings_all, embeddings_exp_ft, embeddings_label_ft, list_idx_all_post, list_id_all_pre_fd, top_k = 5)
    df_nearest_pre.write_csv(PATH_LOAD + "evaluation/nearest_pre_post_using_label.csv")
    df_nearest_pre_fd.write_csv(PATH_LOAD + "evaluation/nearest_pre_post_fd_using_label.csv")