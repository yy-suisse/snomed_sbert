import random
from tqdm import tqdm
import numpy as np
import networkx as nx


def hard_triplet(df_concepts, df_relations_is_a_def):
    id_concept = df_concepts["id"].unique()
    seen = set()
    rows_hard = []

    # Build parent -> children and child -> parents maps
    child_to_parents = {}
    parent_to_children = {}

    for row in df_relations_is_a_def.iter_rows(named=True):
        child = row['src.id']
        parent = row['dst.id']
        child_to_parents.setdefault(child, []).append(parent)
        parent_to_children.setdefault(parent, []).append(child)

    for id_anchor in tqdm(id_concept):
        if id_anchor not in child_to_parents:
            continue

        for parent in child_to_parents[id_anchor]:
            siblings = parent_to_children.get(parent, [])
            for sibling_id in siblings:
                if sibling_id == id_anchor:
                    continue
                pair = (id_anchor, sibling_id)
                if pair in seen:
                    continue
                seen.add(pair)
                try:
                    rows_hard.append({
                        "anchor" : id_anchor,
                        "positive": id_anchor,
                        "negative": sibling_id,
                        "level": "hard"
                    })
                except KeyError:
                    continue  # skip missing values

    return rows_hard


def medium_triplet(df_concepts, df_relations_is_a_def):
    df_relations_is_a_def = df_relations_is_a_def.unique()
    src = df_relations_is_a_def['src.id'].to_list()
    dst = df_relations_is_a_def['dst.id'].to_list()

    edges = list(zip(src, dst))
    G = nx.Graph()
    G.add_edges_from(edges)
    
    id_concept = df_concepts["id"].unique()
    id_to_semtag = dict(zip(df_concepts["id"], df_concepts["sem_tag"]))

    rows_medium = []

    for id_anchor in tqdm(id_concept):
        if id_anchor not in G:
            continue

        anchor_tag = id_to_semtag.get(id_anchor)
        if anchor_tag is None:
            continue

        nearby_concepts = nx.single_source_shortest_path_length(G, id_anchor, cutoff=3)
        level_3_candidates = [
            cid for cid, dist in nearby_concepts.items()
            if dist == 3 and id_to_semtag.get(cid) == anchor_tag
        ]
        if not level_3_candidates:
            continue

        sampled_negatives = random.sample(level_3_candidates, min(100, len(level_3_candidates)))

        for candidate_id in sampled_negatives:
            try:
                rows_medium.append({
                    "anchor": id_anchor,
                    "positive": id_anchor,
                    "negative": candidate_id,
                    "level": "medium"
                })
            except KeyError:
                continue  # skip missing data

    return rows_medium


def easy_triplet(df_concepts):
    id_concept = df_concepts["id"].unique()
    
    rows_easy = []

    # Build fast access lookup tables
    id_to_top_cat = dict(zip(df_concepts["id"], df_concepts["top_category"]))
  
    # Pre-group concept IDs by top category
    top_cat_to_ids = {}
    for id_, top_cat in id_to_top_cat.items():
        top_cat_to_ids.setdefault(top_cat, []).append(id_)

    # Flatten all IDs once for sampling
    all_ids = set(id_concept)

    for id_anchor in tqdm(id_concept):
        anchor_top_cat = id_to_top_cat.get(id_anchor)
        if anchor_top_cat is None:
            continue

        # Get IDs that are NOT in the same top category
        in_same_cat = set(top_cat_to_ids.get(anchor_top_cat, []))
        negative_candidates = list(all_ids - in_same_cat)

        if not negative_candidates:
            continue

        sampled_negatives = np.random.choice(
            negative_candidates,
            size=min(50, len(negative_candidates)),
            replace=False
        )

        for neg_id in sampled_negatives:
            try:
                rows_easy.append({
                    "anchor": id_anchor,
                    "positive": id_anchor,
                    "negative": neg_id,
                    "level": "easy"
                })
            except KeyError:
                continue  # Skip missing values

    return rows_easy