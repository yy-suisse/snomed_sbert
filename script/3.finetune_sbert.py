from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import TripletLoss, TripletDistanceMetric
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset_name = "triplet_800000.parquet"

# load data
dataset = load_dataset("parquet", data_files=f"D:/finetune_sbert_new/triplet_sample/" + dataset_name)
dataset_split = dataset['train'].train_test_split(test_size=0.4)
train_dataset = dataset_split['train']
test_validation_split = dataset_split['test'].train_test_split(test_size=0.8) 
test_dataset = test_validation_split['test']
eval_dataset = test_validation_split['train']

# load pretrained model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    # model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    exit(1)

# loss:
loss = TripletLoss(model=model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=0.2)

# training arguments:
args = SentenceTransformerTrainingArguments(
    output_dir="models/snomed_triplet_800k_3_4_3",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    )

# define dev
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    name="snomed_triplet_800k_3_4_3-dev",
)

# train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# evaluate on test set
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="snomed_triplet_800k_3_4_3-dev",
)
test_evaluator(model)

# save the model and push to hub
model.save_pretrained("models/snomed_triplet_800k_3_4_3_final")
model.push_to_hub("snomed_triplet_800k")
