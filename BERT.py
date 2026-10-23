from pathlib import Path
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import torch
import evaluate
import numpy as np
from collections import Counter

#---------- Part 1: Data Prep------------------------
# Load and cache dataset
path = Path(__file__).resolve().parent
data_path = path / "data"
ds = load_dataset(
    "serenityyyyy/fake_job_postings_balanced_en",
    cache_dir=data_path
)
print(ds)

# Define Label Column
num_classes = 2
ds = ds.cast_column(
    "fraudulent",
    ClassLabel(
        num_classes=num_classes,
        names=["real", "fake"]
    )
)

# Split into train, validation, test (stratified to keep balance)
ds = ds["train"]
train_test = ds.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="fraudulent"
)
val_test = train_test["test"].train_test_split(
    test_size=0.5,
    seed=42,
    stratify_by_column="fraudulent"
)
train_ds = train_test["train"]
val_ds = val_test["train"]
test_ds = val_test["test"]
print("Train:", len(train_ds))
print("Validation:", len(val_ds))
print("Test:", len(test_ds))

# Check balance
for name, subset in [
    ("Train", train_ds),
    ("Validation", val_ds),
    ("Test", test_ds)
]:
    counts = Counter(subset["fraudulent"])
    print(f"{name}: {counts}")

#---------- Part 2: Model Prep------------------------
# Define and cache model
model_path = path / "models" / "hf"
model_id = "distilbert/distilbert-base-cased"

# GPU or CPU? Default to CPU if no GPU available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_classes,
    cache_dir=model_path
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=model_path
)
print("Model parameters:", model.num_parameters())
print(train_ds.column_names)

# Tokenization
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
def preprocess_function(examples):
    """Combine job-posting fields and tokenize them."""

    texts = []

    for i in range(len(examples["title"])):
        fields = []

        for column in text_columns:
            value = examples[column][i]

            if value is not None and str(value).strip():
                fields.append(f"{column}: {value}")

        texts.append("\n".join(fields))

    return tokenizer(
        texts,
        truncation=True,
        max_length=512
    )

# Tokenize datasets
tokenized_train = train_ds.map(
    preprocess_function,
    batched=True
)
tokenized_val = val_ds.map(
    preprocess_function,
    batched=True
)
tokenized_test = test_ds.map(
    preprocess_function,
    batched=True
)
tokenized_train = tokenized_train.rename_column(
    "fraudulent", "labels"
)
tokenized_val = tokenized_val.rename_column(
    "fraudulent", "labels"
)
tokenized_test = tokenized_test.rename_column(
    "fraudulent", "labels"
)

#------------- Part 3: Train Model----------
# Training Arguments
batch_size = 8
n_epochs = 4  # for demo purposes, MAY increase performance with more epochs
output_dir = path / "training" / "distilbert_jobpostings"

training_args = TrainingArguments(
   output_dir=output_dir,
   learning_rate=2e-5,
   per_device_train_batch_size=batch_size,
   per_device_eval_batch_size=batch_size,
   num_train_epochs=n_epochs,
   weight_decay=0.01,
   save_strategy="epoch",
   report_to="none"
)

# Padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluation
def compute_metrics(eval_pred):
    """Take predicted logits and true labels to compute F1 score"""
    logits, labels = eval_pred

    # convert to predicted class label (index of highest logit)
    predictions = np.argmax(logits, axis=-1)

    # load and compute F1 from "evaluate" library
    f1_metric = evaluate.load("f1")
    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"  # or "weighted"
    )["f1"]
    
    return {"f1": f1}

# Train
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_val,
   processing_class=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()