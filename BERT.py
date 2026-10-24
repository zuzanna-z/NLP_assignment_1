from pathlib import Path
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import torch
import evaluate
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

#------------- Part 3: Train Model----------------
# Training Arguments
batch_size = 8
n_epochs = 4
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

# Evaluation Set-Up
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    """Compute evaluation metrics for fake-job classification."""
    logits, labels = eval_pred
    # Convert model logits to predicted class labels
    predictions = np.argmax(logits, axis=-1)
    # Calculate metrics
    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )["accuracy"]
    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )["f1"]
    precision = precision_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )["recall"]
    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall
    }

# Training & Evaluation
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
print("Training finished. Starting evaluation...")
eval_results = trainer.evaluate()
print("Evaluation results:")
print(eval_results)

# Confusion Matrix
pred_output = trainer.predict(tokenized_val)
y_pred = np.argmax(pred_output.predictions, axis=-1)
y_true = pred_output.label_ids
labels = ["Real", "Fake"]
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
    normalize="true"
)
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
disp.plot(
    cmap="Reds",
    values_format=".2f",
    ax=ax,
    colorbar=False
)
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()