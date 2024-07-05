import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import logging

logging.set_verbosity_error()

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

# Load the dataset

financial_data = pd.read_csv("stokctwits.csv")


X_train, X_val, y_train, y_val = train_test_split(
    financial_data.text,
    financial_data.polarity,
    test_size=0.20,
    random_state=2022,
    stratify=financial_data.label.values,
)

# Get the BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


encoded_data_train = tokenizer.batch_encode_plus(
    X_train.values,
    return_tensors="pt",
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=150,
)

encoded_data_val = tokenizer.batch_encode_plus(
    X_val.values,
    return_tensors="pt",
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=150,
)


input_ids_train = encoded_data_train["input_ids"]
attention_masks_train = encoded_data_train["attention_mask"]
labels_train = y_train.values

input_ids_val = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val["attention_mask"]
sentiments_val = y_val.values


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, sentiments_val)


batch_size = 32

dataloader_train = DataLoader(
    dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size
)

dataloader_validation = DataLoader(
    dataset_val, sampler=RandomSampler(dataset_val), batch_size=batch_size
)

model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert", num_labels=len(sentiment_dict)
)


epochs = 3
optimizer1 = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(
    optimizer1, num_warmup_steps=0, num_training_steps=len(dataloader_train) * epochs
)


seed_val = 2022
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs + 1)):
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(
        dataloader_train, desc="Epoch {:1d}".format(epoch), leave=False, disable=False
    )
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        # Gradient Clipping is done to restrict the values of the gradient(To prevent the model from exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer1.step()
        scheduler.step()

        progress_bar.set_postfix(
            {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
        )

    torch.save(model.state_dict(), f"finetuned_BERT_epoch_{epoch}.model")

    tqdm.write(f"\nEpoch {epoch}")

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f"Training loss: {loss_train_avg}")

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score(predictions, true_vals, average="weighted")
    tqdm.write(f"Validation loss: {val_loss}")
    tqdm.write(f"F1 Score (Weighted): {val_f1}")


# Load the best model & Make Predictions

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

model.to(device)

model.load_state_dict(
    torch.load("finetuned_model.model", map_location=torch.device("cpu"))
)

_, predictions, true_vals = evaluate(dataloader_validation)

print("Accuracy: ", accuracy_score(predictions, true_vals))
