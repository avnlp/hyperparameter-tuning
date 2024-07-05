import transformers
from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import nltk

nltk.download("punkt")


data = load_dataset("billsum")
metric = load_metric("rouge")
model_checkpoints = "facebook/bart-large-xsum"

max_input = 512
max_target = 128
tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoints)


def preprocess_data(data_to_process):
    # get the dialogue text
    inputs = [dialogue for dialogue in data_to_process["document"]]
    # tokenize text
    model_inputs = tokenizer(
        inputs, max_length=max_input, padding="max_length", truncation=True
    )

    # tokenize labels
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            data_to_process["summary"],
            max_length=max_target,
            padding="max_length",
            truncation=True,
        )

    model_inputs["labels"] = targets["input_ids"]
    # reuturns input_ids, attention_masks, labels
    return model_inputs


tokenize_data = data.map(
    preprocess_data, batched=True, remove_columns=["document", "summary"]
)


# load model
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)

batch_size = 32

# Data Collator is used to create batches of data
# collator to create batches. It preprocess data with the given tokenizer
collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

# Metrics
# Compute Rouge for evaluation


def compute_rouge(pred):
    predictions, labels = pred
    # decode the predictions
    decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # decode labels
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # compute results
    res = metric.compute(
        predictions=decode_predictions, references=decode_labels, use_stemmer=True
    )
    # get %
    res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

    pred_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    res["gen_len"] = np.mean(pred_lens)

    return {k: round(v, 4) for k, v in res.items()}


args = transformers.Seq2SeqTrainingArguments(
    "conversation-summ",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    eval_accumulation_steps=1,
)


trainer = transformers.Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenize_data["train"],
    eval_dataset=tokenize_data["validation"],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_rouge,
)

trainer.train()
