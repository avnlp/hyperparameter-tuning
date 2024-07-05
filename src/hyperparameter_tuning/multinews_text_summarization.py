import transformers
from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import nltk

data = load_dataset("multi_news")
metric = load_metric("rouge")
model_checkpoints = "facebook/bart-large-xsum"


# Multi-News, consists of news articles and human-written summaries of these articles from the site newser.com. Each summary is professionally written by editors and includes links to the original articles cited.
#
# There are two features:
#
#     document: text of news articles seperated by special token "|||||".
#     summary: news summary.
#

# ## Data tokenization
#
# **max_input** and **max_target** can vary depending on the available computing power

max_input = 512
max_target = 128
tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoints)


def preprocess_data(data_to_process):
    """
    Tokenizes the input and target text data using a pre-trained tokenizer, and formats it as input to a language model.

    Args:
        data_to_process (dict): A dictionary containing the text data to be preprocessed, with the following keys:
            - 'document' (list of str): A list of input documents to be summarized.
            - 'summary' (list of str): A list of target summaries for each input document.

    Returns:
        dict: A dictionary with the following keys and values:
            - 'input_ids' (list of list of int): A list of input token IDs for each document, padded to `max_length`.
            - 'attention_mask' (list of list of int): A list of binary attention masks indicating which tokens to attend to for each document.
            - 'labels' (list of list of int): A list of target token IDs for each document, padded to `max_target`.
    """

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


# sample the data
train_sample = tokenize_data["train"].shuffle(seed=123).select(range(1000))
validation_sample = tokenize_data["validation"].shuffle(seed=123).select(range(500))
test_sample = tokenize_data["test"].shuffle(seed=123).select(range(200))


tokenize_data["train"] = train_sample
tokenize_data["validation"] = validation_sample
tokenize_data["test"] = test_sample


# load model
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)

batch_size = 32

# Data Collator is used to create batches of data
# collator to create batches. It preprocess data with the given tokenizer
collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

# Metrics
# Compute Rouge for evaluation


def compute_rouge(pred):
    """
    Computes the ROUGE metric scores and the average length of the generated predictions.

    Args:
    pred : A tuple containing two lists of integer tokens.
        The first list contains the predicted tokens, and the second list contains the actual tokens (labels).

    Returns:
    A dictionary containing the ROUGE scores for precision, recall, and F1, as well as the average length of the predictions.
    """
    predictions, labels = pred
    # decode the predictions
    decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # decode labels
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # compute results
    res = metric.compute(
        predictions=decode_predictions, references=decode_labels, use_stemmer=True
    )
    res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

    pred_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    res["gen_len"] = np.mean(pred_lens)

    return {k: round(v, 4) for k, v in res.items()}


# class transformers.Seq2SeqTrainingArguments
# The evaluation strategy to adopt during training. Possible values are:
# "no": No evaluation is done during training.
# "steps": Evaluation is done (and logged) every eval_steps.
# "epoch": Evaluation is done at the end of each epoch.
# The initial learning rate for AdamW optimizer.
# per_device_train_batch_size (int, optional, defaults to 8) — The batch size per GPU/TPU core/CPU for training.
# Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
# The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
# If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
# num_train_epochs(float, optional, defaults to 3.0) — Total number of training epochs to perform
# Whether to use generate to calculate generative metrics (ROUGE, BLEU).

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

# Class Seq2SeqTrainer
# eval_dataset (Dataset, optional) — Pass a dataset if you wish to override self.eval_dataset. If it is an Dataset, columns not accepted by the model.forward() method are automatically removed. It must implement the __len__ method.
# args (TrainingArguments, optional) — The arguments to tweak for training. Will default to a basic instance of TrainingArguments with the output_dir set to a directory named tmp_trainer in the current directory if not provided.
# data_collator (DataCollator, optional) — The function to use to form a batch from a list of elements of train_dataset or eval_dataset. Will default to default_data_collator() if no tokenizer is provided, an instance of DataCollatorWithPadding otherwise.
# eval_dataset (Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]), optional) — The dataset to use for evaluation. If it is a Dataset, columns not accepted by the model.forward() method are automatically removed. If it is a dictionary, it will evaluate on each dataset prepending the dictionary key to the metric name.
# tokenizer (PreTrainedTokenizerBase, optional) — The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an interrupted training or reuse the fine-tuned model.
# compute_metrics (Callable[[EvalPrediction], Dict], optional) — The function that will be used to compute metrics at evaluation.

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


### Testing the fine tuned model

conversation = """
Rann: Hey Harry, how have you been? Long time no see!
Harry: Hey! What a surprise! 
Harry: Yes, you are right, we haven’t seen each other in a long time. How have you been?
Rann: There is an important campaign next week which is keeping me busy otherwise rest is going good in my life. 
Rann: How about you?
Harry: Oh! I just finished a meeting with a very important client of mine and now I finally have some free time. I feel relieved that I’m done with it.
Rann: Good for you then. Hey! Let’s make a plan and catch up with each other after next week. 
Rann: What do you say?
Harry: Sure, why not? Give me a call when you are done with your project.
Rann: Sure, then. 
Rann: Bye, take care.
Harry: Bye buddy.
"""

model_inputs = tokenizer(
    conversation, max_length=max_input, padding="max_length", truncation=True
)

model_inputs

raw_pred, _, _ = trainer.predict([model_inputs])

raw_pred


tokenizer.decode(raw_pred[0])
