from datasets import load_dataset
import torch
from tqdm.auto import tqdm
from transformers import BertTokenizerFast

# Load certain rows of squad dataset

data = load_dataset("squad")


# Function to add the start and end index for answer context pair
def add_end_idx(answers, contexts):
    new_answers = []
    # loop through each answer-context pair
    for answer, context in tqdm(zip(answers, contexts)):
        # quick reformating to remove lists
        answer["text"] = answer["text"][0]
        answer["answer_start"] = answer["answer_start"][0]
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer["text"]
        # we already know the start index
        start_idx = answer["answer_start"]
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer["answer_end"] = end_idx
        else:
            # this means the answer is off by 1-2 tokens
            for n in [1, 2]:
                if context[start_idx - n : end_idx - n] == gold_text:
                    answer["answer_start"] = start_idx - n
                    answer["answer_end"] = end_idx - n
        new_answers.append(answer)
    return new_answers


def prep_data(dataset):
    questions = dataset["question"]
    contexts = dataset["context"]
    answers = add_end_idx(dataset["answers"], contexts)
    return {"question": questions, "context": contexts, "answers": answers}


dataset = prep_data(data["train"].shuffle(seed=123).select(range(1000)))


# The data format is now ready for tokenization.
# Tokenization

# We need to tokenize the SQuAD data so that it is readable by our Bert model. For the context and question features we can do using the standard tokenizer() function:


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train = tokenizer(
    dataset["context"],
    dataset["question"],
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
)


tokenizer.decode(train["input_ids"][0])[:855]


def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in tqdm(range(len(answers))):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
        end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"]))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(
                i, answers[i]["answer_end"] - shift
            )
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update(
        {"start_positions": start_positions, "end_positions": end_positions}
    )


# apply function to our data
add_token_positions(train, dataset["answers"])


# Which encodes both our context and question strings into single arrays of tokens. This will act as the input to our Q&A training, but we have no targets yet.

# Our targets are the start and end positions of the answer,
# which we previously built using the character start and end positions within the context strings.
# However, we will be feeding tokens into Bert, so we need to provide the token start and end positions.

# To do this, we need to convert the character start and end positions into token start and end positions —
# easily done with our add_token_positions function:


# apply function to our data
add_token_positions(train, dataset["answers"])

# This function adds two more tensors to our Encoding object (which we feed into Bert)
# — the start_positions and end_positions.
train.keys()

train["start_positions"][:5], train["end_positions"][:5]

# Our tensors are now ready for training the Bert Q&A head.
# Training

# We will be training using PyTorch, which means we will need to convert the tensors we’ve built into a PyTorch Dataset object.


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = SquadDataset(train)


# We will feed our Dataset to our Q&A training loop using a Dataloader object, which we initialize with:

loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

from transformers import BertForQuestionAnswering

# Loading the model
model = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.train()

# Defining the optimizer for training
optimizer1 = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

epochs = 5


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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer3.step()
        scheduler.step()

        progress_bar.set_postfix(
            {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
        )

    torch.save(model.state_dict(), f"finetuned_finBERT_epoch_{epoch}.model")

    tqdm.write(f"\nEpoch {epoch}")
    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f"Training loss: {loss_train_avg}")

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)

    tqdm.write(f"Validation loss: {val_loss}")
    tqdm.write(f"F1 Score (Weighted): {val_f1}")
    # print(train_acc = torch.sum(y_pred == true_vals))

# Function to evaluate the model performance
model.eval()

acc = []

for batch in tqdm(valid_loader):
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_true = batch["start_positions"].to(device)
        end_true = batch["end_positions"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        start_pred = torch.argmax(outputs["start_logits"], dim=1)
        end_pred = torch.argmax(outputs["end_logits"], dim=1)

        acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
        acc.append(((end_pred == end_true).sum() / len(end_pred)).item())

acc = sum(acc) / len(acc)

print("\n\nT/P\tanswer_start\tanswer_end\n")
for i in range(len(start_true)):
    print(
        f"true\t{start_true[i]}\t{end_true[i]}\n"
        f"pred\t{start_pred[i]}\t{end_pred[i]}\n"
    )


def get_prediction(context, question):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt").to(device)
    outputs = model(**inputs)

    answer_start = torch.argmax(outputs[0])
    answer_end = torch.argmax(outputs[1]) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    return answer


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, truth):
    return bool(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return round(2 * (prec * rec) / (prec + rec), 2)


def question_answer(context, question, answer):
    prediction = get_prediction(context, question)
    em_score = exact_match(prediction, answer)
    f1_score = compute_f1(prediction, answer)

    print(f"Question: {question}")
    print(f"Prediction: {prediction}")
    print(f"True Answer: {answer}")
    print(f"Exact match: {em_score}")
    print(f"F1 score: {f1_score}\n")
