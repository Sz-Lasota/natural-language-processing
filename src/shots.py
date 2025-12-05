import random
import re
from time import time

import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score


def inject_example(text: str, sentiment: str) -> str:
    return f"Tweet: {text}\nSentiment: {sentiment}\n\n"


def inject_sample(text: str) -> str:
    return (
        f"### Now classify the following:\n"
        f"Tweet: {text}\nSentiment:"
    )

def extract_label(result: str, labels: list[str]) -> str:

    for lab in labels:
        if lab in result.lower():
            return lab
    return "unknown"


def run_experiment(
    base_prompt: str, X: np.ndarray, y: np.ndarray, model_name: str,
) -> tuple[list, list]:
    labels = list(np.unique(y))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto",
    )

    y_true = []
    y_pred = []

    for idx, x in enumerate(X):
        prompt = base_prompt + inject_sample(x)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = result.split()[-1]
        pred = extract_label(pred, labels)

        y_true.append(y[idx])
        y_pred.append(pred)

    return y_pred, y_true


def evaluate(
    model: str, base_prompt: str, path: str, encoding: str = "latin1",
    samples: int = 512, shots_n: int = 3,
) -> tuple[float, float]:
    X, y = load_train_dataset(
        path, encoding=encoding, samples=samples + shots_n,
    )

    if shots_n > 0:
        base_prompt += "### Examples\n"

    for i in range(shots_n):
        base_prompt += inject_example(X[i], y[i])

    X, y = X[shots_n:], y[shots_n:]

    y_pred, y_true = run_experiment(base_prompt, X, y, model)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, f1


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_train_dataset(
    path: str, encoding: str = "latin1", samples: int = 512,
) -> tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(path, encoding=encoding)
    df = df.dropna()

    df['preprocessed_text'] = df['selected_text'].apply(clean_text)

    idxes = random.sample(range(df.shape[0]), samples)
    return df['preprocessed_text'].values[idxes], df['sentiment'].values[idxes]



prompt = """Classify the sentiment of the tweet.
Choose only one: positive, negative, neutral.
Reply ONLY with the label. Do not explain.

"""
shots = [0, 1, 3]
reps = 10
samples_n = 1000
result_path = f"../res/shots/results/{round(time())}.csv"

df_prot = {
    "lp": [],
}
for shot in shots:
    df_prot[f"{shot}-shots f1"] = []
    df_prot[f"{shot}-shots acc"] = []


model_name = "facebook/opt-1.3b"

for i in range(reps):
    df_prot["lp"].append(i)
    for shot in shots:
        acc, f1 = evaluate(model_name, prompt, "../res/shots/train.csv", shots_n=shot, samples=samples_n)
        df_prot[f"{shot}-shots f1"].append(f1)
        df_prot[f"{shot}-shots acc"].append(acc)

df = pd.DataFrame(df_prot)
df.to_csv(result_path)
