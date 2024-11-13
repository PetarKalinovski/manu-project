import pandas as pd
from sklearn.model_selection import train_test_split
#%%
df=pd.read_csv("/content/politifact_fixed.csv")
#%%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#%%
df.head()
#%%
df=df.drop(columns='Unnamed: 0')
#%%
for i in df["content_sentance"]:
  if isinstance(i,int):
    print(i)
#%%
df=df.dropna()
#%%
df.head()
#%%
from datasets import Dataset
#%%
#%%
#%%
df.columns = ['label', 'text']
#%%
df
#%%
dataset = Dataset.from_pandas(df,preserve_index=False)
#%%
train_test_split = dataset.train_test_split(test_size=0.6, seed=42)
#%%
test_unsupervised_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
#%%
from datasets import DatasetDict
#%%
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': test_unsupervised_split['train'],
    'unsupervised': test_unsupervised_split['test']
})
#%%
dataset_dict
#%%
def preprocess_function(examples):
    if isinstance(examples["text"], str):
        return tokenizer(examples["text"], truncation=True)
    else:
        return tokenizer("N/A", truncation=True)
#%%
df_tokenized = dataset_dict.map(preprocess_function)
#%%
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
#%%
import numpy as np
#%%

#%%
import evaluate

accuracy = evaluate.load("accuracy")
#%%
def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)
#%%
from transformers import create_optimizer
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

batch_size = 16
num_epochs = 5
batches_per_epoch = len(df_tokenized["train"])
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
#%%
id2label = {0: "Real", 1: "Fake"}
label2id = {"Real": 0, "Fake": 1}
#%%
from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)
#%%
#!pip install datasets
#%%
tf_train_set = model.prepare_tf_dataset(
    df_tokenized["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    df_tokenized["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)
#%%
import tensorflow as tf

model.compile(optimizer=optimizer)
#%%
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3)