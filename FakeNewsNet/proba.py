import pandas as pd

df = pd.read_csv('politifact_ready_for_testing.csv')
textform = []
for words in df["content"]:
    lista = ' '.join(words)
    textform.append(lista)
