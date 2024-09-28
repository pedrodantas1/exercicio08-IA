import pandas as pd

df = pd.read_csv("dados_treinamento.csv")
df = df.sort_values("Popularidade", ascending=False)

print(df)
