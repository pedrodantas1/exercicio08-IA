import numpy as np
import pandas as pd

# Definir número de linhas
n_rows = 192

# Definir valores aleatórios para cada atributo com base nas descrições fornecidas
np.random.seed(42)  # Para reprodutibilidade

data = {
    "Duracao_Musica": np.round(
        np.random.uniform(2.0, 7.0, n_rows), 2
    ),  # Duração entre 2 e 7 minutos
    "Energia": np.round(np.random.uniform(0, 1, n_rows), 2),  # Energia entre 0 e 1
    "Valencia": np.round(np.random.uniform(0, 1, n_rows), 2),  # Valencia entre 0 e 1
    "Instrumentalidade": np.round(
        np.random.uniform(0, 1, n_rows), 2
    ),  # Instrumentalidade entre 0 e 1
    "Acustica": np.round(np.random.uniform(0, 1, n_rows), 2),  # Acustica entre 0 e 1
    "Habilidade_Vocal_Media": np.round(
        np.random.uniform(0, 10, n_rows), 2
    ),  # Habilidade vocal entre 0 e 10
    "Complexidade_Composicao": np.round(
        np.random.uniform(0, 1, n_rows), 2
    ),  # Complexidade entre 0 e 1
    "Popularidade": np.round(
        np.random.uniform(0, 100, n_rows), 2
    ),  # Popularidade entre 0 e 100
}

# Criar DataFrame
df = pd.DataFrame(data)

# Salvar como CSV
df.to_csv("dados_treinamento.csv", index=False)
