import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# Passo 1: Pre-processamento de dados
data = pd.read_csv("dados_treinamento.csv")

print("--------------------------------------------------------")
print("Dados das músicas da Garage Band (Gerados randomicamente)")
print("--------------------------------------------------------")
print(data.head())

# Separar as variáveis de entrada (X) e de saída (y)
X = data.drop("Popularidade", axis=1)
y = data["Popularidade"]

# Dividir o conjunto de dados em conjunto de treinamento e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalizar os atributos de entrada
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train[0:3, :])

# Passo 2: Criação da rede neural MLP
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=5000,
)

# Passo 3: Treinamento da rede neural
mlp.fit(X_train, y_train)

# Passo 4: Avaliação do modelo
y_pred = mlp.predict(X_test)

# Métricas de desempenho
print("Erro quadratico medio  : " + str(round(mean_squared_error(y_test, y_pred), 3)))
print("R2-Score               : " + str(round(r2_score(y_test, y_pred), 3)))

# Passo 5: Previsões de novas músicas
# Exemplos de novas musicas que serão lançadas
novas_musicas = pd.DataFrame(
    {
        "Duracao_Musica": [4.5, 3.2],
        "Energia": [0.85, 0.45],
        "Valencia": [0.75, 0.53],
        "Instrumentalidade": [0.25, 0.7],
        "Acustica": [0.4, 0.9],
        "Habilidade_Vocal_Media": [8.5, 7.2],
        "Complexidade_Composicao": [0.65, 0.36],
    }
)

# Normalizar os novos dados de musicas
novas_musicas = scaler.transform(novas_musicas)

# Previsoes de popularidade para as novas musicas
previsoes_novas_musicas = mlp.predict(novas_musicas)
print(f"Previsões de popularidade: {previsoes_novas_musicas}")
