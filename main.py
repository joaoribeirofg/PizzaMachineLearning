# 1. Importando as bibliotecas necessárias...
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Dados
x = np.array([15, 20, 25, 35, 45])      # diametro em cm das pizzas
y = np.array([7, 9, 12, 17.5, 18])      # preço em reais

# 3. Gráfico
plt.scatter(x, y)
plt.xlabel('Diâmetro em centímetros')
plt.ylabel('Preço em reais')
plt.title('Preço da pizza em relação ao seu diâmetro :')
plt.show()

# 3. Criando e treinando o modelo de regressão linear
model = LinearRegression()
x = x.reshape(-1, 1)                    #transforma o array x em matriz de uma coluna
model.fit(x, y)                         #treina o modelo com os dados

# 4. Desempenho do modelo
y_pred = model.predict(x)               # Faz as previsoes para os dados de treino
mse = mean_squared_error(y, y_pred)     # Calcula o erro médio
r2 = r2_score(y, y_pred)                # Calcula o coeficiente de determinação
print(f'MSE = {mse:.2f}')
print(f'R2 = {r2:.2f}')

# 5. Fazendo um teste com nova pizza:
x_novo = np.array([30])  # diametro nova pizza
y_novo = model.predict(x_novo.reshape(-1, 1))
print(f'A pizza de {x_novo[0]}cm custa {y_novo[0]:.2f}R$')
