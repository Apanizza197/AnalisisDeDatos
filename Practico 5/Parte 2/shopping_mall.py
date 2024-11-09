import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Cargar el archivo CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'clientes_shopping.csv')
df = pd.read_csv(csv_path)

# Normalización usando Min-Max Scaling para las columnas 'edad' y 'annual income'
scaler = MinMaxScaler()
df[['Age', 'Annual Income (k$)']] = scaler.fit_transform(df[['Age', 'Annual Income (k$)']])

# Crear gráficos de dispersión
plt.figure(figsize=(12, 8))

# Gráfico 1: Annual Income vs Spending Score
plt.subplot(3, 1, 1)
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')

# Gráfico 2: Age vs Spending Score
plt.subplot(3, 1, 2)
plt.scatter(df['Age'], df['Spending Score (1-100)'])
plt.title('Age vs Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')

# Gráfico 3: Age vs Annual Income
plt.subplot(3, 1, 3)
plt.scatter(df['Age'], df['Annual Income (k$)'])
plt.title('Age vs Annual Income')
plt.xlabel('Age')
plt.ylabel('Annual Income')

plt.tight_layout()
plt.show()
