import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Cargar el archivo CSV
# Reemplaza 'tu_archivo.csv' con la ruta a tu archivo CSV
df = pd.read_csv("Practico 5\clientes_shopping.csv")

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
