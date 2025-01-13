import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from MoleculeACE import MPNN, Data, Descriptors, calc_rmse, calc_cliff_rmse, get_benchmark_config

# ============================
# CONFIGURAÇÃO DO MODELO
# ============================
dataset = 'CHEMBL4203_Ki'
descriptor = Descriptors.GRAPH
algorithm = MPNN

# Load data
data = Data(dataset)

# Get hyperparameters
hyperparameters = get_benchmark_config(dataset, algorithm, descriptor)

# Featurize and train
data(descriptor)
model = algorithm(**hyperparameters)
model.train(data.x_train, data.y_train)
y_hat = model.predict(data.x_test)

# Métricas principais
rmse = calc_rmse(data.y_test, y_hat)
rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=data.y_test, cliff_mols_test=data.cliff_mols_test)

print(f"rmse: {rmse}")
print(f"rmse_cliff: {rmse_cliff}")

# ============================
# ORGANIZAR OS DADOS PARA PLOTAGEM
# ============================
df = pd.DataFrame({
    'y_test': data.y_test,
    'y_hat': y_hat,
    'cliff_mol': data.cliff_mols_test
})

df['cliff_mol'] = df['cliff_mol'].astype(str)  # Converter para string
df['residuals'] = df['y_test'] - df['y_hat']

# ============================
# FUNÇÃO PARA SALVAR OS GRÁFICOS
# ============================
def salvar_grafico(nome):
    plt.tight_layout()
    plt.savefig(f"{nome}.png", dpi=300, bbox_inches="tight")
    plt.close()

# ============================
# 1. Boxplot dos Resíduos por Activity Cliff
# ============================
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='cliff_mol', y='residuals', palette={'0': 'blue', '1': 'red'})
plt.title("Boxplot dos Resíduos - Activity Cliffs")
plt.xlabel("Activity Cliff")
plt.ylabel("Resíduos")
salvar_grafico("boxplot_residuos_activity_cliff")

# ============================
# 2. Histograma dos Resíduos
# ============================
plt.figure(figsize=(8, 6))
sns.histplot(df['residuals'], kde=True, bins=20, color='purple')
plt.title("Distribuição dos Resíduos")
plt.xlabel("Resíduos")
plt.ylabel("Frequência")
salvar_grafico("histograma_residuos")

# ============================
# 3. Scatterplot com Resíduos
# ============================
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='y_test', y='residuals', hue='cliff_mol', palette={'0': 'blue', '1': 'red'})
plt.axhline(0, linestyle='--', color='gray')
plt.title("Resíduos vs. Valores Experimentais")
plt.xlabel("y_test (Experimental)")
plt.ylabel("Resíduos")
salvar_grafico("scatter_residuos")

# ============================
# 4. RMSE Visualizado
# ============================
plt.figure(figsize=(8, 6))
sns.barplot(x=["RMSE Geral", "RMSE Activity Cliff"], y=[rmse, rmse_cliff], palette='coolwarm')
plt.title("Comparação de RMSE")
plt.ylabel("RMSE")
salvar_grafico("rmse_comparacao")

# ============================
# 5. Histograma de y_test e y_hat
# ============================
plt.figure(figsize=(8, 6))
sns.histplot(df['y_test'], label='Experimental', kde=True, color='blue')
sns.histplot(df['y_hat'], label='Predito', kde=True, color='orange')
plt.title("Distribuição de y_test e y_hat")
plt.xlabel("Valores")
plt.ylabel("Frequência")
plt.legend()
salvar_grafico("histograma_ytest_yhat")

# ============================
# 6. Correlação entre y_test e y_hat
# ============================
correlation = df['y_test'].corr(df['y_hat'])
plt.figure(figsize=(8, 6))
sns.heatmap(df[['y_test', 'y_hat']].corr(), annot=True, cmap='coolwarm', center=0)
plt.title(f"Correlação entre y_test e y_hat\n(r = {correlation:.3f})")
salvar_grafico("correlacao_ytest_yhat")

# ============================
# 7. Gráfico de Erro Absoluto
# ============================
df['absolute_error'] = abs(df['residuals'])
plt.figure(figsize=(8, 6))
sns.scatterplot(x='y_test', y='absolute_error', hue='cliff_mol', data=df, palette={'0': 'blue', '1': 'red'})
plt.title("Erro Absoluto vs. Valores Experimentais")
plt.xlabel("y_test (Experimental)")
plt.ylabel("Erro Absoluto")
salvar_grafico("erro_absoluto")

# ============================
# 8. Gráfico QQ dos Resíduos
# ============================
import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(df['residuals'], dist="norm", plot=plt)
plt.title("Gráfico QQ dos Resíduos")
salvar_grafico("qqplot_residuos")

# ============================
# 9. Boxplot de y_hat por Activity Cliff
# ============================
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='cliff_mol', y='y_hat', palette={'0': 'blue', '1': 'red'})
plt.title("Distribuição de y_hat - Activity Cliffs")
plt.xlabel("Activity Cliff")
plt.ylabel("y_hat")
salvar_grafico("boxplot_yhat")

print("Todos os gráficos foram salvos com sucesso!")
