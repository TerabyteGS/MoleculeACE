from MoleculeACE import MPNN, Data, Descriptors, calc_rmse, calc_cliff_rmse, get_benchmark_config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuração inicial
dataset = 'CHEMBL4203_Ki'
descriptor = Descriptors.GRAPH
algorithm = MPNN

# Load data
data = Data(dataset)

# Get the already optimized hyperparameters
hyperparameters = get_benchmark_config(dataset, algorithm, descriptor)

# Featurize SMILES strings with a specific method
data(descriptor)

# Train and a model
model = algorithm(**hyperparameters)
model.train(data.x_train, data.y_train)
y_hat = model.predict(data.x_test)

# Evaluate your model on activity cliff compounds
rmse = calc_rmse(data.y_test, y_hat)
rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=data.y_test, cliff_mols_test=data.cliff_mols_test)

print(f"rmse: {rmse}")
print(f"rmse_cliff: {rmse_cliff}")

# Criar dataframe para visualização apenas com os dados de teste
df = pd.DataFrame({
    'y_test': data.y_test,
    'y_hat': y_hat,
    'cliff_mol': data.cliff_mols_test
})

# 1. Distribuição dos Valores de Bioatividade (y)
plt.figure(figsize=(8, 6))
sns.histplot(data.y_test, kde=True, bins=20, color='blue')
plt.title('Distribuição dos Valores de Bioatividade (y)')
plt.xlabel('y (log Ki)')
plt.ylabel('Frequência')
plt.savefig('bioatividade_distribuicao.png')
plt.close()

# 2. Comparação entre RMSE Geral e RMSE Cliff
plt.figure(figsize=(6, 6))
plt.bar(['RMSE Geral', 'RMSE Cliff'], [rmse, rmse_cliff], color=['green', 'orange'])
plt.title('Comparação entre RMSE Geral e RMSE Cliff')
plt.ylabel('Erro')
plt.savefig('comparacao_rmse.png')
plt.close()

# 3. Relação entre Afinidade Experimental e Predições do Modelo
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['y_test'], y=df['y_hat'], hue=df['cliff_mol'], palette=['blue', 'red'])
plt.plot([min(df['y_test']), max(df['y_test'])], [min(df['y_test']), max(df['y_test'])], color='gray', linestyle='--')
plt.title('Afinidade Experimental vs. Predições do Modelo')
plt.xlabel('y_test (Experimental)')
plt.ylabel('y_hat (Predito)')
plt.legend(title='Activity Cliff', labels=['Não', 'Sim'])
plt.savefig('relacao_experimental_predicao.png')
plt.close()

# 4. Distribuição dos Valores Experimentais (exp_mean [nM]) por Tipo de Divisão
# Ajustando para apenas valores de teste, já que treino não está disponível no DataFrame
plt.figure(figsize=(8, 6))
sns.boxplot(x='cliff_mol', y='y_test', data=df, palette='Set2')
plt.title('Distribuição dos Valores Experimentais por Cliff Mol')
plt.xlabel('Cliff Mol (0 = Não, 1 = Sim)')
plt.ylabel('y (log Ki)')
plt.savefig('distribuicao_por_divisao.png')
plt.close()

# 5. Distribuição de Molecules com "Activity Cliffs"
plt.figure(figsize=(6, 6))
cliff_counts = df['cliff_mol'].value_counts()
plt.pie(cliff_counts, labels=['Não', 'Sim'], autopct='%1.1f%%', colors=['lightblue', 'salmon'])
plt.title('Distribuição de Moléculas com Activity Cliffs')
plt.savefig('distribuicao_activity_cliffs.png')
plt.close()
