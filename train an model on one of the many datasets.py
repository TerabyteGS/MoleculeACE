from MoleculeACE import MPNN, Data, Descriptors, calc_rmse, calc_cliff_rmse, get_benchmark_config

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