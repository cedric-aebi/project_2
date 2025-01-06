# FL Parameters
## XGBoost
```toml
[tool.flwr.app.components]
serverapp = "src.learning_methods.federated.xgboost_.server:app"
clientapp = "src.learning_methods.federated.xgboost_.client:app"

[tool.flwr.app.config]
# ServerApp
num-server-rounds = 3
fraction-fit = 0.1
fraction-evaluate = 0.1

# ClientApp
local-epochs = 1
params.objective = "binary:logistic"
params.eta = 0.1 # Learning rate
params.max-depth = 8
params.eval-metric = "auc"
params.nthread = 16
params.num-parallel-tree = 1
params.subsample = 1
params.tree-method = "hist"
```

## TensorFlow
```toml
[tool.flwr.app.components]
serverapp = "src.learning_methods.federated.shallow_nn.server:app"
clientapp = "src.learning_methods.federated.shallow_nn.client:app"

[tool.flwr.app.config]
num-server-rounds = 3
local-epochs = 1
batch-size = 32
learning-rate = 0.005
fraction-fit = 0.5
verbose = false
```