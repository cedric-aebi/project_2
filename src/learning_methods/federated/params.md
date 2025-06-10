# Configs for Federated Learning

## Hint
Learning rates are 0.01 without windows, and 0.001 with windows.
## Nurse
### XGBoost
```python
NUM_SUPERNODES = 12
NUM_SERVER_ROUNDS = 100

XGBOOST_CONFIG = {
    "num_server_rounds": NUM_SERVER_ROUNDS,
    "fraction_fit": 1,
    "fraction_evaluate": 1,
    "local_epochs": 1,
    "params": {
        "objective": "binary:logistic",
        "eta": 0.01,
        "max_depth": 12,
        "num_parallel_tree": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "tree_method": "hist",
    },
}
```

### Shallow NN
```python
NUM_SUPERNODES = 12
NUM_SERVER_ROUNDS = 100

NN_CONFIG = {
    "num_server_rounds": NUM_SERVER_ROUNDS,
    "fraction_fit": 1,
    "fraction_evaluate": 1,
    "local_epochs": 1,
    "min_available_clients": NUM_SUPERNODES,
    "params": {
        "batch_size": 128,
        "learning_rate": 0.01,
        "verbose": False,
        "optimizer": "sgd",
        "regularization": False,
        "batch_normalization": False,
        "dropout": False,
    },
}
```

### Logistic Regression
```python
NUM_SUPERNODES = 12
NUM_SERVER_ROUNDS = 100

LG_CONFIG = {
    "num_server_rounds": NUM_SERVER_ROUNDS,
    "fraction_fit": 1,
    "fraction_evaluate": 1,
    "min_available_clients": NUM_SUPERNODES,
    "params": {
        "max_iter": 1,
        "penalty": "l2",
    },
}
```

## Stress
### XGBoost
```python
NUM_SUPERNODES = 32
NUM_SERVER_ROUNDS = 100

XGBOOST_CONFIG = {
    "num_server_rounds": NUM_SERVER_ROUNDS,
    "fraction_fit": 1,
    "fraction_evaluate": 1,
    "local_epochs": 1,
    "params": {
        "objective": "binary:logistic",
        "eta": 0.01,
        "max_depth": 12,
        "num_parallel_tree": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "tree_method": "hist",
    },
}
```

### Shallow NN
```python
NUM_SUPERNODES = 32
NUM_SERVER_ROUNDS = 100

NN_CONFIG = {
    "num_server_rounds": NUM_SERVER_ROUNDS,
    "fraction_fit": 1,
    "fraction_evaluate": 1,
    "local_epochs": 1,
    "min_available_clients": NUM_SUPERNODES,
    "params": {
        "batch_size": 128,
        "learning_rate": 0.01,
        "verbose": False,
        "optimizer": "sgd",
        "regularization": False,
        "batch_normalization": False,
        "dropout": False,
    },
}
```

### Logistic Regression
```python
NUM_SUPERNODES = 32
NUM_SERVER_ROUNDS = 100

LG_CONFIG = {
    "num_server_rounds": NUM_SERVER_ROUNDS,
    "fraction_fit": 1,
    "fraction_evaluate": 1,
    "min_available_clients": NUM_SUPERNODES,
    "params": {
        "max_iter": 1,
        "penalty": "l2",
    },
}
```