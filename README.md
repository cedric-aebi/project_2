# MTh
## Run experiments
To run the different experiments, you can run the `main_` files in the `src/learning_methods` folder. For example, to run the MTh experiment, you can run:

```bash
python src/learning_methods/centralized/main_centralized_learning.py
python src/learning_methods/individual/main_individual_learning.py
python src/learning_methods/federated/main_federated_learning.py
```

## Notebooks
The jupyter notebooks in the `src/notebooks` folder are used for some visualization and prototyping and are not part
of the main experiments.

## Main Experimental pipeline
1. Run `src/main_feature_extraction_nurse.py` and `src/main_feature_extraction_stress.py`
2. Run `src/main_split_data.py`
3. Run one of the main files in the `src/learning_methods` folder, e.g. `src/learning_methods/centralized/main_centralized_learning.py`
4. Run `src/main_average_results.py`

## Parametrization of the main files
The main files in the `src/learning_methods` folder can be parametrized with some arguments. The file then
executes every combination of parameters given. For example 
```bash
python src/learning_methods/centralized/main_centralized_learning.py -m xgboost shallow_nn -r standard none
```

Runs
1. XGBoost + StandardScaler
2. XGBoost + No scaler
3. Shallow Neural Network + StandardScaler
4. Shallow Neural Network + No scaler