import subprocess
import os
from itertools import product

from service.argumentservice.ArgumentService import ArgumentService

if __name__ == "__main__":
    arg_service = ArgumentService(model=True, resampling=True, scaling=True, database=True, features=True, dataset=True)
    models = arg_service.get_models()
    resampling_methods = arg_service.get_resampling_methods()
    scaling_methods = arg_service.get_scaling_methods()
    database = arg_service.get_database()
    features_list = arg_service.get_features()
    dataset = arg_service.get_dataset()

    # Execute machine learning pipeline for each configured model
    for model_enum, resampling_method, scaling_method, with_features in product(
            models, resampling_methods, scaling_methods, features_list
    ):


    # List of strings to iterate over
    string_list = ["value1"]

    for s in string_list:
        # Set environment variables
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "../../../"}
        path = "./xgboost_"

        process = subprocess.Popen(
            ["flwr", "run", "."],
            env=env,
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output line by line
        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            print(f"\nCommand failed with exit code {process.returncode}")
