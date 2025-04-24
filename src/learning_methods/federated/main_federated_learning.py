import json
from itertools import product

from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

from enums.Model import Model
from enums.Participant import NurseParticipant
from service.argumentservice.ArgumentService import ArgumentService
from learning_methods.federated.xgboost_.server import get_server_fn as get_server_fn_xgboost
from learning_methods.federated.xgboost_.client import get_client_fn as get_client_fn_xgboost
from learning_methods.federated.shallow_nn.server import get_server_fn as get_server_fn_shallow
from learning_methods.federated.shallow_nn.client import get_client_fn as get_client_fn_shallow
from service.exportservice.ExportService import ExportService

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

if __name__ == "__main__":
    arg_service = ArgumentService(model=True, resampling=True, scaling=True, database=True, features=True, dataset=True)
    models = arg_service.get_models()
    resampling_methods = arg_service.get_resampling_methods()
    scaling_methods = arg_service.get_scaling_methods()
    database = arg_service.get_database()
    features_list = arg_service.get_features()
    dataset = arg_service.get_dataset()

    export_service = ExportService(database=database, collection="federated")

    # Execute a machine learning pipeline for each configured model
    for model_enum, resampling_method, scaling_method, with_features in product(
        models, resampling_methods, scaling_methods, features_list
    ):
        match model_enum:
            case Model.XGBOOST:
                config = XGBOOST_CONFIG
            case Model.SHALLOW_NN:
                config = NN_CONFIG
            case Model.LOGISTIC_REGRESSION:
                config = XGBOOST_CONFIG
            case _:
                raise Exception(f"Could not initialize model {model_enum.value} for config")

        run_info = {
            "model": model_enum.value,
            "pre-processing": {
                "features": with_features,
                "resampling": {"method": resampling_method.value if resampling_method is not None else None},
                "scaling": {"method": scaling_method.value if scaling_method is not None else None},
            },
            "training_runs": [],
            "fl_config": config,
        }

        run_id = export_service.generate_unique_id([database, json.dumps(run_info)])
        run_info["_id"] = run_id
        mongo_id = export_service.export_run_to_mongodb(run_info=run_info)

        if export_service.run_is_finished(run_id):
            print(f"Run with id: {run_id} on database {database} already exists")
            continue

        print(f"Executing run with configuration: {run_info} on database {database}")

        for idx, participant_leave_out in enumerate(
            [NurseParticipant.n_F5, NurseParticipant.n_E4, NurseParticipant.n_DF]
        ):
            export_service.update_run(
                run_id=run_id,
                set_dict={"$set": {f"training_runs.{idx}.participant_leave_out": str(participant_leave_out)}},
            )

            match model_enum:
                case Model.XGBOOST:
                    server_app = ServerApp(
                        server_fn=get_server_fn_xgboost(
                            cfg=config,
                            mongo_id=mongo_id,
                            run_index=idx,
                            participant_leave_out=participant_leave_out,
                            export_service=export_service,
                            scaling_method=scaling_method,
                        )
                    )
                    client_app = ClientApp(
                        client_fn=get_client_fn_xgboost(
                            cfg=config,
                            mongo_id=mongo_id,
                            run_index=idx,
                            scaling_method=scaling_method,
                            resampling_method=resampling_method,
                            database=database,
                            participant_leave_out=participant_leave_out,
                        )
                    )
                case Model.SHALLOW_NN:
                    server_app = ServerApp(
                        server_fn=get_server_fn_shallow(
                            cfg=config,
                            mongo_id=mongo_id,
                            run_index=idx,
                            participant_leave_out=participant_leave_out,
                            export_service=export_service,
                            scaling_method=scaling_method,
                        )
                    )
                    client_app = ClientApp(
                        client_fn=get_client_fn_shallow(
                            cfg=config,
                            mongo_id=mongo_id,
                            run_index=idx,
                            scaling_method=scaling_method,
                            resampling_method=resampling_method,
                            database=database,
                            participant_leave_out=participant_leave_out,
                        )
                    )
                case Model.LOGISTIC_REGRESSION:
                    server_app = ServerApp(
                        server_fn=get_server_fn_xgboost(cfg=config, mongo_id=mongo_id, export_service=export_service)
                    )
                    client_app = ClientApp(client_fn=get_client_fn_xgboost(cfg=config, mongo_id=mongo_id))
                case _:
                    raise Exception(f"Could not initialize model {model_enum.value} for config")

            run_simulation(server_app=server_app, client_app=client_app, num_supernodes=NUM_SUPERNODES)

            del server_app, client_app

        # set "finished" to true in database
        export_service.update_run(run_id, {"$set": {"finished": True}})
