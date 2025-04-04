from argparse import ArgumentParser

from enums.Dataset import Dataset
from enums.Model import Model
from enums.ResamplingMethod import ResamplingMethod
from enums.ScalingMethod import ScalingMethod


class ArgumentService:
    def __init__(
        self,
        database: bool | None = None,
        collection: bool | None = None,
        model: bool | None = None,
        features: bool | None = None,
        resampling: bool | None = None,
        scaling: bool | None = None,
        dataset: bool | None = None,
    ):
        parser = ArgumentParser()
        if database:
            parser.add_argument("-d", "--database", required=True)
        if collection:
            parser.add_argument("-c", "--collection", required=True, help="One of: centralized, individual, federated")
        if model:
            parser.add_argument(
                "-m",
                "--model",
                nargs="+",
                required=True,
                help="Space-separated list of models: shallow-nn, lr, xgboost",
            )
        if features:
            parser.add_argument("-f", "--features", nargs="+", required=True, help="Space-separated list of: yes or no")
        if resampling:
            parser.add_argument(
                "-r",
                "--resampling",
                nargs="+",
                required=True,
                help="Space-separated list of resampling methods: smote, tl, oversampling, none",
            )
        if scaling:
            parser.add_argument(
                "-s",
                "--scaling",
                nargs="+",
                required=True,
                help="Space-separated list of scaling methods: standard, minmax, none",
            )
        if dataset:
            parser.add_argument(
                "-ds",
                "--dataset",
                required=True,
                help="One of: nurse or stress",
            )
        self.args = parser.parse_args()

    def get_database(self) -> str:
        return self.args.database

    def get_collection(self) -> str:
        match self.args.collection:
            case "centralized":
                return self.args.collection
            case "individual":
                return self.args.collection
            case "federated":
                return self.args.collection
            case _:
                raise ValueError("Collection not recognized")

    def get_features(self) -> list[bool]:
        features: list[bool] = []
        for feature in self.args.features:
            match feature:
                case "yes":
                    features.append(True)
                case "no":
                    features.append(False)
                case _:
                    raise ValueError("Feature param not recognized")
        return features

    def get_models(self) -> list[Model]:
        models: list[Model] = []
        for model in self.args.model:
            match model:
                case "shallow-nn":
                    models.append(Model.SHALLOW_NN)
                case "lr":
                    models.append(Model.LOGISTIC_REGRESSION)
                case "xgboost":
                    models.append(Model.XGBOOST)
                case _:
                    raise ValueError("Model not recognized")
        return models

    def get_resampling_methods(self) -> list[ResamplingMethod | None]:
        resampling_methods: list[ResamplingMethod | None] = []
        for method in self.args.resampling:
            match method:
                case "smote":
                    resampling_methods.append(ResamplingMethod.SMOTE)
                case "tl":
                    resampling_methods.append(ResamplingMethod.TL)
                case "oversampling":
                    resampling_methods.append(ResamplingMethod.OVERSAMPLING)
                case "none":
                    resampling_methods.append(None)
                case _:
                    raise ValueError("Resampling method not recognized")
        return resampling_methods

    def get_scaling_methods(self) -> list[ScalingMethod | None]:
        scaling_methods: list[ScalingMethod | None] = []
        for method in self.args.scaling:
            match method:
                case "standard":
                    scaling_methods.append(ScalingMethod.STANDARDSCALER)
                case "minmax":
                    scaling_methods.append(ScalingMethod.MINMAXSCALER)
                case "none":
                    scaling_methods.append(None)
                case _:
                    raise ValueError("Scaling method not recognized")
        return scaling_methods

    def get_dataset(self) -> Dataset:
        match self.args.dataset:
            case "nurse":
                return Dataset.NURSE
            case "stress":
                return Dataset.STRESS
            case _:
                raise ValueError("Dataset not recognized")
