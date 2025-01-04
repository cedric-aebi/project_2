from argparse import ArgumentParser

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
    ):
        parser = ArgumentParser()
        if database:
            parser.add_argument("-d", "--database", required=True)
        if collection:
            parser.add_argument("-c", "--collection", required=True, help="One of: centralized, individual, federated")
        if model:
            parser.add_argument("-m", "--model", required=True, help="One of: shallow-nn, lr, xgboost")
        if features:
            parser.add_argument("-f", "--features", required=True, help="yes or no")
        if resampling:
            parser.add_argument(
                "-r",
                "--resampling",
                nargs="+",
                required=True,
                help="Space-separated list of resampling methods: smote, smoteenn, tl, oversampling, none",
            )
        if scaling:
            parser.add_argument(
                "-s",
                "--scaling",
                nargs="+",
                required=True,
                help="Space-separated list of scaling methods: standard, minmax, none",
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

    def get_features(self) -> bool:
        match self.args.features:
            case "yes":
                return True
            case "no":
                return False
            case _:
                raise ValueError("Feature param not recognized")

    def get_model(self) -> Model:
        match self.args.model:
            case "shallow-nn":
                return Model.SHALLOW_NN
            case "lr":
                return Model.LOGISTIC_REGRESSION
            case "xgboost":
                return Model.XGBOOST
            case _:
                raise ValueError("Model not recognized")

    def get_resampling_methods(self) -> list[ResamplingMethod | None]:
        resampling_methods: list[ResamplingMethod | None] = []
        for method in self.args.resampling:
            match method:
                case "smote":
                    resampling_methods.append(ResamplingMethod.SMOTE)
                    break
                case "smoteenn":
                    resampling_methods.append(ResamplingMethod.SMOTEENN)
                    break
                case "tl":
                    resampling_methods.append(ResamplingMethod.TL)
                    break
                case "oversampling":
                    resampling_methods.append(ResamplingMethod.OVERSAMPLING)
                    break
                case "none":
                    resampling_methods.append(None)
                    break
                case _:
                    raise ValueError("Resampling method not recognized")
        return resampling_methods

    def get_scaling_methods(self) -> list[ScalingMethod | None]:
        scaling_methods: list[ScalingMethod | None] = []
        for method in self.args.scaling:
            match method:
                case "standard":
                    scaling_methods.append(ScalingMethod.STANDARDSCALER)
                    break
                case "minmax":
                    scaling_methods.append(ScalingMethod.MINMAXSCALER)
                    break
                case "none":
                    scaling_methods.append(None)
                    break
                case _:
                    raise ValueError("Scaling method not recognized")
        return scaling_methods
