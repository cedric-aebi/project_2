from pathlib import Path
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
from enums.Model import Model

BASE_PATH = Path(__file__).parent.parent / "results" / "csv"
COLLECTION = "federated"
MODEL = Model.LOGISTIC_REGRESSION

if __name__ == "__main__":
    collection: Collection = MongoClient().project_2_no_windows[COLLECTION]

    match COLLECTION:
        case "centralized":
            best = collection.find({"model": MODEL}).sort("average_scoring.mean_f1", -1)[0]

            rows = []

            # Individual scores
            idx = 2
            for subject in best["individual_scoring"]:
                rows.append(
                    [
                        idx,
                        round(subject["testing_set"]["accuracy"], 4),
                        round(subject["testing_set"]["recall"], 4),
                        round(subject["testing_set"]["precision"], 4),
                        round(subject["testing_set"]["f1"], 4),
                    ]
                )
                idx += 1

            # Average scores
            rows.append(
                [
                    "Average",
                    round(best["average_scoring"]["mean_accuracy"], 4),
                    round(best["average_scoring"]["mean_recall"], 4),
                    round(best["average_scoring"]["mean_precision"], 4),
                    round(best["average_scoring"]["mean_f1"], 4),
                ]
            )

            # Centralized scores
            rows.append(
                [
                    "Centralized",
                    round(best["centralized_scoring"]["testing_set"]["accuracy"], 4),
                    round(best["centralized_scoring"]["testing_set"]["recall"], 4),
                    round(best["centralized_scoring"]["testing_set"]["precision"], 4),
                    round(best["centralized_scoring"]["testing_set"]["f1"], 4),
                ]
            )

            df = pd.DataFrame(data=rows, columns=["Subject", "Accuracy", "Recall", "Precision", "F1"])
            df.to_csv(BASE_PATH / COLLECTION / f"{MODEL}_{best['_id']}.csv", index=False)
        case "individual":
            best = collection.find({"model": MODEL}).sort("average_scoring.mean_f1", -1)[0]

            rows = []

            # Individual scores
            for subject in best["subjects"]:
                rows.append(
                    [
                        subject["subject"],
                        round(subject["scores"]["testing_set"]["accuracy"], 4),
                        round(subject["scores"]["testing_set"]["recall"], 4),
                        round(subject["scores"]["testing_set"]["precision"], 4),
                        round(subject["scores"]["testing_set"]["f1"], 4),
                    ]
                )

            # Average scores
            rows.append(
                [
                    "Average",
                    round(best["average_scoring"]["mean_accuracy"], 4),
                    round(best["average_scoring"]["mean_recall"], 4),
                    round(best["average_scoring"]["mean_precision"], 4),
                    round(best["average_scoring"]["mean_f1"], 4),
                ]
            )

            df = pd.DataFrame(data=rows, columns=["Subject", "Accuracy", "Recall", "Precision", "F1"])
            df.to_csv(BASE_PATH / COLLECTION / f"{MODEL}_{best['_id']}.csv", index=False)
        case "federated":
            rows = []

            # Individual scores
            for subject in range(2, 36):
                document = collection.find_one({"model": MODEL, "subject_nr": subject})
                rows.append(
                    [
                        document["subject_nr"],
                        round(document["rounds"][-1]["testing_set"]["accuracy"], 4),
                        round(document["rounds"][-1]["testing_set"]["recall"], 4),
                        round(document["rounds"][-1]["testing_set"]["precision"], 4),
                        round(document["rounds"][-1]["testing_set"]["f1"], 4),
                    ]
                )

            # Average scores
            average = collection.find_one({"model": MODEL, "subject_nr": "average"})
            rows.append(
                [
                    "Average",
                    round(average["average_scoring"]["mean_accuracy"], 4),
                    round(average["average_scoring"]["mean_recall"], 4),
                    round(average["average_scoring"]["mean_precision"], 4),
                    round(average["average_scoring"]["mean_f1"], 4),
                ]
            )

            # Centralized Scoring
            server = collection.find_one({"model": MODEL, "subject_nr": "server"})
            rows.append(
                [
                    "Centralized",
                    round(server["rounds"][-1]["testing_set"]["accuracy"], 4),
                    round(server["rounds"][-1]["testing_set"]["recall"], 4),
                    round(server["rounds"][-1]["testing_set"]["precision"], 4),
                    round(server["rounds"][-1]["testing_set"]["f1"], 4),
                ]
            )

            df = pd.DataFrame(data=rows, columns=["Subject", "Accuracy", "Recall", "Precision", "F1"])
            df.to_csv(BASE_PATH / COLLECTION / f"{MODEL}.csv", index=False)
