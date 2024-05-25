import statistics

from pymongo import MongoClient
from pymongo.collection import Collection

if __name__ == "__main__":
    collection: Collection = MongoClient().project_2_no_windows.individual

    documents = collection.find({"model": "Logistic Regression"})

    comparison_dicts = []
    for document in documents:
        subsets = []
        for subject in document["subjects"]:
            subsets.append(
                {
                    "f1": subject["scores"]["testing_set"]["f1"],
                    "best_params": subject["best_params"],
                }
            )
        average_f1 = statistics.fmean([subset["f1"] for subset in subsets])
        comparison_dicts.append(
            {
                "pre-processing": document["pre-processing"],
                "best_params": document["best_params"],
                "average_f1": average_f1,
            }
        )
    print(sorted(comparison_dicts, key=lambda x: x["average_f1"]))
