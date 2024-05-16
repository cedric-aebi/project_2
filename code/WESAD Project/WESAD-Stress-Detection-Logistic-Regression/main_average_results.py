import statistics

import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection

if __name__ == '__main__':
    collection_individual: Collection = MongoClient().wesad.individual
    collection_centralized: Collection = MongoClient().wesad.centralized
    collection_federated: Collection = MongoClient().wesad.federated

    # Centralized
    document = collection_centralized.find_one()
    accs, precs, recs, f1s = [], [], [], []
    for subject in document["individual_scoring"]:
        accs.append(subject["acc"])
        precs.append(subject["prec"])
        recs.append(subject["rec"])
        f1s.append(subject["f1"])

    mean_acc = statistics.fmean(accs)
    mean_prec = statistics.fmean(precs)
    mean_rec = statistics.fmean(recs)
    mean_f1 = statistics.fmean(f1s)

    df = pd.DataFrame({"mean_acc": mean_acc, "mean_prec": mean_prec, "mean_rec": mean_rec, "mean_f1": mean_f1},
                      index=[0])
    df.to_csv("centralized_average_results.csv", index=False)

    # Individual
    document = collection_individual.find_one()
    accs, precs, recs, f1s = [], [], [], []
    for subject in document["individual_scoring"]:
        accs.append(subject["acc"])
        precs.append(subject["prec"])
        recs.append(subject["rec"])
        f1s.append(subject["f1"])

    mean_acc = statistics.fmean(accs)
    mean_prec = statistics.fmean(precs)
    mean_rec = statistics.fmean(recs)
    mean_f1 = statistics.fmean(f1s)

    df = pd.DataFrame({"mean_acc": mean_acc, "mean_prec": mean_prec, "mean_rec": mean_rec, "mean_f1": mean_f1},
                      index=[0])
    df.to_csv("individual_average_results.csv", index=False)

    # Federated
    documents = collection_federated.find()
    accs, precs, recs, f1s = [], [], [], []
    for document in documents:
        if document["_id"] != "centralized":
            accs.append(document["epochs"][-1]["acc"])
            precs.append(document["epochs"][-1]["prec"])
            recs.append(document["epochs"][-1]["rec"])
            f1s.append(document["epochs"][-1]["f1"])

    mean_acc = statistics.fmean(accs)
    mean_prec = statistics.fmean(precs)
    mean_rec = statistics.fmean(recs)
    mean_f1 = statistics.fmean(f1s)

    df = pd.DataFrame({"mean_acc": mean_acc, "mean_prec": mean_prec, "mean_rec": mean_rec, "mean_f1": mean_f1},
                      index=[0])
    df.to_csv("federated_average_results.csv", index=False)
