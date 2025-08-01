import torch

def f1_score_db_tuning(logits, targets, average="micro", type="single"):
    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")


    probs = torch.sigmoid(logits)
    dbs = torch.linspace(0, 1, 100)

    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    precision_micro = torch.zeros(len(dbs))
    precision_macro = torch.zeros(len(dbs))
    recall_micro = torch.zeros(len(dbs))
    recall_macro = torch.zeros(len(dbs))

    for idx, db in enumerate(dbs):
        predictions = (probs > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)


        precision_micro[idx] = tp[idx].sum() / (tp[idx].sum() + fp[idx].sum() + 1e-10)
        recall_micro[idx] = tp[idx].sum() / (tp[idx].sum() + fn[idx].sum() + 1e-10)


        precision_macro[idx] = torch.mean(tp[idx] / (tp[idx] + fp[idx] + 1e-10))
        recall_macro[idx] = torch.mean(tp[idx] / (tp[idx] + fn[idx] + 1e-10))


    if average == "micro":
        f1_micro = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
        f1_macro = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)
    else:
        f1_micro = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
        f1_macro = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)

    if type == "single":
        best_f1_micro = f1_micro.max()
        best_f1_macro = f1_macro.max()
        best_db = dbs[f1_micro.argmax()]
        print(f"Best F1 micro: {best_f1_micro} at DB: {best_db}")
        print(f"Best F1 macro: {best_f1_macro} at DB: {best_db}")
        return best_f1_micro, best_f1_macro, best_db

    if type == "per_class":
        best_f1_micro = f1_micro.max(1)
        best_f1_macro = f1_macro.max(1)
        best_db = dbs[f1_micro.argmax(0)]
        print(f"Best F1 per class (micro): {best_f1_micro} at DB: {best_db}")
        print(f"Best F1 per class (macro): {best_f1_macro} at DB: {best_db}")
        return best_f1_micro, best_f1_macro, best_db
