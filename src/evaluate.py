
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize

def evaluate(y_true, y_pred, y_proba=None, classes=('N','S','V','F','Q'),
             name='model', plot=True):
    labels = list(range(len(classes)))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    rows = {c: dict(precision=p[i], recall=r[i], f1=f1[i], support=int(sup[i]))
            for i, c in enumerate(classes)}

    if y_proba is not None:
        Y = label_binarize(y_true, classes=labels)
        for i, c in enumerate(classes):
            ok = 0 < Y[:, i].sum() < len(Y)
            rows[c]['auc'] = roc_auc_score(Y[:, i], y_proba[:, i]) if ok else np.nan

    df = pd.DataFrame(rows).T
    summary = {'model': name,
               'accuracy': accuracy_score(y_true, y_pred),
               'macro_precision': p.mean(), 'macro_recall': r.mean(),
               'macro_f1': f1.mean()}
    if y_proba is not None:
        summary['macro_auc'] = df['auc'].mean()

    print(f"=== {name} ===")
    print(df.round(4)); print({k: (round(v, 4) if not isinstance(v, str) else v)
                               for k, v in summary.items()})

    if plot:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(1, 2 if y_proba is not None else 1,
                               figsize=(12 if y_proba is not None else 5, 4.5))
        ax = np.atleast_1d(ax)
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax[0], colorbar=False, values_format='d')
        ax[0].set_title(f'Confusion matrix - {name}')
        if y_proba is not None:
            for i, c in enumerate(classes):
                if 0 < Y[:, i].sum() < len(Y):
                    fpr, tpr, _ = roc_curve(Y[:, i], y_proba[:, i])
                    ax[1].plot(fpr, tpr, label=c)
            ax[1].plot([0, 1], [0, 1], 'k--', lw=0.8)
            ax[1].set_xlabel('FPR'); ax[1].set_ylabel('TPR')
            ax[1].set_title('ROC curves'); ax[1].legend()
        plt.tight_layout(); plt.show()
    return summary, df
