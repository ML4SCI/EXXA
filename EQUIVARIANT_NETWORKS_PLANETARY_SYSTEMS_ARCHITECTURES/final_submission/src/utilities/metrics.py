import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
)
import os
import torch


def plot_confusion_matrix_and_roc(model, test_loader, channels):
    print("Started plotting confusion matrix and ROC curve...")
    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            print(f"labels: {labels}")
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(
                probs[:, 1].cpu().numpy()
            )  # Get the probabilities for the positive class
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(
        f"Accuracy for the model trained on channels {channels} is {accuracy}, with all_preds={all_preds} and all_labels={all_labels}"
    )

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    conf_matrix_path = os.path.join(
        save_dir, f"conf_matrix_channels_{'_'.join(map(str, channels))}.png"
    )
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Confusion matrix saved to {conf_matrix_path}")

    print("Roc curve!!!")
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

    roc_curve_path = os.path.join(
        save_dir, f"roc_curve_channels_{'_'.join(map(str, channels))}.png"
    )
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC curve saved to {roc_curve_path}")

    return accuracy
