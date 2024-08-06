import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.metrics import (
    auc,
    roc_curve,
)
from config import ORDER, STRAINS
from config import ATCC_GROUPINGS, antibiotics, ab_order

def plot_ROC_curve(save_name, y_true, y_test, y_pred_prob, fold_index):
    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = [STRAINS[i] for i in ORDER]

    for i in range(np.unique(y_true).shape[0]):

        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )
        
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"Bacteria_ROC_curve_{save_name}_{fold_index}.png")
    plt.close()

def plot_bile_acids_ROC_curve(save_name, y_true, y_test, y_pred_prob, fold_index):
    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = ["Blank", "CA", "DCA", "GCDCA", "LCA", "TCDCA"]

    for i in range(np.unique(y_true).shape[0]):

        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )
        
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"Bile_acids_ROC_curve_{save_name}_{fold_index}.png")
    plt.close()

def plot_heatmap(save_name, cm, fold_index):
    plt.figure(figsize=(15, 12))
    label = [STRAINS[i] for i in ORDER]
    ax = sns.heatmap(
        cm, annot=True, cmap="Greys", fmt="0.0f", xticklabels=label, yticklabels=label
    )
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"Bacteria_heatmap_{save_name}_{fold_index}.png")
    plt.close()

def plot_bile_acids_heatmap(save_name, cm, fold_index):
    plt.figure(figsize=(8, 6))
    label = ["Blank", "CA", "DCA", "GCDCA", "LCA", "TCDCA"]
    ax = sns.heatmap(
        cm, annot=True, cmap="Greys", fmt="0.0f", xticklabels=label, yticklabels=label
    )
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"Bile_acids_heatmap_{save_name}_{fold_index}.png")
    plt.close()

def plot_melanoma_heatmap(save_name, cm, fold_index):
    plt.figure(figsize=(8, 6))
    label = ['A', 'A-S', 'DMEM', 'DMEM-S', 'G', 'G-S', 'HF', 'HF-S', 'MEL', 'MEL-S', 'ZAM', 'ZAM-S']
    ax = sns.heatmap(
        cm, annot=True, cmap="Greys", fmt="0.0f", xticklabels=label, yticklabels=label
    )
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"Melanoma_heatmap_{save_name}_{fold_index}.png")
    plt.close()

def plot_melanoma_ROC_curve(save_name, y_true, y_test, y_pred_prob, fold_index):
    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = ['A', 'A-S', 'DMEM', 'DMEM-S', 'G', 'G-S', 'HF', 'HF-S', 'MEL', 'MEL-S', 'ZAM', 'ZAM-S']

    for i in range(np.unique(y_true).shape[0]):

        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )
        
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"Melanoma_ROC_curve_{save_name}_{fold_index}.png")
    plt.close()

def plot_antibiotic_groupings(save_name, y_true, y_pred, fold_index):

    # Mapping predictions into antibiotic groupings
    y_ab = np.asarray([ATCC_GROUPINGS[i] for i in y_true])
    y_ab_hat = np.asarray([ATCC_GROUPINGS[i] for i in y_pred])

    # Computing accuracy
    acc = (y_ab_hat == y_ab).mean()
    print('Accuracy: {:0.1f}%'.format(100*acc))

    sns.set_context("talk", rc={"font":"Helvetica", "font.size":12})
    label = [antibiotics[i] for i in ab_order]
    cm = confusion_matrix(y_ab, y_ab_hat, labels=ab_order)
    plt.figure(figsize=(5, 4))
    cm = 100 * cm / cm.sum(axis=1)[:,np.newaxis]
    ax = sns.heatmap(cm, annot=True, cmap='Greys', fmt='0.0f',
                    xticklabels=label, yticklabels=label)
    ax.xaxis.tick_top()
    plt.xticks(rotation=90) 
    plt.tight_layout()
    plt.savefig(f"Bacteria_antibiotic_groupings_{save_name}_{fold_index}.png")
    plt.close()

    return acc

def plot_MRSA_MSSA_heatmap(save_name, y_true, y_pred, index):

    sns.set_context("talk", rc={"font":"Helvetica", "font.size":12})
    label = ['MRSA','MSSA']
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    cm = 100 * cm / cm.sum(axis=1)[:,np.newaxis]
    print(cm)
    ax = sns.heatmap(cm, annot=True, cmap='Greys', fmt='0.0f',
                    xticklabels=label, yticklabels=label)
    ax.xaxis.tick_top()
    plt.xticks(rotation=90) 
    plt.tight_layout()
    plt.savefig(f"MRSA_MSSA_{save_name}_{index}.png")
    plt.close()

def plot_MRSA_MSSA__ROC_curve(save_name, y_true, y_test, y_pred_prob, index):

    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = ['MRSA','MSSA']

    for i in range(np.unique(y_true).shape[0]):

        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )
        
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"MRSA_MSSA_ROC_curve_{save_name}_{index}.png")
    plt.close()

def plot_clinical_heatmap(save_name, y_true, y_pred, index):

    sns.set_context("talk", rc={"font":"Helvetica", "font.size":12})
    label = [antibiotics[i] for i in ab_order]
    y_label=['S.aureus', 'E.faecalis', 'E.faecium', 'E.coli', 'P.aeruginosa']
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6])
    plt.figure(figsize=(5, 4))
    cm = 100 * cm / cm.sum(axis=1)[:,np.newaxis]
    cm = np.delete(cm, [1, 4], axis=0)
    ax = sns.heatmap(cm, annot=True, cmap='Greys', fmt='0.0f',
                    xticklabels=label,
                    yticklabels=y_label)
    
    ax.xaxis.tick_top()
    plt.xticks(rotation=90) 
    plt.tight_layout()
    plt.savefig(f"clinical_{save_name}_{index}.png")
    plt.close()


def plot_clinical_ROC_curve(save_name, y_true, y_test, y_pred_prob, index):

    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = [STRAINS[i] for i in ORDER]

    for i in range(np.unique(y_true).shape[0]):

        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )
        
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"clinical_ROC_curve_{save_name}_{index}.png")
    plt.close()

def plot_loss_metrics(training_results, fold_index, fold_name):
    picture_name = f"Bile_acids_{fold_name}_loss_{fold_index}.png"

    plt.figure(figsize=(8, 6))
    plt.plot(
        training_results["training_loss"], color="darkred", label="Train Loss", linewidth=1,
    )
    plt.plot(
        training_results["validation_loss"], color="olive", label="Validation Loss", linewidth=1,
    )
    plt.grid(True)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7
    ) 
    plt.savefig(picture_name, bbox_inches="tight", pad_inches=0.05) 
    plt.close()


def plot_metrics(training_results, fold_index, fold_name):
    picture_name = f"Bile_acids_{fold_name}_accuracy_{fold_index}.png"

    plt.figure(figsize=(8, 6))
    plt.plot(
        training_results["train_metrics"],
        color="darkred",
        label=f"Train accuracy",
        linewidth=1,
    )
    plt.plot(
        training_results["validation_metrics"],
        color="olive",
        label=f"Valid accuracy",
        linewidth=1,
    )
    plt.grid(True)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7
    ) 
    plt.savefig(picture_name, bbox_inches="tight", pad_inches=0.05) 
    plt.close()
