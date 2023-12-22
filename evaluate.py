import torch
import argparse
import numpy as np

from dataset.classifier_dataset import ClassifierDataSet
from models.classifier2_cnn import Classifier2CNN, Classifier2CNNEvaluater
from models.classifier2_ViT import Classifier2ViT, Classifier2ViTEvaluater
from models.classifier5_cnn import Classifier5CNN, Classifier5CNNEvaluater
from models.classifier5_ViT import Classifier5ViT, Classifier5ViTEvaluater
from models.classifier_n_cnn import ClassifierNCNN, ClassifierNCNNEvaluater

from utils.draw_confusion_matrix import draw_confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    roc_auc_score
)

def evaluate_2_cnn(model_path:str="output/Classifier_2_CNN/Classifier_2_CNN_000.pth",
                   save_path:str="eval_out/Classifier_2_CNN"):
    model_dict = torch.load(model_path, map_location=device)
    model = Classifier2CNN()
    model.load_state_dict(model_dict)
    model.to(device)
    
    evaluater = Classifier2CNNEvaluater(dataset_loader_test)
    loss, acc = evaluater.evaluate(model)
    
    label_name = ["Smiling","Not Smiling"]
    label_true = np.array(evaluater.gld)
    label_pred = np.array(evaluater.pred)
    accuracy = accuracy_score(label_true,label_pred)
    precision = precision_score(label_true,label_pred)
    recall = recall_score(label_true,label_pred)
    f1 = f1_score(label_true,label_pred)
    roc_auc = roc_auc_score(label_true,label_pred)
    confusion_matrix_path = draw_confusion_matrix(label_true,label_pred,label_name,save_path=save_path)
    
    print("Accuracy: ",accuracy)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("F1: ",f1)
    print("ROC AUC: ",roc_auc)
    print("Confusion matrxi has been saved in ",confusion_matrix_path)
    
    
def evaluate_2_ViT(model_path:str="output/Classifier_2_ViT/Classifier_2_ViT_000.pth",
                   save_path:str="eval_out/Classifier_2_ViT"):
    model_dict = torch.load(model_path, map_location=device)
    model = Classifier2ViT()
    model.load_state_dict(model_dict)
    model.to(device)
    
    evaluater = Classifier2ViTEvaluater(dataset_loader_test)
    loss, acc = evaluater.evaluate(model)
    
    label_name = ["Smiling","Not Smiling"]
    label_true = np.array(evaluater.gld)
    label_pred = np.array(evaluater.pred)
    accuracy = accuracy_score(label_true,label_pred)
    precision = precision_score(label_true,label_pred)
    recall = recall_score(label_true,label_pred)
    f1 = f1_score(label_true,label_pred)
    roc_auc = roc_auc_score(label_true,label_pred)
    confusion_matrix_path = draw_confusion_matrix(label_true,label_pred,label_name,save_path=save_path)
    
    print("Accuracy: ",accuracy)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("F1: ",f1)
    print("ROC AUC: ",roc_auc)
    print("Confusion matrxi has been saved in ",confusion_matrix_path)
    
def evaluate_5_cnn(model_path:str="output/Classifier_5_CNN/Classifier_5_CNN_002.pth",
                  save_path:str="eval_out/Classifier_5_CNN"):
    model_dict = torch.load(model_path, map_location=device)
    model = Classifier5CNN()
    model.load_state_dict(model_dict)
    model.to(device)
    evaluater = Classifier5CNNEvaluater(dataset_loader_test)
    loss, acc = evaluater.evaluate(model)
    
    label_name = evaluater.classifier_kinds
    label_true = np.array(evaluater.gld)
    label_pred = np.array(evaluater.pred)
    draw_confusion_matrix(label_true,label_pred,label_name,save_path=save_path)

def evaluate_5_ViT(model_path:str="output/Classifier_5_ViT/Classifier_5_ViT_002.pth",
                  save_path:str="eval_out/Classifier_5_ViT"):
    model_dict = torch.load(model_path, map_location=device)
    model = Classifier5ViT()
    model.load_state_dict(model_dict)
    model.to(device)
    evaluater = Classifier5ViTEvaluater(dataset_loader_test)
    loss, acc = evaluater.evaluate(model)
    
    label_name = evaluater.classifier_kinds
    label_true = np.array(evaluater.gld)
    label_pred = np.array(evaluater.pred)
    draw_confusion_matrix(label_true,label_pred,label_name,save_path=save_path)

def evaluate_n_cnn(model_path:str="output/Classifier_n_CNN/Classifier_n_CNN_001.pth"):
    model_dict = torch.load(model_path, map_location=device)
    model = ClassifierNCNN()
    model.load_state_dict(model_dict)
    model.to(device)
    evaluater = ClassifierNCNNEvaluater(dataset_loader_test)
    loss, acc = evaluater.evaluate(model)

def main(args):
    if args.task == 1:
        if args.model_type == "CNN":
            evaluate_2_cnn()
        else:
            evaluate_2_ViT()
    elif args.task == 2:
        if args.model_type == "CNN":
            evaluate_5_cnn()
        else:
            evaluate_5_ViT()
    else:
        evaluate_n_cnn()

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",type=str,default=None
    )
    parser.add_argument(
        "--model_type",type=str,default="ViT"
    )
    parser.add_argument(
        "--task",type=int,default=3
    )
    parser.add_argument(
        "--save_path",type=str,default="eval_out"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data"
    )
    args = parser.parse_args()
    
    dataset_test = ClassifierDataSet(args, "test")
    dataset_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=True, num_workers=4
    ) 
    main(args)
