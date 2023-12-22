from datetime import datetime

import torch

import argparse
from models.classifier2_cnn import Classifier2CNNTrainer
from models.classifier5_cnn import Classifier5CNNTrainer
from models.classifier2_ViT import Classifier2ViTTrainer
from models.classifier5_ViT import Classifier5ViTTrainer
from models.classifier_n_cnn import ClassifierNCNNTrainer

def main(args):
    print("Task:", args.task)
    print("Learning Rate:", args.lr)
    print("Batch Size:",args.batch_size)
    print()
    if args.task == 1:
        if args.model_type == "CNN":
            trainer = Classifier2CNNTrainer(args, classifier_kind = "Smiling")
        else:
            trainer = Classifier2ViTTrainer(args, classifier_kind="Smiling")
    elif args.task == 2:
        if args.model_type == "CNN":
            trainer = Classifier5CNNTrainer(args,classifier_kinds=\
                ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair","Others"])
        else:
            trainer = Classifier5ViTTrainer(args,classifier_kinds=\
                ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair","Others"])
    else:
        trainer = ClassifierNCNNTrainer(args)
    trainer.train()

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--save_name", type=str, default="Classifier")
    parser.add_argument("--data_agumentation", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="ViT")
    
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=300)
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epoch", type=int, default=5)
    
    args = parser.parse_args()
    
    start = datetime.now()
    print("start time: ", start)
    main(args)

    end = datetime.now()
    time_cost = (end - start).total_seconds()
    print("end time: ", end)
    print("time cost (s): ", time_cost)
    print("All finsihed!")
    