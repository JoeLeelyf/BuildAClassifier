from tqdm import tqdm
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_32
import wandb

from dataset.classifier_dataset import ClassifierDataSet
from utils.loss_fn import LossFN

"""
Classifier all images into two kinds
- classifier_kind: e.g. "Smiling"
"""
class Classifier4ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Classifier_4_ViT"
        self.ViT = vit_b_32(pretrained=True)
        self.classifier_head_ViT = nn.Sequential(
            nn.Linear(1000,4),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        # Vision Transformer
        pred = self.ViT(img)
        pred = self.classifier_head_ViT(pred)
        return pred

class Classifier4ViTTrainer():
    def __init__(self, args, \
            classifier_kinds = ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair"]) -> None:
        self.name = "Classifier_4_ViT"
        self.args = args
        self.classifier_kinds = classifier_kinds
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        dataset = ClassifierDataSet(args, "train")
        dataset_val = ClassifierDataSet(args, "val")
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        ) 
        self.data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, shuffle=False
        )

        self.model = Classifier4ViT()

        self.lossfn = LossFN()
        self.evaluater = Classifier4ViTEvaluater(self.data_loader_val, self.classifier_kinds)

    def train(self):
        self.model.to(self.device)
    
        print(
            f"model.parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        num_epochs = self.args.epoch
        print_freq = self.args.print_freq
        eval_freq = self.args.eval_freq
        
        config = dict(
            model_name = self.name,
            epochs = self.args.epoch,
            classes = 2,
            batch_size = self.args.batch_size,
            learning_rate = self.args.lr,
            agumentation = self.args.data_agumentation
        )
        recorder = wandb.init(project="AI_Project2")
        recorder.config.update(config)
        recorder.watch(self.model)

        for epoch in range(num_epochs):
            self.model.train()

            for step, sample_batched in enumerate(tqdm(self.data_loader)):
                img,anno = (
                    sample_batched["img"],
                    sample_batched["anno"]
                )
                
                gld = [anno[self.classifier_kinds[i]].tolist() for i in range(len(self.classifier_kinds))]
                gld = torch.tensor(gld, dtype=torch.float32, device=self.device).T.to(self.device)
                if gld.shape[0]==1:
                    gld = gld.squeeze()
                
                img = img.to(self.device)

                outputs = self.model(img).squeeze()

                pred_loss = self.lossfn.celoss(outputs, gld)
                loss_value = pred_loss.item()
                
                if step % 25 == 0:
                    recorder.log({"Loss":loss_value})

                if step % print_freq == 0:
                    print(f"Epoch [{epoch+1}], Loss: {loss_value:.8f}")
                    
                if step % eval_freq ==0:
                    _,acc = self.evaluater.evaluate(self.model)
                    recorder.log({"Acc":acc})

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))

                optimizer.zero_grad()
                pred_loss.sum().backward()
                optimizer.step()
                
            lr_scheduler.step()

            print("Evaluating...")
            self.evaluater.evaluate(self.model)
            
            os.makedirs(self.args.save_path, exist_ok=True)
            save_path = os.path.join(self.args.save_path, self.name)
            os.makedirs(save_path,exist_ok=True)
            model_path = os.path.join(
                save_path, "%s_%03d.pth" % (self.name, epoch)
            )
            torch.save(self.model.state_dict(), model_path)
            print("Saving %s" % model_path)

        print("Done training!")

class Classifier4ViTEvaluater():
    def __init__(self, data_loader, classifier_kinds=\
                  ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair"]) -> None:
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_loader = data_loader
        self.classifier_kinds = classifier_kinds
        self.lossfn = LossFN()
        
        self.gld = []
        self.pred = []
        
    def evaluate(self, model):
        model.eval()

        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for step, sample_batched in enumerate(tqdm(self.data_loader)):
                img,anno = (
                    sample_batched["img"],
                    sample_batched["anno"]
                )

                gld = [anno[self.classifier_kinds[i]].tolist() for i in range(len(self.classifier_kinds))]
                gld = torch.tensor(gld, dtype=torch.float32, device=self.device).T.to(self.device)
                if gld.shape[0]==1:
                    gld = gld.squeeze()
                
                img = img.to(self.device)

                outputs = model(img).squeeze()

                pred_loss = self.lossfn.celoss(outputs, gld)
                loss_value = pred_loss.item()
                total_loss += loss_value
                
                self.gld.append(torch.argmax(gld).tolist())
                self.pred.append(torch.argmax(outputs).tolist())

                if torch.argmax(gld) == torch.argmax(outputs):
                    total_acc += 1

        print(f"Test, Loss: {total_loss / len(self.data_loader)}")
        print(f"Test, Acc: {total_acc / len(self.data_loader)}")
                
        return total_loss / len(self.data_loader), total_acc / len(self.data_loader)