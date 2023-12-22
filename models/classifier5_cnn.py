from tqdm import tqdm
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet101
import wandb

from dataset.classifier_dataset import ClassifierDataSet
from utils.loss_fn import LossFN

"""
Classifier all images into five kinds
- classifier_kind: e.g. ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair","Others"]
"""
class Classifier5CNN(nn.Module):
    def __init__(self, input_channel=3, output_channel=64):
        super().__init__()
        self.name = "Classifier_5_CNN"
        # Naive CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 64, 8, stride=2, padding=3),   
            nn.ReLU(),  
            nn.MaxPool2d(2, stride=2), 

            nn.Conv2d(64, 128, 8, stride=2, padding=3),   
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, output_channel, 8, stride=2, padding=3),   
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        # The loss func nn.CrossEntropyLoss includes softmax layer
        self.classifier_head_cnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_channel * 3 * 3, 128),  
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Softmax(dim=1)
        )
        
        # AlexNet
        self.AlexNet = nn.Sequential(
            nn.Conv2d(3,96,11,4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),

            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),

            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
        )
        self.classifier_head_alexnet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 5),
            nn.Softmax(dim=1) 
        )
        
        # VGG
        self.VGG = vgg16(pretrained=True)
        self.classifier_head_VGG = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),		
            nn.ReLU(),					
            nn.Dropout(0.5),				
            nn.Linear(4096, 4096),			
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 5),
            nn.Softmax(dim=1)	
        )

        # ResNet
        self.ResNet = resnet101(pretrained=True)
        num_ftrs = self.ResNet.fc.in_features
        self.ResNet.fc = nn.Linear(num_ftrs, 5)
        self.classifier_head_ResNet = nn.Sequential(
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        # Navie CNN
        # self.name = "NaiveCNN"
        # pred = self.cnn(img)
        # pred = self.classifier_head_cnn(pred)
        
        # AlexNet
        # self.name = "AlexNet"
        # pred  = self.AlexNet(img)
        # pred = self.classifier_head_alexnet(pred)

        #VGG
        # self.name = "VGG"
        # pred = self.VGG.features(img)
        # pred = self.classifier_head_VGG(pred)

        # ResNet
        self.name = "ResNet"
        pred = self.ResNet(img)
        pred = self.classifier_head_ResNet(pred)
        return pred

class Classifier5CNNTrainer():
    def __init__(self, args, \
            classifier_kinds = ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair","Others"]) -> None:
        self.name = "Classifier_5_CNN"
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

        self.model = Classifier5CNN(input_channel=3, output_channel=64)

        self.lossfn = LossFN()
        self.evaluater = Classifier5CNNEvaluater(self.data_loader_val, self.classifier_kinds)

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
            model_name = self.model.name,
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
                
                gld = [anno[self.classifier_kinds[i]].tolist() for i in range(len(self.classifier_kinds) - 1)]
                gld.append([0 if any(row[j] == 1 for row in gld) else 1 for j in range(len(gld[0]))])
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

class Classifier5CNNEvaluater():
    def __init__(self, data_loader, classifier_kinds=\
                  ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair","Others"]) -> None:
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

                gld = [anno[self.classifier_kinds[i]].tolist() for i in range(len(self.classifier_kinds) - 1)]
                gld.append([0 if any(row[j] == 1 for row in gld) else 1 for j in range(len(gld[0]))])
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
                
