# Build A Simple Classifier - From CNNs to ViTs

## Environment

First use conda to create a virtual environment:

```bash
conda create -n classifier python==3.8 -y
```

Then install the required python model with pip:

```bash
pip install -n requirements.txt
```

The following packages will be installed in this process:

```txt
numpy
wandb
tqdm
Pillow
scikit-learn
matplotlib
pandas
torchvision==0.15.2
torch==2.0.1
```

## Dataset/Pretrained Weights Setup

The downloaded ones are both zip files. Unzip and put them under the root workspace directory.

- Download the preprocessed dataset from [Google Drive](https://drive.google.com/file/d/1jCCMe54y1CeU4jUemfAojBEmnNBhI1l8/view?usp=drive_link).

- *[Optional]*: Download the pretrained weight from [Google Drive]([output – Google Drive](https://drive.google.com/drive/folders/1l-dNBNtWrmYyKMuYP-X0t2a463Wz-urm?usp=sharing)) for evaluation.

## Train & Evaluate

### Train

Run the train script for training:

```bash
bash train.sh
```

The following parameters can be assigned:

```txt
--task=1/2/3: 
    1 for two kinds classifier, 2 for five-kinds classifier, 3 for multi-label classifier
--model_type="CNN"/"ViT": which model to use
    the default CNN model: pretrained resnet152; ViT: pretrained vit_b_32
--data_dir="data"/"data_resampled": which dataset for training; 
    "data" for task 1/3, "data_resampled" for task 2

# HyperParameters
--lr: learning rate
--batch_size
--epoch: 
    One epoch is enough for task 1; 5/2 epoches is recommend for task 2/3
```

Trained model dict will be saved to `./output/[model_name]` directory for following evaluation process 

### Evaluate

Run the evaluate script for evaluation:

```bash
bash evaluate.sh
```

The following parameters can be assigned:

```
--task=1/2/3:
    1 for two kinds classifier, 2 for five-kinds classifier, 3 for multi-label classifier
--model_type="CNN"/"ViT": which model to use
    the default CNN model: pretrained resnet152; ViT: pretrained vit_b_32
--data_dir="data"/"data_resampled": which dataset for training; 
    "data" for task 1/3, "data_resampled" for task 2
```

## File Structure

```txt
.
├── README.md
├── requirements.txt
├── train.py    
├── train.sh
├── evaluate.py
├── evaluate.sh
├── dataset
│   ├── __init__.py
│   ├── classifier_dataset.py    # Create dataset, including img and anno
│   ├── data_agumentation.py    # Use data agumentation for imgs
│   ├── data_preprocess.py    # Split original dataset to train/val/split
│   └── data_resample.py    # Balance the dataset propotion
├── models
│   ├── __init__.py
│   ├── classifier2_ViT.py    # Use pretrained ViT for task 1
│   ├── classifier2_cnn.py    # Use pretrained AlexNet/VGG/ResNet for task 1
│   ├── classifier5_ViT.py    # Use pretrained ViT for task 2
│   ├── classifier5_cnn.py    # Use pretrained AlexNet/VGG/ResNet for task 2
│   └── classifier_n_cnn.py    # Use pretrain ResNet101 for task 3
└── utils
    ├── __init__.py
    ├── draw_confusion_matrix.py 
    └── loss_fn.py
```
