import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from transformers import BertModel, BertTokenizer, BertConfig, BertLMHeadModel
import timm
import numpy as np
import math
import wandb
import time
import argparse
import torch.nn.functional as F
from torchsummary import summary

class ViT(nn.Module):
    def __init__(self, output_dim):
        super(ViT, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=output_dim)
        # self.vit.load_state_dict(torch.load('/home/liyue/CLIP/vit_small_p16_224-15ec54c9.pth'))
    def forward(self, x):
        return self.vit(x)
    
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        BERT_LOCAL_PATH = '/home/liyue/model/bert-base-uncased'
        self.model = BertModel.from_pretrained(BERT_LOCAL_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)

    def forward(self, texts):
        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].to(torch.device("cuda:1"))
        outputs = self.model(**encoded_input)
        return outputs.last_hidden_state[:, 0, :]
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.2)
        # nn.init.xavier_normal_(self.query_linear.weight, gain=1.414)
        # nn.init.xavier_normal_(self.key_linear.weight, gain=1.414)
        # nn.init.xavier_normal_(self.value_linear.weight, gain=1.414)

    def forward(self, x):
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, V)
        return context_layer
def load_cifar100_dataset(batch_size):
    transform = transforms.Compose(
            [
                transforms.Resize((224, 224)), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15), 
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
    train_dataset = CIFAR100(root='./cifar100', train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root='./cifar100', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader,train_dataset.classes, test_loader,test_dataset.classes
class CLIP(nn.Module):
    def __init__(self, image_output_dim, text_output_dim):
        super(CLIP, self).__init__()
        self.image_encoder = ViT(image_output_dim)
        self.text_encoder = TextEncoder()
       
        self.W_i = nn.Parameter(torch.randn(image_output_dim, text_output_dim))
        self.W_t = nn.Parameter(torch.randn(768, text_output_dim))  # BERT-base的最后隐藏层大小为768
        nn.init.normal_(self.W_i, std=0.2)
        nn.init.normal_(self.W_t, std=0.2)

    def forward(self, images, texts):
        I_f = self.image_encoder(images) # (B,3,224,224) -> (B, 512)
        T_f = self.text_encoder(texts) # （B）-> (B, 768)
        I_e = torch.matmul(I_f, self.W_i) # (B, 512)
        T_e = torch.matmul(T_f, self.W_t) # (B, 512)
        logits = torch.matmul(I_e, T_e.T) # (B,B)
        return logits
class CLIP_Next(nn.Module):
    def __init__(self, image_output_dim, text_output_dim):
        super(CLIP_Next, self).__init__()
        self.image_encoder = ViT(image_output_dim)
        self.text_encoder = TextEncoder()
        self.self_attention_image = SelfAttention(image_output_dim)
        self.self_attention_text = SelfAttention(768)
        self.W_i = nn.Parameter(torch.randn(image_output_dim, text_output_dim))
        self.W_t = nn.Parameter(torch.randn(768, text_output_dim))
        nn.init.normal_(self.W_i, std=0.2)
        nn.init.normal_(self.W_t, std=0.2)
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(text_output_dim, 512), 
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),  
            nn.ReLU(),
            nn.BatchNorm1d(512),          
            nn.Dropout(0.2),        
            nn.Linear(512, text_output_dim)  
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(text_output_dim + text_output_dim, 512), 
            nn.ReLU(),
            nn.BatchNorm1d(512),              
            nn.Linear(512, 512),          
            nn.ReLU(),
            nn.BatchNorm1d(512),          
            nn.Dropout(0.2),                  
            nn.Linear(512, text_output_dim)   
        )
        self._init_weights(self.mlp2)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, images, texts):
        I_f = self.image_encoder(images)
        T_f = self.text_encoder(texts) 
        
        I_e = torch.matmul(I_f, self.W_i) 
        T_e = torch.matmul(T_f, self.W_t) 
        combined_features = torch.cat((I_e, T_e), dim=1)
        MLP_output = self.mlp2(combined_features)
        logits = torch.matmul(MLP_output, T_e.T)

        return logits
    
def load_checkpoint(checkpoint_path, model, optimizer, device):
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer
def main(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_dataset,train_classes,test_dataset,test_classes = load_cifar100_dataset(args.batch_size)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    if args.train :
        wandb.init(project="test")
        if args.version == 2 :
            clip_model = CLIP(image_output_dim=512, text_output_dim=512).to(device)
            optimizer = torch.optim.Adam(clip_model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
            for param in clip_model.image_encoder.parameters():
                param.requires_grad = False
            for param in clip_model.text_encoder.parameters():
                param.requires_grad = False
            print(len(train_dataset))
            print(len(test_dataset))
            for epoch in range(args.num_epochs):
                j = 1
                correct = 0
                total = 0
                for images, labels in train_dataset:
                    print('start: epoch='+str(epoch+1)+' j='+str(j)+' '+ str(time.localtime()))
                    j+=1
                    images = torch.stack([image.to(device) for image in images])
                    texts = [train_classes[label] for label in labels]
                    # texts = torch.stack([train_classes[label].to(device) for label in labels])
                    logits = clip_model(images, texts)
                    labels = torch.arange(logits.shape[0]).to(device)
                    # labels = torch.tensor(labels).to(device)
                    loss_i = loss_fn(logits, labels)
                    loss_t = loss_fn(logits.T, labels)
                    loss = (loss_i + loss_t) / 2

                    total += logits.shape[0]
                    predictions = torch.argmax(logits, dim=1)
                    now_acc = (predictions == torch.arange(logits.shape[0]).to(device)).sum().item()
                    correct += now_acc

                    wandb.log({"loss_i" : loss_i, "loss_t":loss_t, "loss":loss, "acc":now_acc/logits.shape[0]})

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                accuracy = correct / total
                print(f'Epoch {epoch}, Loss: {loss.item()}, Acc:{accuracy}')
            
            torch.save({
                'model_state_dict': clip_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.save_pth)#'clip_model_finetuning_v2.pth')
        elif args.version == 3:
            clip_model = CLIP_Next(image_output_dim=512, text_output_dim=512).to(device)
            optimizer = torch.optim.Adam(clip_model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
            for param in clip_model.image_encoder.parameters():
                param.requires_grad = False
            for param in clip_model.text_encoder.parameters():
                param.requires_grad = False
            print(len(train_dataset))
            print(len(test_dataset))
            for epoch in range(args.num_epochs):
                j = 1
                correct = 0
                total = 0
                for images, labels in train_dataset:
                    print('start: epoch='+str(epoch+1)+' j='+str(j)+' '+ str(time.localtime()))
                    j+=1
                    images = torch.stack([image.to(device) for image in images])
                    texts = [train_classes[label] for label in labels]
                    # texts = torch.stack([train_classes[label].to(device) for label in labels])
                    logits = clip_model(images, texts)
                    labels = torch.arange(logits.shape[0]).to(device)
                    # labels = torch.tensor(labels).to(device)
                    loss_i = loss_fn(logits, labels)
                    loss_t = loss_fn(logits.T, labels)
                    loss = (loss_i + loss_t) / 2

                    total += logits.shape[0]
                    predictions = torch.argmax(logits, dim=1)
                    now_acc = (predictions == torch.arange(logits.shape[0]).to(device)).sum().item()
                    correct += now_acc

                    wandb.log({"loss_i" : loss_i, "loss_t":loss_t, "loss":loss, "acc":now_acc/logits.shape[0]})

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                accuracy = correct / total
                print(f'Epoch {epoch}, Loss: {loss.item()}, Acc:{accuracy}')
            
            torch.save({
                'model_state_dict': clip_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.save_pth)#'clip_model_finetuning_v2.pth')

        wandb.finish()
    else :
        if args.model_path != None:
            if args.version == 2:
                clip_model = CLIP(image_output_dim=512, text_output_dim=512).to(device)
                optimizer = torch.optim.Adam(clip_model.parameters(), lr=5e-4, weight_decay=1e-4)
                clip_model, optimizer = load_checkpoint(args.model_path, clip_model, optimizer, device)
            elif args.version == 3:
                clip_model = CLIP_Next(image_output_dim=512, text_output_dim=512).to(device)
                optimizer = torch.optim.Adam(clip_model.parameters(), lr=5e-4, weight_decay=1e-4)
                clip_model, optimizer = load_checkpoint(args.model_path, clip_model, optimizer, device)
            # summary(clip_model, (3, 224, 224), (768))
            with open("clip" + args.model_path.split('.')[0].split("-")[-1] + ".txt", 'w',encoding='utf-8') as file:
                file.write(str(clip_model))
    clip_model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataset:
            images = torch.stack([image.to(device) for image in images]) 
            texts = [test_classes[label] for label in labels]
            logits = clip_model(images, texts)
            predictions = torch.argmax(logits, dim=1)
            predicted_classes = [test_classes[idx] for idx in predictions.cpu().numpy()]
            correct += (predictions == torch.arange(logits.shape[0]).to(device)).sum().item()
            total += logits.shape[0]
            labels = torch.arange(logits.shape[0]).to(device)
            loss_i = loss_fn(logits, labels)
            loss_t = loss_fn(logits.T, labels)
            loss = (loss_i + loss_t) / 2
            total_loss += loss.item()
            for i in range(logits.shape[0]):
                if total < logits.shape[0] * 2:
                    print(f"Image {i}, Predicted: {predicted_classes[i]}, Actual: {test_classes[labels[i]]}")  
    average_loss = total_loss / len(test_dataset)
    # 
    print(f'Average Loss on Test Set: {average_loss}')
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='输入数据的批量大小')
    parser.add_argument('--num_epochs', type=int, default=16, help='训练的轮数')
    parser.add_argument('--train', action='store_true', help='是否需要训练')
    parser.add_argument('--model_path',type=str, default='None',help='已经训练好的模型位置')
    parser.add_argument('--version', type=int, default=2, help='默认2为重现model，3为改进model')
    parser.add_argument('--save_pth', type=str, default='clip_model_finetuning_v2.pth')
    args = parser.parse_args()
    main(args)
