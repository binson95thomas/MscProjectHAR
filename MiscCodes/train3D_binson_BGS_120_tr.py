import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from dataset_prep_2d import OpticalFlow2DDataset
from timeit import default_timer as timer
from datetime import datetime

class FallDetectionCNN(nn.Module):
    def __init__(self):
        super(FallDetectionCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 128, (3, 3, 3), padding = 1)
        self.conv2 = nn.Conv3d(128, 128, (3, 3, 3), padding = 1)
        self.conv3 = nn.Conv3d(128, 64, (3, 3, 3), padding = 1)
        
        self.pool = nn.MaxPool3d(2)

        # self.fc1 = nn.Linear(64 * 2 * 4 * 6, 64)
        self.fc1 = nn.Linear(64 * 600, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 254)
        self.fc4 = nn.Linear(254, 2) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
def compute_metrics(true_labels, predictions):
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    specificity = tn / (tn + fp)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, specificity, f1

def plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses,tr_losses):
    epochs_range = range(1, len(accuracies) + 1)
    fig=plt.figure(figsize=(20, 13))

    suptitle=f'{str_model_type}: {num_epochs} Epochs with learning rate {learning_rate}'
   
    fig.suptitle(suptitle)

    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, accuracies, 'o-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, precisions, 'o-', label='Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.ylim([0.5, 1])

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, recalls, 'o-', label='Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.ylim([0.5, 1])

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, specificities, 'o-', label='Specificity')
    plt.title('Specificity')
    plt.xlabel('Epochs')
    plt.ylabel('Specificity')
    plt.ylim([0.5, 1])

    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, f1_scores, 'o-', label='F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.ylim([0.5, 1])
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, val_losses, 'o-', label='Validation Loss')
    plt.plot(epochs_range, tr_losses, 'r', label='Training Loss')
    plt.leg
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    savepath=f'Results/plt_{str_model_type}.png'
    plt.savefig(savepath)
    plt.show()

def train_model(dataloader_train, dataloader_val, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Creating Model")

    model = FallDetectionCNN().to(device)
    print(f"Completed model creation")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # tm=datetime.datetime.now()
        # print(tm)
        #New changes
        start =timer()
        model.train()
        for batch_features, batch_labels in dataloader_train:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        epoch_val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in dataloader_val:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                epoch_val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
        avg_tr_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}")
        print("Time Taken: ", timer() -start)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Process Completed at : ", current_time)
        logging_output(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}",log_path)
        # if avg_val_loss < best_loss:
        #     best_loss = avg_val_loss
        #     epochs_no_improve = 0
        #     torch.save(model.state_dict(), 'best_model.pth')
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve == early_stopping_epoch:
        #         print("Early stopping!")    
        #         model.load_state_dict(torch.load('best_model.pth'))
        #         break
            
    plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses, avg_tr_loss)
        
    return model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, precision, recall, specificity, f1

# Function to print to console and write to a file
def logging_output(message, file_path='/def_log'):
    try:
    # Write to file
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        with open(file_path, 'a') as file:
            file.write(f'{current_time} : {message} \n')
    except Exception as e:
        print(f"Error logging: {e}")    


if __name__ == "__main__": 
    features_path = '../GPU_DS/Bal_BGS_Canny_50_150_160x120'
    log_path='./Results/log_tr.txt'
    print(f"Path identified is {features_path}")
    logging_output(f"Path identified is {features_path}",log_path)
    dataset = OpticalFlow2DDataset(features_path)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=np.array(dataset.labels)[train_idx])
    batch_size=16
    num_epochs=50
    learning_rate=0.000001

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    dataloader_train = DataLoader(train_dataset, batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size, shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size, shuffle=False)
    if torch.cuda.is_available() :
        print(f"Running on GPU")
    else:
        print(f"CPU Only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    str_model_type=f'tr_Bal+BGS+Canny(50-150)+size(160x140)_b{batch_size}e{num_epochs}L{learning_rate}'
    print(f"Model is {str_model_type}")
    logging_output(f"Model is {str_model_type}",log_path)
    model = FallDetectionCNN().to(device)
    model = train_model(dataloader_train, dataloader_val,num_epochs,learning_rate)
    
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1 = evaluate_model(model, dataloader_test, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}")
    logging_output(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}",log_path)
    torch.save(model.state_dict(), 'fall_detection_model_3d_canny_BGS120.pth')

