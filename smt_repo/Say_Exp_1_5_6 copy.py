import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from dataset_prep_2d import OpticalFlow2DDataset
from timeit import default_timer as timer
from datetime import datetime
import cv2
import seaborn as sns
import traceback

class FallDetectionCNN(nn.Module):
    def __init__(self):
        super(FallDetectionCNN, self).__init__()
 

        # Convolutional layers
        self.conv1 = nn.Conv3d(1, 128, (3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2) 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
def visualize_optical_flow(u_component, v_component, upscale_factor=10):
    num_frames, height, width = u_component.shape
    rgb_sequence = np.zeros((num_frames, height * upscale_factor, width * upscale_factor, 3), dtype=np.uint8)
    
    for i in range(num_frames):
        magnitude, angle = cv2.cartToPolar(u_component[i], v_component[i])
        
        # Create an empty HSV image and populate it with the magnitude and angle
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert the HSV image to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Upscale the RGB image
        rgb_upscaled = cv2.resize(rgb, (width * upscale_factor, height * upscale_factor), interpolation=cv2.INTER_LINEAR)
        
        # Store in the sequence
        rgb_sequence[i] = rgb_upscaled
    
    return rgb_sequence
        
def visualize_misclassified_optical_flow(model, dataloader, device):
    model.eval()
    misclassified_samples = []

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
                
            # Get indices of misclassified samples
            misclassified_indices = (predicted != batch_labels).nonzero(as_tuple=True)[0]
                
            for idx in misclassified_indices:
                data = batch_features[idx].cpu().numpy()
                true_label = batch_labels[idx].item()
                predicted_label = predicted[idx].item()
                optical_flow_u = data[0, :, :, :]
                optical_flow_v = data[1, :, :, :]
                optical_flow_rgb = visualize_optical_flow(optical_flow_u, optical_flow_v)
                misclassified_samples.append((optical_flow_rgb, true_label, predicted_label))
                     
    return misclassified_samples
   
    
def compute_metrics(true_labels, predictions):
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    specificity = tn / (tn + fp)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, specificity, f1

def plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses, train_losses):
    epochs_range = range(1, len(accuracies) + 1)
    fig=plt.figure(figsize=(20, 13))

    # suptitle=f'{str_model_type}: {num_epochs} Epochs with learning rate {learning_rate}'
    # fig.suptitle(suptitle)
    plt.figtext(x=0, y=0,s=f"{str_model_type}: {num_epochs} Epochs with learning rate {learning_rate}" )
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, accuracies, 'o-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.ylim([0, 1])

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, precisions, 'o-', label='Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    # plt.ylim([0, 1])

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, recalls, 'o-', label='Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    # plt.ylim([0, 1])

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, specificities, 'o-', label='Specificity')
    plt.title('Specificity')
    plt.xlabel('Epochs')
    plt.ylabel('Specificity')
    # plt.ylim([0, 1])

    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, f1_scores, 'o-', label='F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    # plt.ylim([0, 1])
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, val_losses, 'o-', label='Validation Loss', color='red')
    plt.plot(epochs_range, train_losses, 'o-', label='Training Loss', color='blue')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.tight_layout()

    savepath=f'Results/{model_folder}/plt_{str_model_type}.png'
    plt.savefig(savepath)
    # plt.show()
def calculate_cm_percentages(cm):
    cm_percentages = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    cm_percentages = np.around(cm_percentages * 100, decimals = 2)
    return cm_percentages

def plot_confusion_matrix(true_labels, predictions, classes):
    cm = confusion_matrix(true_labels, predictions)
    cm_percentages = calculate_cm_percentages(cm)
    
    labels = (np.asarray(["{0}\n({1}%)".format(value, percentage)
                         for value, percentage in zip(cm.flatten(), cm_percentages.flatten())])
                ).reshape(cm.shape)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'./Results/{model_folder}/CM_{str_model_type}.png')
    # plt.show()
    
def train_model(dataloader_train, dataloader_val, num_epochs=50, learning_rate=0.001, weight_decay=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Creating Model")

    model = FallDetectionCNN().to(device)
    print(f"Completed model creation")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    val_losses = []
    train_losses = []
    
    for epoch in range(num_epochs):
        # tm=datetime.datetime.now()
        # print(tm)
        #New changes
        start =timer()
        model.train()
        epoch_train_losses = []
        for batch_features, batch_labels in dataloader_train:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
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
        
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        
        accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}")
        logging_output(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}",log_path)
       
        print("Time Taken: ", timer() -start)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Process Completed at : ", current_time)
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
            
    plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses, train_losses)
        
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
    
    return avg_loss, accuracy, precision, recall, specificity, f1, all_labels, all_preds

# Function to print to console and write to a file
def logging_output(message, file_path='./def_log.txt'):
    try:
    # Write to file
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        with open(file_path, 'a') as file:
            file.write(f'{current_time} : {message} \n')
    except Exception as e:
        print(f"Error logging: {e}")    


if __name__ == "__main__":
    try:
        for i in ["Exp_6_80_Split","exp_1_80_Split","Exp_5_80_Split"]:

            script_path = os.path.abspath(__file__)
            # Extract the file name from the path
            script_name = os.path.basename(script_path)
            name_only=os.path.splitext(script_name)
            # model_folder=name_only[0]
            model_folder=i

            features_path = f'C:\\Users\\bt22aak\\GPU_DS\\{model_folder}\\Balanced'
            test_path = f'C:\\Users\\bt22aak\\GPU_DS\\{model_folder}\\Unbalanced'
            batch_size=32
            num_epochs=50
            learning_rate=0.0001


            str_model_type=f'{model_folder}_b{batch_size}e{num_epochs}L{learning_rate}'

            if not os.path.exists(f'./Results/{model_folder}'):
                try:
                    # Create the directory and its parents if they don't exist
                    os.makedirs(f'./Results/{model_folder}')
                except OSError as e:
                    print(f"Error creating directory ./Results/{model_folder}: {e}")

            log_path=f'./Results/{model_folder}/Trainlog.log'
            print(f"Path identified is {features_path} and {test_path}")
            logging_output(f"Path identified is {features_path} and {test_path}",log_path)
            print(f"Starting the processing of {str_model_type}")
            logging_output(f"Starting the processing of {str_model_type}",log_path)

            code_start=timer()

            train_val_dataset = OpticalFlow2DDataset(features_path)
            test_dataset = OpticalFlow2DDataset(test_path)
            
            train_idx, val_idx = train_test_split(range(len(train_val_dataset)), test_size=0.25, random_state=42, stratify=train_val_dataset.labels)

            train_dataset = torch.utils.data.Subset(train_val_dataset, train_idx)
            val_dataset = torch.utils.data.Subset(train_val_dataset, val_idx)

            dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
            dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
            dataloader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)

            if torch.cuda.is_available() :
                print(f"Running on GPU")
                logging_output("Running on GPU",log_path)
            else:
                print(f"CPU Only")
                logging_output("Running on CPU Only",log_path)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"Model is {str_model_type}")
            logging_output(f"Model is {str_model_type}",log_path)
            model = FallDetectionCNN().to(device)
            model = train_model(dataloader_train, dataloader_val,num_epochs,learning_rate)
            
            criterion = nn.CrossEntropyLoss().to(device)
            test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1, true_labels, predictions = evaluate_model(model, dataloader_test, criterion, device)
            
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}")
            logging_output(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}",log_path)
            

            print(f"Total Time taken: {timer() -code_start } seconds")

            logging_output(f"Total Time taken: { timer() -code_start } Seconds", log_path)

            plot_confusion_matrix(true_labels, predictions, classes = ["No Fall", "Fall"])

            torch.save(model.state_dict(), f'./Results/{model_folder}/{str_model_type}.pth')
        
    except Exception as E :
        err_msg=f"Error occured {E}"
        print(err_msg)
        error_stack = traceback.format_exc()
        print(f"Error stack:\n{error_stack}")
        logging_output("******************************",log_path)
        logging_output(err_msg,log_path)
        logging_output(error_stack,log_path)
        logging_output("******************************",log_path)

    