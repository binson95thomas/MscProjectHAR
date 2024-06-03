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
from dataset_prep_3d import OpticalFlow3DDataset
from timeit import default_timer as timer
from datetime import datetime
import cv2
import seaborn as sns
import traceback
import random
import gc
from torchsummary import summary
from io import StringIO



# Set seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()
gc.collect()


class research_spatial(nn.Module):
    def __init__(self):
        super(research_spatial, self).__init__()
        self.conv1 = nn.Conv3d(1, 96, (3,3,3), stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2 = nn.Conv3d(96, 256, (3,3,3), stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((2, 2, 2)) 

        self.conv3 = nn.Conv3d(256, 512, (3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(512, 512, (3, 3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv3d(512, 512, (3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        # Fully connected layers
        self.fc6 = nn.Linear(512 * 4 * 4 * 6, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(2048, 2) 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout1(x)
        x = F.relu(self.fc7(x))
        x = self.dropout2(x)
        x = self.fc8(x)

        return x

class Raw_stream(nn.Module):
    def __init__(self):
        super(Raw_stream, self).__init__()
        # Commented for Say Version
        # self.conv1 = nn.Conv3d(1, 128, (3, 3, 3), padding = 1)
        # self.conv2 = nn.Conv3d(128, 128, (3, 3, 3), padding = 1)
        # self.conv3 = nn.Conv3d(128, 64, (3, 3, 3), padding = 1)
        
        # self.pool = nn.MaxPool3d(2)

        # # self.fc1 = nn.Linear(3072, 64)
        # self.fc1 = nn.Linear(64 * 2 * 4 * 6, 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, 254)
        # self.fc4 = nn.Linear(254, 2) 

        # Convolutional layers
        self.conv1 = nn.Conv3d(1, 128, (3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 128, (3, 3, 3), padding=1)
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
        # print(f"shape: {x.shape}")
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        logging_output(f"shape before Linear RAW: {x.shape}",log_dir,debug_log,False)

        x = self.global_avg_pool(x)
        logging_output(f"shape after pool RAW: {x.shape}",log_dir,debug_log,False)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
    
        return x
    
class OF_Stream(nn.Module):
    def __init__(self):
        super(OF_Stream, self).__init__()
        # self.conv1 = nn.Conv3d(2, 128, (3, 3, 3), padding = 1)
        # self.conv2 = nn.Conv3d(128, 128, (3, 3, 3), padding = 1)
        # self.conv3 = nn.Conv3d(128, 64, (3, 3, 3), padding = 1)
        
        # self.pool = nn.MaxPool3d(2)

        # self.fc1 = nn.Linear(64 * 2 * 4 * 6, 64)
        # # self.fc1 = nn.Linear(64 * 48, 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, 254)
        # self.fc4 = nn.Linear(254, 2) 

        
        # Convolutional layers
        self.conv1 = nn.Conv3d(2, 64, (3, 3, 3), padding=1)
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
        # print(f"shape: {x.shape}")
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

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
    
class TwoStreamConv(nn.Module):
    def __init__(self):
        super(TwoStreamConv, self).__init__()
        self.stream1 = Raw_stream()
        self.stream2 = OF_Stream()
        self.fc_fusion = nn.Linear(4, 2)  # Fusion layer for combining logits
        
    def forward(self, x1, x2):
        logits1 = self.stream1(x1)
        logits2 = self.stream2(x2)
        
        # Concatenate logits
        fused_logits = torch.cat((logits1, logits2), dim=1)
        
        # Apply fusion layer
        fusion_logits = self.fc_fusion(fused_logits)
        
        # Apply softmax activation
        output = F.softmax(fusion_logits, dim=1)
        
        return output
    
      
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

    savepath=f'Results/{model_folder_main}/plt_{str_model_type}.png'
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
    plt.savefig(f'./Results/{model_folder_main}/CM_{str_model_type}.png')
    # plt.show()
    
def train_model(dataloader_train1, dataloader_val1,dataloader_train2, dataloader_val2, num_epochs=50, learning_rate=0.001, weight_decay=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Creating Model")

    model = TwoStreamConv().to(device)
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
    # summary(model)

    for epoch in range(num_epochs):
        # tm=datetime.datetime.now()
        # print(tm)
        #New changes
        start =timer()
        model.train()
        epoch_train_losses = []
        for batch_features1,batch_features2 in zip(dataloader_train1, dataloader_train2): 
           
            batch_features1, batch_labels1 = batch_features1
            batch_features2, batch_labels2 = batch_features2
            batch_features1, batch_labels1 =batch_features1.to(device), batch_labels1.to(device)
            batch_features2, batch_labels2 = batch_features2.to(device), batch_labels2.to(device)
           

            outputs = model(batch_features1,batch_features2)

            # logging_output(f"size is {outputs.size()}",log_path)
            # loss = criterion(outputs, torch.cat((batch_labels1, batch_labels2), dim=0))
            loss = criterion(outputs, batch_labels1)
            optimizer.zero_grad()
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
            for batch_features1,batch_features2  in zip(dataloader_val1, dataloader_val2):
                batch_features1, batch_labels1 = batch_features1
                batch_features2, batch_labels2 = batch_features2
                batch_features1, batch_labels1 =batch_features1.to(device), batch_labels1.to(device)
                batch_features2, batch_labels2 = batch_features2.to(device), batch_labels2.to(device)


                outputs = model(batch_features1,batch_features2)
                epoch_val_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                #total += batch_labels1.size(0) + batch_labels2.size(0)
                #correct += (predicted == torch.cat((batch_labels1, batch_labels2), dim=0)).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels1.cpu().numpy())
            #accuracy = 100 * correct / total
            #logging_output(f"Accuracy is {accuracy}",log_path)           
             
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        
        accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}")
        logging_output(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}",log_dir,log_name)
       
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

def evaluate_model(model, dataloader1, dataloader2, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for (batch_features1, batch_labels1), (batch_features2, batch_labels2) in zip(dataloader1, dataloader2):
            batch_features1, batch_labels1 = batch_features1.to(device), batch_labels1.to(device)
            batch_features2, batch_labels2 = batch_features2.to(device), batch_labels2.to(device)

            outputs = model(batch_features1, batch_features2)
            loss = criterion(outputs, batch_labels1)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels1.cpu().numpy())

    accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / (len(dataloader1) + len(dataloader2))

    return avg_loss, accuracy, precision, recall, specificity, f1, all_labels, all_preds


# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     total_loss = 0.0
#     with torch.no_grad():
#        for batch_features1,batch_features2  in zip(dataloader_val1, dataloader_val2):
#             batch_features1, batch_labels1 = batch_features1
#             batch_features2, batch_labels2 = batch_features2
#             batch_features1, batch_labels1 = batch_features1.to(device), batch_labels1.to(device)
#             batch_features2, batch_labels2 = batch_features2.to(device), batch_labels2.to(device)


#             outputs = model(batch_features1,batch_features2)
#             loss = criterion(outputs, torch.cat((batch_labels1, batch_labels2), dim=0))
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(torch.cat((batch_labels1, batch_labels2), dim=0).cpu().numpy())
    
#     accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
#     avg_loss = total_loss / len(dataloader)
    
#     return avg_loss, accuracy, precision, recall, specificity, f1, all_labels, all_preds

# Function to print to console and write to a file
def logging_output(message, File_dir='./', file_name='def_log.txt',print_text=True):
    try:
       
    # Write to file
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if print_text:
            print(f'Output : {message}')
        if not os.path.exists(File_dir):
            try:
                # Create the directory and its parents if they don't exist
                os.makedirs(File_dir)
            except OSError as e:
                print(f"Error creating directory {File_dir}: {e}")
        file_path=f"{File_dir}{file_name}"

        with open(file_path, 'a') as file:
            file.write(f'{current_time} : {message} \n')
    except Exception as e:
        print(f"Error logging to {file_path} : {e}")    


if __name__ == "__main__":
    try:
            #######################################
            # Set these Values according to the requirement

            lbool_model_sub=True
            lbool_UH = False
            lbool_sleep = False
            ##########################################

            script_path = os.path.abspath(__file__)
            # Extract the file name from the path
            script_name = os.path.basename(script_path)
            name_only=os.path.splitext(script_name)
            model_folder_main=name_only[0]
            # model_folder1="TwoStr_LateFusionChannelwise"
            model_sub_folder="CNN_Contour_LateFusion"
            if lbool_model_sub:
                model_folder_main=f"{model_folder_main}/{model_sub_folder}"
            else:
                model_sub_folder=model_folder_main

            log_dir=f'./Results/{model_folder_main}/'
            log_name=f'Trainlog_UH.log'
            debug_log='debug.log'
            

            if lbool_UH:

                # Raw_stream_path= f'C:\\Users\\bt22aak\\GPU_DS\\Exp_6_1_Baseline\\balanced'
                # OF_Stream_path = f'C:\\Users\\bt22aak\\GPU_DS\Exp10_1_Contour_OF\\balanced'
                Raw_stream_path = "C:/Users/bt22aak/GPU_DS/Exp_6_1_Baseline_80_Split/balanced"
                OF_Stream_path = "C:/Users/bt22aak/GPU_DS/Exp10_1_Contour_OF_80_Split/balanced"

                Raw_test_path = f'C:/Users/bt22aak/GPU_DS/Exp_6_1_Baseline_80_Split/unbalanced'
                OF_Test_path = f'C:/Users/bt22aak/GPU_DS/Exp10_1_Contour_OF_80_Split/unbalanced'
            else:

                Raw_stream_path = "D:/UH_KTP/Outputs/Exp_6_1_Baseline_80_Split/balanced"
                OF_Stream_path = "D:/UH_KTP/Outputs/Exp10_1_Contour_OF_80_Split/balanced"

                Raw_test_path = f'D:/UH_KTP/Outputs/Exp_6_1_Baseline_80_Split/unbalanced'
                OF_Test_path = f'D:/UH_KTP/Outputs/Exp10_1_Contour_OF_80_Split/unbalanced'



                

            batch_size=32
            num_epochs=50
            learning_rate=0.0001


            str_model_type=f'{model_sub_folder}_b{batch_size}e{num_epochs}L{learning_rate}'

            if not os.path.exists(f'./Results/{model_folder_main}'):
                try:
                    # Create the directory and its parents if they don't exist
                    os.makedirs(f'./Results/{model_folder_main}')
                except OSError as e:
                    print(f"Error creating directory ./Results/{model_folder_main}: {e}")

            # print(f"Path identified is {stream_1} and {test_path}")
            # logging_output(f"Path identified is {stream_1} and {test_path}",log_path)
            # print(f"Starting the processing of {str_model_type}")
            # logging_output(f"Starting the processing of {str_model_type}",log_path)

            code_start=timer()

            train_val_dataset_raw = OpticalFlow2DDataset(Raw_stream_path)
            train_val_dataset_OF = OpticalFlow3DDataset(OF_Stream_path)

            test_dataset_Raw= OpticalFlow2DDataset(Raw_test_path)
            test_dataset_OF = OpticalFlow3DDataset(OF_Test_path)
            
            train_idx_raw, val_idx_raw = train_test_split(range(len(train_val_dataset_raw)), test_size=0.25, random_state=42, stratify=train_val_dataset_raw.labels)
            train_idx_OF, val_idx_OF = train_test_split(range(len(train_val_dataset_OF)), test_size=0.25, random_state=42, stratify=train_val_dataset_OF.labels)

            train_dataset_raw = torch.utils.data.Subset(train_val_dataset_raw, train_idx_raw)
            val_dataset_raw = torch.utils.data.Subset(train_val_dataset_raw, val_idx_raw)
            train_dataset_OF = torch.utils.data.Subset(train_val_dataset_OF, train_idx_OF)
            val_dataset_OF = torch.utils.data.Subset(train_val_dataset_OF, val_idx_OF)

            dataloader_train_raw = DataLoader(train_dataset_raw, batch_size=32, shuffle=True)
            dataloader_val_raw = DataLoader(val_dataset_raw, batch_size=32, shuffle=False)
            dataloader_train_OF = DataLoader(train_dataset_OF, batch_size=32, shuffle=True)
            dataloader_val_OF = DataLoader(val_dataset_OF, batch_size=32, shuffle=False)

            dataloader_test_Raw = DataLoader(test_dataset_Raw, batch_size=32, shuffle=False)
            dataloader_test_OF = DataLoader(test_dataset_OF, batch_size=32, shuffle=False)

            if torch.cuda.is_available() :
                print(f"Running on GPU")
                logging_output("Running on GPU",log_dir,log_name)
            else:
                print(f"CPU Only")
                logging_output("Running on CPU Only",log_dir,log_name)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"Model is {str_model_type}")
            logging_output(f"Model is {str_model_type}",log_dir,log_name)
            model = TwoStreamConv().to(device)
            saved_model_path = f'./Results/{model_folder_main}/{str_model_type}.pth'

            try:
                model_summ = StringIO()
                summary(model, [(1, 17, 38, 51), ( 2, 17, 38, 51)],file=model_summ)
                str_model_Summ=model_summ.getvalue()
                logging_output(str_model_Summ,log_dir,'Model_summ',False)
            except Exception as E :
                logging_output(f"summary Failed due to {E}",log_dir,debug_log,True)


            if os.path.isfile(saved_model_path):
                model.load_state_dict(torch.load(saved_model_path, map_location=device))
                logging_output(f"Loaded saved model.",log_dir,log_name)

            elif os.path.isfile (f"{saved_model_path}_pre"):
                model.load_state_dict(torch.load(f"{saved_model_path}_pre", map_location=device))
                logging_output(f"Loaded Pre-saved model.",log_dir,log_name)

            else:
                logging_output("No saved model found. Training a new model.",log_dir,log_name)
                
                model = train_model(dataloader_train_raw, dataloader_val_raw,dataloader_train_OF, dataloader_val_OF, num_epochs,learning_rate)
                torch.save(model.state_dict(), f'{saved_model_path}_pre')
            
            criterion = nn.CrossEntropyLoss().to(device)
            test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1, true_labels, predictions = evaluate_model(model, dataloader_test_Raw,dataloader_test_OF, criterion, device)
            
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}")
            logging_output(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}",log_dir,log_name)
            

            print(f"Total Time taken: {timer() -code_start } seconds")

            logging_output(f"Total Time taken: { timer() -code_start } Seconds", log_dir,log_name)

            plot_confusion_matrix(true_labels, predictions, classes = ["No Fall", "Fall"])

            torch.save(model.state_dict(), saved_model_path)

            print("###############Data Process Complete############")
            logging_output("process completed",log_dir,log_name)
            if lbool_sleep:
                print("~~~~~~~~Entering Hibernation Mode ~~~~~~~~~~~~~")
                os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

        
    except Exception as E :
        err_msg=f"Error occured {E}"
        print(err_msg)
        error_stack = traceback.format_exc()
        print("=======================================================================")
        print(f"Error stack:\n{error_stack}")
        logging_output("******************************",log_dir),log_name
        logging_output(err_msg,log_dir,log_name)
        logging_output(error_stack,log_dir,log_name)
        logging_output("******************************",log_dir,log_name)

    