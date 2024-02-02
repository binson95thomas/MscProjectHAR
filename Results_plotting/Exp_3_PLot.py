
import matplotlib
matplotlib.use('TKAgg')
import os

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10, 5))

x = ['51x38','76x57','102x76','128x95','153x114']

y1 = [0.9494,0.9567,0.9555,0.95209,0.9506]
y2=[0.3752,0.4128,0.4061,0.3878,0.3814]
y3=[0.9828,0.9870,0.9564,0.9854,0.9916]
y4=[0.9483,0.9557,0.9545,0.9509,0.9493]
y5=[0.5430,0.5821,0.5753,0.5565,0.5510]
y6=[0.2577,0.2968,0.3674,0.6312,0.4569]


# Plotting multiple curves with legends
plt.subplot(1, 2, 1)
plt.plot(x, y1, label='Accuracy')
plt.plot(x, y2, label='Precision')
plt.plot(x, y3, label='Recall')
plt.subplot(1, 2, 1)
plt.plot(x, y4, label='Specificity')
plt.plot(x, y5, label='F1-Score')
# plt.plot(x, y6, label='Loss')

# Adding labels and title
plt.xlabel('Image Resolution')
plt.ylabel('Results')
plt.title('Comaprison of metrics')

# Adding legend 
plt.legend()

script_path = os.path.abspath(__file__)
    # Extract the file name from the path
script_name = os.path.basename(script_path)
name_only=os.path.splitext(script_name)
model_name=name_only[0]

savepath=f'./Results_plotting/{model_name}.png'
plt.savefig(savepath)
# Display the plot
plt.show()