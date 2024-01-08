
import matplotlib
matplotlib.use('TKAgg')
import os

import matplotlib.pyplot as plt

plt2=matplotlib.pyplot
fig=plt.figure(figsize=(14, 7))
script_path = os.path.abspath(__file__)
    # Extract the file name from the path
script_name = os.path.basename(script_path)
name_only=os.path.splitext(script_name)
model_name=name_only[0]


x = ['51x38','76x57','102x76','128x95','153x114']

y1 =[0.9494,0.9567,0.9555,0.95209,0.9506]
y2=[0.3752,0.4128,0.4061,0.3878,0.3814]
y3=[0.9828,0.9870,0.9564,0.9854,0.9916]
y4=[0.9483,0.9557,0.9545,0.9509,0.9493]
y5=[0.5430,0.5821,0.5753,0.5565,0.5510]
y6=[0.2577,0.2968,0.3674,0.6312,0.4569]

# z1=["1883","1891","1890","1888","1900"]
# z2=["57580","58026","57952","57735","57635"]
# z3=["3136","2690","2764","2981","3081"]
z4=["33","25","26","28","16"]

plt.subplot(1, 2, 1)
plt.plot(x, y1, label='Accuracy')
plt.plot(x, y2, label='Precision')
plt.plot(x, y3, label='Recall')
plt.plot(x, y4, label='Specificity')
plt.plot(x, y5, label='F1-Score')
plt.xlabel('Image Resolution')
plt.ylabel('Observed Values')
plt.title('Experiment 3 - Comparison of metrics')
plt.legend()
# plt.ylim([0, 1])

plt.subplot(1, 2, 2)
# plt.plot(x, z1, label='TP')
# plt.plot(x, z2, label='TN')
# plt.plot(x, z3, label='FP')
plt.plot(x, z4, label='FN')
plt.gca().invert_yaxis()  # Invert the y-axis to make it ascending
plt.xlabel('Image Resolution')
plt.ylabel('Count')
plt.title('Confusion Matrix')
# plt.ylim([0, 1])


# # Plotting multiple curves with legends
# plt.plot(x, y1, label='Accuracy')
# plt.plot(x, y2, label='Precision')
# plt.plot(x, y3, label='Recall')
# plt.plot(x, y4, label='Specificity')
# plt.plot(x, y5, label='F1-Score')
# # plt.plot(x, y6, label='Loss')

# Adding labels and title


# Adding legend 
plt.legend()

savepath=f'./Results_plotting/{model_name}.png'
plt.savefig(savepath)
# Display the plot
plt.show()




