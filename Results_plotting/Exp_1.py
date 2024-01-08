
import matplotlib
matplotlib.use('TKAgg')
import os

import matplotlib.pyplot as plt

plt2=matplotlib.pyplot
fig=plt.figure(figsize=(10, 7))
script_path = os.path.abspath(__file__)
    # Extract the file name from the path
script_name = os.path.basename(script_path)
name_only=os.path.splitext(script_name)
model_name=name_only[0]


x = ["1.1(Canny-1)", "1.2(Canny-2)", "1.3(Laplacian-1)", "1.4(Laplacian-2)", "1.5(Sobel-1)", "1.6(Sobel-2)"]

y1=[0.945,0.9154,0.9543,0.9459,0.9432,0.9419]
y2=[0.3564,0.2629,0.3999,0.36,0.3485,0.3427]
y3=[0.9885,0.9781,0.9854,0.9888,0.9843,0.9812]
y4=[0.9437,0.9135,0.9533,0.9446,0.9419,0.9406]
y5=[0.5293,0.4144,0.5689,0.5277,0.5147,0.508]
# y6==["0.5772","0.9357","0.4837","0.6158","0.8487","0.6327"]

# z1=["1883","1891","1890","1888","1900"]
# z2=["57580","58026","57952","57735","57635"]
# z3=["3136","2690","2764","2981","3081"]
z4=["33","25","26","28","16"]


plt.plot(x, y1, label='Accuracy',linestyle='-', marker='o')
plt.plot(x, y2, label='Precision',linestyle='-',marker='o')
plt.plot(x, y3, label='Recall', linestyle='-', marker='o')
plt.plot(x, y4, label='Specificity',linestyle='-', marker='o')
plt.plot(x, y5, label='F1-Score',linestyle='-',marker='o')
plt.xlabel('Model Types')
plt.ylabel('Observed Values')
plt.title('Experiment 1 - Comparison of metrics')
plt.legend()
# plt.ylim([0, 1])

# Adding legend 
plt.legend()

savepath=f'./Results_plotting/{model_name}.png'
plt.savefig(savepath)
# Display the plot
plt.show()




