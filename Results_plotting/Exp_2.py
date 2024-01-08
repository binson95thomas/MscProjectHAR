
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


x = ["2.1 (Type 1)", "2.2 (Type 1)", "2.3 (Type 1)", "2.4 (Type 2)", "2.5 (Type 2)", "2.6 (Type 2)", "2.7 (Type 2)"]

y1=[0.9427,0.9509,0.9477,0.9542,0.9444,0.9381,0.9459] 
y2=[0.347,0.3829,0.3676,0.3994,0.35399,0.3297,0.3599] 
y3=[0.989,0.989,0.9849,0.9864,0.9901,0.989,0.9864] 
y4=[0.9413,0.9497,0.9465,0.9532,0.9429,0.9365,0.9446] 
y5=[0.5138,0.5521,0.5354,0.568,0.5214,0.4945,0.5273] 
# y6==[0.523,0.5556,0.5689,0.5514,0.5352,0.6075,0.5862]

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
plt.title('Experiment 2 - Comparison of test metrics')
plt.legend()
# plt.ylim([0, 1])

# Adding legend 
plt.legend()

savepath=f'./Results_plotting/{model_name}.png'
plt.savefig(savepath)
# Display the plot
plt.show()




