import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import seaborn as sns

#plt.style.use(['ieee','no-latex'])
file = uproot.open("train.root")  # Replace with your ROOT file
tTree = file["groundTruthPoCA"]  # Replace with the actual TTree name
parameters = tTree.keys()  # List available branches
df = pd.DataFrame(tTree.arrays(library="pd"))
df =df.dropna()
df = df[(df['angleDev'] > 5e-3)]
muonFeatures = parameters[0:12] + [parameters[15]]
pocaLabel = parameters[12:15]
print(f'Features:{muonFeatures} and Labels:{pocaLabel}')

def originalData():
    correlation_matrix = df[muonFeatures + pocaLabel].corr()
    plt.figure(figsize=(8, 4))      #12X8 for saving
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', vmin=-1, vmax=1, annot_kws={"size": 6})
    plt.title('Correlation Matrix of Input Data', pad=0.5, fontsize=4)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tick_params(axis='both', which='minor', bottom=False, left=False, top=False, right=False)
    plt.tick_params(axis='both', which='major', bottom=False, left=False, top=False, right=False)
    plt.tight_layout()
    plt.show()

def engineeredData(edf):
    edf =edf.copy()
    for i in ['X','Y','Z']:
        edf['inCosAngle'+i] = np.arccos(edf['dIn'+i])
    for i in ['X','Y','Z']:
        edf['outCosAngle'+i] = np.arccos(edf['dOut'+i])
    edf['AngDev'] = edf['outCosAngleX'] + edf['inCosAngleX']
    edf = edf.drop(columns=muonFeatures[0:12])
    correlation_matrix = edf.corr()
    plt.figure(figsize=(8, 4))      #12X8 for saving
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', vmin=-1, vmax=1, annot_kws={"size": 6})
    plt.title('Correlation Matrix of Input Data', pad=0.5, fontsize=4)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tick_params(axis='both', which='minor', bottom=False, left=False, top=False, right=False)
    plt.tick_params(axis='both', which='major', bottom=False, left=False, top=False, right=False)
    plt.tight_layout()
    plt.show()



if __name__=="__main__":
    originalData()
    engineeredData(df[muonFeatures + pocaLabel])
