import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

#plt.style.use(['science', 'ieee','no-latex'])

file = uproot.open("MLData.root")  # Replace with your ROOT file
tree = file["mlData"]  # Replace with the actual TTree name
df = tree.arrays(library="pd")
dataFilter = df[['deviation','g4Momentum','pathLength','myMomentum']].dropna()
initialCount = len(dataFilter)
y=[]
x=[]

def getCorrelation():
    for i in range(0,200,1):
        dataFilterNew = dataFilter[(dataFilter["deviation"] >= i*1e-3) & (dataFilter["g4Momentum"] <=3000)]
        #finalCount = len(dataFilter)
        #print(f"Total rows dropped:{initialCount - finalCount} out of {initialCount}")
        #df.dropna().to_csv("filtered_data.csv", index=False, sep=";", header=True)
        scatteringAngle = dataFilterNew['deviation'].to_numpy()
        momentum = dataFilterNew['g4Momentum'].to_numpy()
        pathLength = dataFilterNew['pathLength'].to_numpy()
        df_corr = pd.DataFrame({
            'Scattering Angle': scatteringAngle,
            'Path Length': pathLength,
            'Momentum': momentum
        })
        # Compute Pearson correlation matrix
        correlation_matrix = df_corr.corr()
        y.append(correlation_matrix.iloc[0,2])
        x.append(i)
    plt.plot(x,y, color='crimson')
    plt.title("Momentum-Scattering Angle Correlation")
    plt.xlabel("Scattering Angle Threshold [mRad]")
    plt.ylabel("Pearson correlation")
    plt.tight_layout()
    plt.show()

def dataAnalysis():
    dataFilterNew = dataFilter[dataFilter["deviation"] >= 0 * 1e-3]
    scatteringAngle = dataFilterNew['deviation'].to_numpy()
    momentum = dataFilterNew['g4Momentum'].to_numpy()
    pathLength = dataFilterNew['pathLength'].to_numpy()
    print(f"Std Deviation: {np.std(scatteringAngle)}")
    data =[(scatteringAngle-np.mean(scatteringAngle))/np.std(scatteringAngle), (momentum-np.mean(momentum))/np.std(momentum)]
    #plt.ylim(-2, 30)
    plt.boxplot(data, showfliers=False)
    plt.show()
    plt.scatter(data[0], data[1])
    plt.title("Momentum, v/s Scattering angle")
    print(2*np.std(scatteringAngle)+np.mean(scatteringAngle))
    plt.show()


if __name__=="__main__":
    getCorrelation()
    dataAnalysis()
