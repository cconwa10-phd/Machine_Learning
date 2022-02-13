import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import math
class pca_HW:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, raw_data, mean_centered = False):
        if mean_centered:
            norm_data = raw_data
        else:
            #mean centering
            col_mean = raw_data.mean(axis=0)
            col_mean_array = np.tile(col_mean, reps=(raw_data.shape[0], 1))
            raw_data_mean_centered = raw_data - col_mean_array
            #std centering
            col_std = np.std(raw_data, axis=0)
            col_std_array = np.tile(col_std, reps=(raw_data.shape[0], 1))
            norm_data = raw_data_mean_centered/col_std_array
        return norm_data

    def transform(self, norm_data):
        #cov matrix
        norm_data_trans = np.transpose(norm_data)
        cov_matrix = np.cov(norm_data_trans)
        #eigenvalue decomposition
        mat1, mat2 = np.linalg.eig(cov_matrix)
        #take only real solutions
        mat1_real = mat1.real
        mat2_real = mat2.real
        mat3 = mat1_real.argsort()[::-1]
        eigenvals = mat1_real[mat3]
        eigenvecs = mat2_real[:, mat3]
        return cov_matrix, eigenvals, eigenvecs

    def variance_eigenval(self, eigenvals):
        per_var = eigenvals/ sum(eigenvals)*100
        return per_var

    def scores(self, norm_data, eigenvecs):
        scores_PCA = np.matmul(norm_data, eigenvecs)
        return scores_PCA

class ldaClass:
    def __init__(self, n_components):
        self.n_components = n_components
    def fileRead(self, file, data, labels):
        rData = pd.read_csv(file, header=0)
        x = rData.iloc[:, 0:data]
        y = rData.iloc[:, labels]
        return x, y
    def fit(self, x, target):
        tClasses = np.unique(target)
        meanO = np.mean(x, axis=0)
        mat1_W = np.zeros((x.shape[1], x.shape[1])) #features = x.shape[1]
        mat2_B = np.zeros((x.shape[1], x.shape[1])) #features = x.shape[1]
        for tClass in tClasses:
            meanV = np.mean(x[target == tClass], axis=0)
            mat1_W += (x[target == tClass] - meanV).T.dot(x[target == tClass] - meanV)
            nCom = x[target == tClass].shape[0]
            meanD = (meanV - meanO).values.reshape(x.shape[1], 1)
            mat2_B += nCom*(meanD).dot(meanD.T)
        tran_data = np.linalg.inv(mat1_W).dot(mat2_B)
        return tran_data
    def transform(self, x, tran_data):
        eigenvals, eigenvecs = np.linalg.eig(tran_data)
        # eigenvecs = eigenvecs.real
        # eigenvals = eigenvals.real
        eigenvecs = eigenvecs.T
        sortV = np.argsort(abs(eigenvals))[::-1]
        eigenvals = eigenvals[sortV]
        eigenvecs = eigenvecs[sortV]
        lin_d = eigenvecs[0:self.n_components]
        projData = np.dot(x, lin_d.T)
        return projData, eigenvals, eigenvecs, lin_d

    def dataFrame(self, projData, targets):
        projData = pd.DataFrame(projData, columns=["colA", "colB"])
        projData["target"] = targets
        return projData

    def dataFrame_1(self, projData, targets):
        projData = pd.DataFrame(projData, columns=["colA"])
        projData["colB"] = np.zeros((projData.shape[0], 1))
        projData["target"] = targets
        return projData

        

def raw(rawData):
    plt.scatter(rawData.iloc[:,0], rawData.iloc[:,1], color='blue')
    plt.title("Raw Data")
    plt.xlabel("V1")
    plt.ylabel("V2")
    plt.show()

def scree(PCA_per, num):
    PCA_DF = pd.DataFrame(PCA_per)
    plt.scatter(range(0, num), PCA_DF.iloc[:,0].head(num), color='green')
    plt.title("Scree Plot")
    plt.xlabel("principle component index")
    plt.ylabel("% var explained")
    print(PCA_per)
    plt.show()


def score(PCA_score):
    PCA_DF = pd.DataFrame(PCA_score)
    plt.scatter(PCA_DF.iloc[:,0], PCA_DF.iloc[:,1], color='red')
    plt.title("Score Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    print(PCA_score)
    plt.show()


def loadings(PCA_evecs):
    PCA_DF = pd.DataFrame(PCA_evecs)
    plt.scatter(PCA_DF.iloc[:,0], PCA_DF.iloc[:,1], color='blue')
    plt.title("Loading Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    print(PCA_evecs)
    plt.show()

def lda_sklearn(file):
    data = pd.read_csv(file, header=0)
    x = data.iloc[:, 0:4]
    y = data.iloc[:, 4]
    #y = y.to_numpy()
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_proj = lda.fit_transform(x, y)
    lda_proj = pd.DataFrame(lda_proj, columns=["colA", "colB"])
    lda_proj["target"] = y
    print(lda_proj)
    return lda_proj

def lda_sklearn_check(file):
    data = pd.read_csv(file, header=0)
    x = data.iloc[:, 0:2]
    y = data.iloc[:,2]
    #y = y.to_numpy()
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda_proj = lda.fit_transform(x, y)
    lda_proj = pd.DataFrame(lda_proj, columns=["colA"])
    lda_proj["colB"] = np.zeros((lda_proj.shape[0], 1))
    lda_proj["target"] = y
    print(lda_proj)
    return lda_proj

def plot_raw_lda(data):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    targets = [1, 0]
    colors = ["teal", "g"]
    for target, color in zip(targets, colors):
        indices_keep = data["label"] == target
        ax.scatter(data.loc[indices_keep, "V1"], data.loc[indices_keep, "V2"], c=color, s=data.shape[0])
    ax.legend(targets)
    plt.show()

def plot_raw_lda_pca(data, eigenvecs, lin_d):
    k=50
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    targets = [1, 0]
    colors = ["teal", "g"]
    for target, color in zip(targets, colors):
        indices_keep = data["label"] == target
        ax.scatter(data.loc[indices_keep, "V1"], data.loc[indices_keep, "V2"], c=color, s=data.shape[0])
    ax.plot([0, (-1)*k*eigenvecs[0, 0]], [0, (-1)*k*eigenvecs[1, 0]], color='red', linewidth=3)
    ax.plot([0, (-1)*k*lin_d[0, 0]], [0, (-1)*k*lin_d[1, 0]], color='yellow', linewidth=3)
    plt.show()

def plot_lda_sklearn(lda):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    targets = ["setosa", "versicolor", "virginica"]
    colors = ["teal", "g", "b"]
    for target, color in zip(targets, colors):
        indices_keep = lda["target"] == target
        ax.scatter(lda.loc[indices_keep, "colA"], lda.loc[indices_keep, "colB"], c=color, s=lda.shape[0])
    #plt.scatter(lda[:,0], lda[:,1], color='blue')
    ax.legend(targets)
    plt.show()

def plot_lda_Q2(data):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    targets = [1, 0]
    colors = ["teal", "g"]
    for target, color in zip(targets, colors):
        indices_keep = data["target"] == target
        ax.scatter(data.loc[indices_keep, "colA"], data.loc[indices_keep, "colB"], c=color, s=data.shape[0])
    ax.legend(targets)
    plt.show()

def HW3_pca(file):
    data = pd.read_csv(file)
    n_com = min(data.shape[0], data.shape[1])
    PCA = pca_HW(n_components=n_com)
    PCA_fit = PCA.fit(data)
    PCA_cov, PCA_eval, PCA_evecs = PCA.transform(PCA_fit)
    print(PCA_cov)
    PCA_per = PCA.variance_eigenval(PCA_eval)
    PCA_score = PCA.scores(PCA_fit, PCA_evecs)
    return PCA_per, PCA_score, PCA_evecs

def HW3_Q1(file):
    #sklearn iris
    lda = lda_sklearn(file)
    plot_lda_sklearn(lda)
    #scratch class iris
    lda_cc = ldaClass(2)
    x, y = lda_cc.fileRead(file, 4, 4)
    tran_data = lda_cc.fit(x, y)
    projData, eigenvals, eigenvecs, lin_d = lda_cc.transform(x, tran_data)
    plotLda = lda_cc.dataFrame(projData, y)
    plot_lda_sklearn(plotLda)

def HW3_Q2(file_2):
    percent, scores, eigenvec = HW3_pca(file_2)
    raw(pd.read_csv(file_2))
    scree(percent, num=2)
    score(scores)
    loadings(eigenvec)
    #sklearn
    #lda = lda_sklearn_check(file_2)
    #plot_lda_Q2(lda)
    #scratch code
    lda_cc = ldaClass(1)
    x, y = lda_cc.fileRead(file_2, 2, 2)
    tran_data = lda_cc.fit(x, y)
    projData, eigenvals, eigenvecs, lin_d = lda_cc.transform(x, tran_data)
    plotLda = lda_cc.dataFrame_1(projData, y)
    loadings(eigenvec)
    plot_raw_lda_pca(pd.read_csv(file_2), eigenvec, eigenvecs)
    plot_lda_Q2(plotLda)
    print(lin_d)

def main():
    ### Question 1 ###
    file = "iris.csv"
    HW3_Q1(file)
    ### Question 2 ###
    file_2 = "dataset_1.csv"
    HW3_Q2(file_2)


if __name__ == '__main__':
    main()
