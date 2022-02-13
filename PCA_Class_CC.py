import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import pandas as pd

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
        mat1, mat2 = la.eig(cov_matrix)
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



def HW2_p3(data):
    n_com = min(data.shape[0], data.shape[1])
    PCA = pca_HW(n_components=n_com)
    PCA_fit = PCA.fit(data)
    PCA_cov, PCA_eval, PCA_evecs = PCA.transform(PCA_fit)
    print(PCA_cov)
    PCA_per = PCA.variance_eigenval(PCA_eval)
    PCA_score = PCA.scores(PCA_fit, PCA_evecs)
    return PCA_per, PCA_score, PCA_evecs

def HW2_p4(data):
    data = np.transpose(data)
    n_com = min(data.shape[0], data.shape[1])
    PCA = pca_HW(n_components=n_com)
    PCA_fit = PCA.fit(data)
    PCA_cov, PCA_eval, PCA_evecs = PCA.transform(PCA_fit)
    print(PCA_cov)
    PCA_per = PCA.variance_eigenval(PCA_eval)
    PCA_score = PCA.scores(PCA_fit, PCA_evecs)
    return PCA_per, PCA_score, PCA_evecs

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

def main():
    #data = pd.read_csv("/Users/ciaraconway/Desktop/Homework_2_dataset_prob3.csv")
    data = pd.read_table("/Users/ciaraconway/PycharmProjects/MSA/avg_E_coli_exp380_clean.dat.txt")
    percent, scores, eigenvec = HW2_p3(data)
    raw(data)
    scree(percent, num=2)
    score(scores)
    loadings(eigenvec)
    # data_2 = pd.read_csv("/Users/ciaraconway/Desktop/Homework_2_dataset_prob4.csv", header=0, index_col= "ComponetID")
    # percent2, scores2, eigenvec2 = HW2_p4(data_2)
    # scree(percent2, num=10)
    # score(scores2)
    # loadings(eigenvec2)

if __name__ == '__main__':
    main()




