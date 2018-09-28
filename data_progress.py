import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


data = np.genfromtxt('18_pdf_1.csv', delimiter=',')
data_all = pd.read_csv("18_pdf_1_all.csv")
def number_drift(data):
    
    x = np.arange(-10,10,0.001)
    sigma = 1
    def norm_pdf(x,mu,sigma):
        pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))    
        return pdf

    mu = 0    
    plt.hist(data, bins=25, alpha=0.7)
    #y = norm_pdf(x, mu, sigma)
    #plt.plot(x,y, color='orange', lw=3)
    plt.show()
#number_drift(data)

def group(data_all):
    X_group = data_all.groupby('PCI').sum()
    print "X_group==",X_group
    
group(data_all)