import numpy as np
#import matplotlib.pyplot as plt
#import time
 

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        features=iris[:,:-1]
        return np.mean(features, axis=0)
        

    def covariance_matrix(self, iris):
        dim=iris.shape[1]-1
        n_samples=iris.shape[0]
        mean_features=self.feature_means(iris)
        cov_matrix=np.zeros((dim,dim))
        
        for i in range(dim):
          for j in range(dim):
              val=np.sum((iris[:,i]-mean_features[i])*(iris[:,j]-mean_features[j]))  
              cov_matrix[i][j]=val/(n_samples-1)
        return cov_matrix

    def feature_means_class_1(self, iris):
         class1=np.where(iris[:,-1]==1)[0]
         samples_c1=iris[class1,:-1] 
         return(np.mean(samples_c1,axis=0))

    def covariance_matrix_class_1(self, iris):
        class1=np.where(iris[:,-1]==1)[0]
        samples_c1=iris[class1]
        
        return (self.covariance_matrix(samples_c1))


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels

    def compute_predictions(self, test_data):
        neighbors_ind=[]
        counts=np.ones((test_data.shape[0],len(self.label_list)))
        classes_pred=np.zeros(test_data.shape[0])
        
        for (i, ex) in enumerate(test_data):
            distances= (np.sum((ex - self.train_inputs) ** 2, axis=1))**(0.5)
            neighbors_ind=[]
            for j in range(len(distances)):
              if(distances[j]<self.h):
                neighbors_ind.append(j)
            
            for j in neighbors_ind:
                ind=int(self.train_labels[j])
                counts[i,ind-1]+=1     
            
            if(len(neighbors_ind)==0):
              classes_pred[i]=draw_rand_label(ex, self.label_list)    
            else:  
              classes_pred[i]=np.argmax(counts[i])+1  
                     
        return classes_pred   
            

class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels

    def compute_predictions(self, test_data):
        counts=np.ones((test_data.shape[0],len(self.label_list)))
        classes_pred=np.zeros(test_data.shape[0])
        
        for (i, ex) in enumerate(test_data):
            d= (np.sum((ex - self.train_inputs) ** 2, axis=1))**(0.5)
            
            for j in range(len(d)): 
               const1=((2*np.pi)**(d[j]/2.0))*(self.sigma**d[j])
               const2=(d[j]**2)/(self.sigma**2)
               kernel=(1.0/const1)*np.exp(-0.5*const2)
               ind=int(self.train_labels[j])
               counts[i,ind-1]+=kernel
            
            classes_pred[i]=np.argmax(counts[i])+1
            
        return classes_pred

def split_dataset(iris):
    i_train=[]
    i_valid=[]
    i_test=[]
    for i in range(iris.shape[0]):
        if(i%5==0 or i%5==1 or i%5==2):
          i_train.append(i)
        if(i%5==3):
          i_valid.append(i)
        if(i%5==4):
          i_test.append(i)
    
    train=iris[i_train,:]
    valid=iris[i_valid,:]
    test=iris[i_test,:]
    
    split_data=(train,valid,test)
    return split_data 
                 

class ErrorRate:
    
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        f=HardParzen(h)
        f.train(self.x_train, self.y_train)
        classes_pred=f.compute_predictions(self.x_val)
        
        count=0
        for i in range(len(classes_pred)):
            if(classes_pred[i]==self.y_val[i]):
              count+=1    
        prop=count/len(self.y_val)
        return (1*(1.0-prop)) 

    def soft_parzen(self, sigma):
        f=SoftRBFParzen(sigma)
        f.train(self.x_train, self.y_train)
        classes_pred=f.compute_predictions(self.x_val)
        
        count=0
        for i in range(len(classes_pred)):
            if(classes_pred[i]==self.y_val[i]):
              count+=1    
        prop=count/len(self.y_val)
        l_error=1*(1.0-prop)
        return (l_error)


def get_test_errors(iris):
    train,valid,test=split_dataset(iris)
    x_train=train[:,:-1]
    y_train=train[:,-1].astype('int32')
    x_val=valid[:,:-1]
    y_val=valid[:,-1].astype('int32')

    test_error = ErrorRate(x_train, y_train, x_val, y_val)
    
    h=[0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    sigma=[0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    
    error_hardParzen=[]
    for hi in h:
       error_hardParzen.append(test_error.hard_parzen(hi)) 
       
    min_errorHP_ind=np.argmin(error_hardParzen)
    h_star=h[min_errorHP_ind]
    
    error_softParzen=[]
    for sigma_i in sigma:
       error_softParzen.append(test_error.soft_parzen(sigma_i)) 
       
    min_errorSP_ind=np.argmin(error_softParzen)
    sigma_star=sigma[min_errorSP_ind]
    
    x_val=test[:,:-1]
    y_val=test[:,-1].astype('int32')
    test_error = ErrorRate(x_train, y_train, x_val, y_val)
    test_error_hstar=test_error.hard_parzen(h_star)
    test_error_sstar=test_error.soft_parzen(sigma_star)
    
    error=[test_error_hstar,  test_error_sstar]
    
    return error

def random_projections(X, A):
    Y=(1.0/np.sqrt(2))*np.dot(X,A)
    return Y








