import numpy as np

class SVM:
     def __init__(self,eta, C, niter, batch_size, verbose):
         self.eta = eta; self.C = C; self.niter = niter; self.batch_size = batch_size; self.verbose = verbose

     def make_one_versus_all_labels(self, y, m):
          """
		y : numpy array of shape (n,)
		m : int (in this homework, m will be 10)
		returns : numpy array of shape (n,m)
          """
          
          Lf=[]
          for pos in y:
             arrI=np.ones(m)*-1
             arrI[pos]=1
             Lf.append(arrI)
             
                 
          return np.array(Lf)        
 
     def compute_loss(self, x, y):
        """
  	   x : numpy array of shape (minibatch size, 401)
  	   y : numpy array of shape (minibatch size, 10)
  	   returns : float
        """
        n=x.shape[0]
        m=10
        loss=[] 
        for i in range(n):
             loss_i=0.0
             scores=np.dot((self.w.T),x[i])
             for j in range(m):
                  loss_i+=(np.maximum(0, 1.0-scores[j]*y[i,j]))**2
             loss.append(loss_i)
        
        lossT=(self.C)*np.mean(loss)
        lossT+=0.5* np.sum(self.w * self.w)        
        return lossT   
        
     def compute_gradient(self, x, y):
         """
   		x : numpy array of shape (minibatch size, 401)
   		y : numpy array of shape (minibatch size, 10)
   		returns : numpy array of shape (401, 10)
         """
         n=x.shape[0]
         p=x.shape[1]
         m=10
         grad=np.zeros((p,m))
        
         for i in range(n):
              scores=np.dot((self.w.T),x[i])
              for j in range(m):
                   margin=1.0-scores[j]*y[i,j]
                   if(margin>0): 
                       factor=scores[j]*x[i,:]-x[i,:]*y[i,j]
                       grad[:,j]+=factor
         
         grad=self.w+(2.0*self.C/n)*grad         
         return grad 
     
               
     # Batcher function
     def minibatch(self, iterable1, iterable2, size=1):
          l = len(iterable1)
          n = size
          for ndx in range(0, l, n):
              index2 = min(ndx + n, l)
              yield iterable1[ndx: index2], iterable2[ndx: index2]
              
              
     def infer(self, x):
       """
	  x : numpy array of shape (number of examples to infer, 401)
	  returns : numpy array of shape (number of examples to infer, 10)
       """
       n=x.shape[0]
       m=10
       y_infer=np.dot(x,self.w)
       
       for i in range(n):
          row=y_infer[i]
          max_pos=np.argmax(row)
          mask=np.ones(len(row))*-1
          mask[max_pos]=1
          y_infer[i]=mask            
       
       return y_infer    
       
     def compute_accuracy(self, y_inferred, y):
          """
  		y_inferred : numpy array of shape (number of examples, 10)
  		y : numpy array of shape (number of examples, 10)
  		returns : float
          """
          sum=0.0
          for i in range(len(y)):
              infer_label=np.argmax(y_inferred[i]) 
              real_label=np.argmax(y[i])
              if(infer_label==real_label):
                   sum+=1
          
          accuracy=sum/(len(y)) 
          return accuracy
          
     def fit(self, x_train, y_train, x_test, y_test):
          """
		x_train : numpy array of shape (number of training examples, 401)
		y_train : numpy array of shape (number of training examples, 10)
		x_test : numpy array of shape (number of training examples, 401)
		y_test : numpy array of shape (number of training examples, 10)
		returns : float, float, float, float
          """
          self.num_features = x_train.shape[1]
          self.m = y_train.max() + 1
          y_train = self.make_one_versus_all_labels(y_train, self.m)
          y_test = self.make_one_versus_all_labels(y_test, self.m)
          self.w = np.zeros([self.num_features, self.m])          
          
          for iteration in range(self.niter):
               
               # Train one pass through the training set
               for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                    
                    grad = self.compute_gradient(x,y)
                    self.w -= self.eta * grad
                    
                    
               # Measure loss and accuracy on training set
               train_loss = self.compute_loss(x_train,y_train)
               y_inferred = self.infer(x_train)
               train_accuracy = self.compute_accuracy(y_inferred, y_train) 
               
               # Measure loss and accuracy on test set
               test_loss = self.compute_loss(x_test,y_test)
               y_inferred = self.infer(x_test)
               test_accuracy = self.compute_accuracy(y_inferred, y_test) 
               
               if self.verbose:
                    print("Iteration %d:" % iteration)
                    print("Train accuracy: %f" % train_accuracy)
                    print("Train loss: %f" % train_loss)
                    print("Test accuracy: %f" % test_accuracy)
                    print("Test loss: %f" % test_loss)
                    print("") 
                    
           
                 
          return train_loss, train_accuracy, test_loss, test_accuracy          
          
         





   