import numpy as np








class layer_dense:
    def __init__(self,n_input,n_neurons):
        self.weight=np.random.rand(n_input,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weight)+self.biases

class activation_sigmoid:
    def forward(self,inputs):
        self.outputs=1/(1+np.exp(inputs))
class activation_softmax:
    def forward(self,inputs):
        exp_value=np.exp(inputs-max(inputs))
        probabilities=exp_value/sum(exp_value)
        self.output=probabilities


class loss:
    def calculus(self,outputs,label):
        sample_losses=self.forward(outputs,label)
        data_loss=np.mean(sample_losses)
        return data_loss

class losscategoricalcrossentropy(loss):
    def forward(self,y_pred,y_targ):
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)
        loss=-np.log(y_pred_clipped)
        return loss



