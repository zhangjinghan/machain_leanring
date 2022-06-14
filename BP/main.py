import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torch.autograd import Variable
from torch.nn import *
from torch.optim import Adam

# 超参数定义(由于我们的隐藏层只有一层，所以可以直接定义为超参数)
batch_size=100
input_feature=100
hidden_feature=1000
output_feature=10
learning_rate=1e-6
epochs=1000
loss_f=MSELoss()


# 参数初始化
x=Variable(torch.randn(batch_size,input_feature),requires_grad=False)
y=Variable(torch.randn(batch_size,output_feature),requires_grad=False)
w1=Variable(torch.randn(input_feature,hidden_feature),requires_grad=True)
w2=Variable(torch.randn(hidden_feature,output_feature),requires_grad=True)

Epoch=[]
Loss=[]
model=Sequential(
    Linear(input_feature,hidden_feature),
    Linear(hidden_feature,output_feature)
)
# optimizer需要传入训练参数和lr
optim=Adam(model.parameters(),lr=learning_rate)
print(model)
# 迭代训练
for epoch in tqdm.tqdm(range(1,epochs+1)):
    # 前向传播
    y_pred=model(x)
    loss=loss_f(y_pred,y)

    Epoch.append(epoch)
    Loss.append(loss.data)

    if epoch%50==0:
        print("Epoch:{},loss:{}".format(epoch,loss))
    optim.zero_grad()
    # 后向传播
    loss.backward()
    # 参数微调
    optim.step()
    # for parm in model.parameters():
    #     parm.data-=parm.grad.data*learning_rate   

Epoch=np.array(Epoch)
Loss=np.array(Loss)
plt.plot(Epoch,Loss)
plt.show()
