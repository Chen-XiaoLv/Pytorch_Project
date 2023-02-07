import math
import random


# step 1. 构建常用函数

# 激活函数
def sigmoid(x):
    return math.tanh(x)
def ReLU(x):
    return x if x>0 else 0
def derived_sigmiod(x):
    # (O)(1-O)(T-O)
    return x-x**2


# 生成随机数
def getRandom(a,b):
    return (b-a)*random.random()+a

# 生成一个矩阵
def makeMatrix(m,n,val=0.0):
    # 默认以0填充这个m*n的矩阵
    return [[val]*n for _ in range(m)]

# step 2. 初始化参数
# 这部分主要有：节点个数、隐层个数、输出层个数
# 可以类似于torch.nn.Linear
class BPNN:
    def __init__(self,n_in,n_out,n_hidden=10,lr=0.1,m=0.1):
        self.n_in=n_in+1 # 加一个偏置节点
        self.n_hidden=n_hidden+1 # 加一个偏置节点
        self.n_out=n_out
        self.lr=lr
        self.m=m

        # 生成链接权重
        # 这里用的是全连接，所以对应的映射就是 [节点个数A，节点个数B]
        self.weight_hidden=makeMatrix(self.n_in,self.n_hidden)
        self.weight_out=makeMatrix(self.n_hidden,self.n_out)
        # 对权重进行初始化
        for i,row in enumerate(self.weight_hidden):
            for j,val in enumerate(row):
                self.weight_hidden[i][j]=getRandom(-0.2,0.2)
        for i,row in enumerate(self.weight_out):
            for j,val in enumerate(row):
                self.weight_out[i][j]=getRandom(-0.2,0.2)

        # 存储数据的矩阵
        self.in_matrix=[1.0]*self.n_in
        self.hidden_matrix=[1.0]*self.n_hidden
        self.out_matrix=[1.0]*self.n_out

        # 设置动量矩阵
        # 保存上一次梯度下降方向
        self.ci=makeMatrix(self.n_in,self.n_hidden)
        self.co=makeMatrix(self.n_hidden,self.n_out)


    # step 3. 正向传播
    # 根据传播规则对节点值进行更新
    def update(self,inputs):
        if len(inputs)!=self.n_in-1:
            raise ValueError("Your data length is %d, but our input needs %d"%(len(inputs),self.n_in-1))
        # 设置初始值
        self.in_matrix[:-1]=inputs
        # 注意我们最后一个节点依旧是1，表示偏置节点

        # 隐层
        for i in range(self.n_hidden-1):
            accumulate=0
            for j in range(self.n_in-1):
                accumulate+=self.in_matrix[j]*self.weight_hidden[j][i]
            self.hidden_matrix[i]=sigmoid(accumulate)

        # 输出层
        for i in range(self.n_out):
            accumulate = 0
            for j in range(self.n_hidden - 1):
                accumulate += self.hidden_matrix[j] * self.weight_out[j][i]
            self.out_matrix[i] = sigmoid(accumulate)

        return self.out_matrix[:] # 返回一个副本

    # step 4. 误差反向传播
    def backpropagate(self,target):
        if len(target) != self.n_out :
            raise ValueError("Your data length is %d, but our input needs %d" % (len(target), self.n_out))
        # 计算输出层的误差
        # 根据公式： Err=O(1-O)(T-O)=(O-O**2)(True-O)
        out_err=[derived_sigmiod(o:=self.out_matrix[i])*(t-o) for i,t in enumerate(target)]
        # 计算隐层的误差
        # 根据公式：Err=(O-O**2)Sum(Err*W)
        hidden_err=[0.0]*self.n_hidden
        # for i in range(self.n_hidden):
        #     err_tot=0.0
        #     for j in range(self.n_out):
        #         err_tot+=out_err[j]*self.weight_out[i][j]
        #
        #     hidden_err[i]=derived_sigmiod(self.hidden_matrix[i])*err_tot
        hidden_err=[derived_sigmiod(self.hidden_matrix[i])*sum(out_err[j]*self.weight_out[i][j] for j in range(self.n_out)) for i in range(self.n_hidden) ]

        # 更新权重
        # 输出层：
        # w=bias+lr*O*Err+m*(w(n-1))
        # m表示动量因子，w(n-1)是上一次的梯度下降方向
        for i in range(self.n_hidden):
            for j in range(self.n_out):
                # 更新变化量 change=O*Err
                change=self.hidden_matrix[i]*out_err[j]
                self.weight_out[i][j]+=self.lr*change+self.m*self.co[i][j]
                # 更新上一次的梯度
                self.co[i][j]=change

        # 隐含层
        for i in range(self.n_in):
            for j in range(self.n_hidden):
                change=hidden_err[j]*self.in_matrix[i]
                self.weight_hidden[i][j]+=self.lr*change+self.m*self.ci[i][j]
                self.ci[i][j]=change

        # 计算总误差
        err=0.0
        for i,v in enumerate(target):
            err+=(v-self.out_matrix[i])**2
        err/=len(target)
        return math.sqrt(err)

    def train(self,data,epochs=1000):
        best_err=1e10
        for i in range(epochs):
            err=0.0
            for j in data:
                x=j[0]
                y=j[1]

                self.update(x)
                err+=self.backpropagate(y)
            if err<best_err:
                best_err=err
        print(best_err)

    def fit(self,x):
        return [self.update(i) for i in x]



net=BPNN(5,1)
def getData(m,n,c=None):
    # 随机生成一组大小为m*n,类别为c的数据
    if c!=None:
        data=[[[random.uniform(0.0,2.0)]*n,[random.randint(0,c)]] for i in range(m)]
    else:
        data=[[random.uniform(0.0,2.0)]*n for _ in range(m)]
    return data
d_train=getData(20,5,2)
d_test=getData(10,5,2)

# 固定模式
d=[
    [[1,0,1,0,1],[1]],
    [[1,0,1,0,1],[1]],
    [[1,0,1,0,1],[1]],
    [[1,0,1,1,1],[0]],
    [[1,0,1,0,1],[1]],
    [[1,0,1,1,1],[0]],
]
c=[
    [1,0,1,0,1],
    [1,0,1,0,1],
    [1,0,1,1,1],
    [1,0,1,0,1],
    [1,0,1,1,1],
    [1,0,1,0,1],
    [1,0,1,0,1],
    [1,0,1,1,1],
    [1,0,1,0,1],
    [1,1,1,0,1],
]

net.train(d)
print(net.fit(c))




