#2020-3-15  回顾pytorch知识


#
'''
torch.form_numpy()  numpy转换为tensor格式
x = torch.randn((3,4),requires_grad)  //requires_grad 是tensor的属性，默认为False，如果设置为True，将tensor加入计算图，用来追踪tensor的变化，所有依赖他的节点的requires_grad都为True

a = torch.Tensor(1,1,3)
b = a.squeeze(0)//对tensor a的第一个维度进行缩减，如果该维度为1，则取消该维度变成（1,3）
c = a.unsqueeze(3)//对tensor的第四个维度进行扩充，扩充指定位置的维度，变为(1,1,3,1)
a.permute(1, 0, 2) //对tensor进行维度变换1,0,2分别代表原来的维度变换到第1,0,2

torch.manual_seed(1)//一个矩阵初始化后，再次初始化认为相同的矩阵
检查gpu_torch是否可以使用，并设置为GPU版本
if torch.cuda.is_available：
    device = torch.device("cuda")


model.parameter()里面的保存的是反向传播需要被optimizer更新的参数

x = x.to(device)   x从一个普通的tensor变成一个gpu版本的tensor

x = x.cpu().data.numpy()   x从gpu的tensor转换为numpy形式的数组

x = np.random.randn(3,3)    随机创建一些数据

x = torch.randn(3,3)    随机创建一些torch数据
x.mm(y) x 和 y是torch类型的两个相乘

x.dot(y)  表示矩阵x乘上矩阵y


optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)    优化函数 ，第一个参数告诉模型应该优化哪些权重

torch.nn.linear(in_features,out_features,bias=True)    in_features是每个输入样本的大小，其维度等于输入的第二个维度。out_features是每个输出样本的大小    


========================一般的神经网络定义步骤===================================================

<1> 首先定义一个模型：
class mymodel(torch.nn.Model):
    def __init__(self,神经网络的参数):
        super(mymodel,self).__init__()   初始化父类的参数
        
        <2>定义模型的结构	//实例化nn.Linear等，并将他们作为成员变量
        self.linear = torch.nn.Linear(in_features,out_features,bias=True)
        self.embed = torch.nn.Embedding(vocab_size,embed_size) 
        self.lstm = torch.nn.lstm(input_size,output_size,num_layers,bias,batch_first,droup_out,bidirecter)

        <3>模型中的前向传播  //forward函数可以使用其他模块或者其他的自动求导运算来接收输入tensor，并产生输出tensor
        def forward(self,x):
            y_pred = self.linear(...)
        
        <>定义hidden，因为不确定是在gpu还是CPU上跑，所以创建一个和参数类型相同的hidden——state,并且初始化为0
        def ini_hidden(self,bsz,requires_grad=True):
            weight = next(self.parameters())
            return weight.new_zeros((1,bsz,self.hidden_size,requires_grad=True)
    
        
<4>实例化模型
model = mymodel(模型参数)

<5>定义损失函数
loss_fn = nn.Adam()
<6>定义优化函数
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
当损失函数降低很小时，可以加入学习率动态调节机制
scheduler = torch.optim.lr_scheduler.ExponentialLr(optimizer,0.5)   #学习率减少为之前的一半.
val_losses = []


<7>开始循环训练网络
for i in range():
    <8>前向传播
    y_pred = model(x)    #执行model.forward()
    
    <9>计算损失
    loss = loss_fn(y_pred,y)
    print(i,loss.item())
    
    <10>清空梯度，//在方向传播之前，使用optimizer将他要更新的所有张量(需要学习的权重参数)的梯度清零，以免梯度累加
    optimizer.zero_grad()
    
    <11>反向传播  //根据模型的参数计算loss的梯度
    loss.backward()
    
    <12>更新模型中的参数    //调用optimizer的step函数使他所有的参数更新
    optimizer.step()
    
    <13>保存模型参数
    if i % 10000:
    val_loss = evaluate(model,val_iter)
    如果当前模型比之前模型的损失要小，就保存这个模型
    if len(val_losses) == 0 or val_loss < min(val_losses):
        torch.save(model.state_dict(),"im.path")
        print("best model saved to lm.path")
    else：
        减少学习率
        scheduler.step()
    val_losses.append(val_loss) #将新的val_loss加入列表之中
    
    
<14>加载一个已经保存的模型
best_model = mymodel(参数）
if USE_CUDA:
    model = model.to(device)
best_model.load_state_dict(torch.load("lm.path"))


<15>lstm中使用训练好的模型去运行代码实例化,例子是语言模型的训练
hidden = best_model.init_hidden(1)  #取模型训练好的hidden
input = torch.randint(VOCAB_SIZE，(1,1),dtype=torch.long).to(device)
words = []
for i in range():
    output,hidden = best_model(input,hidden)
    
===============================================================================    
torch.manual_seed(1)   每次初始化的数据是一样的      
第一个参数是单词的数量，第二个是每个单词的词嵌入维度。
一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。
torch.nn.Embedding(vocab_size,embed_size)  


input_size:
    输入特征维数
hidden_size:
    隐层状态的维数
num_layers:
    RNN层的个数，在图中竖向的是层数，横向的是seq_len
bias:
    隐层状态是否带bias，默认为true
batch_first:
    是否输入输出的第一维为batch_size，因为pytorch中batch_size维度默认是第二维度，故此选项可以将 batch_size放在第一维度。如input是(4,1,5)，中间的1是batch_size，指定batch_first=True后就是(1,4,5)
dropout:
    是否在除最后一个RNN层外的RNN层后面加dropout层
bidirectional:
    是否是双向RNN，默认为false，若为true，则num_directions=2，否则为1
out,hidden_size = torch.nn.lstm(input_size,output_size,num_layers,bias,batch_first,droup_out,bidirecter)

比如：lstm = nn.LSTM(10, 20, 2)
      x = torch.randn(5, 3, 10)
      h0 = torch.randn(2, 3, 20)
      c0 = torch.randn(2, 3, 20)
      output, (hn, cn)=lstm(x, (h0, c0))
输出：
      output.shape  torch.Size([5, 3, 20])
      hn.shape  torch.Size([2, 3, 20])
      cn.shape  torch.Size([2, 3, 20])



模型每1万步保存一步
if i % 10000:
    torch.save(model.state_dict(),"im.path")
    
英文分词可以用nltk
import nltk
===========================词表的构建==================================================
利用python的Counter类实现对词表的构建
from collections import Counter
UNK_IDX = 0
PAD_IDX = 1
word_count = Counter()  
def build_dict(sentences,max_words=300000):
    for sentence in sentences:
        for i in sentences:
            word_count[i] +=1   #统计词频
    ls = word_count.most_common(max_words)   输出的格式为[("单词",出现次数)。。。]

    total_words = len(ls) + 2   #词表内单词总数
    
    按照词频统计排序，构造词典，词频越大的单词索引越小
    word_dict = {w[0]:index+2 for index,w in enumerate(ls)}

    word_dict["UNK"] = UNK_IDX
    word_dict["PAD"] = PAD_IDX
    
    return word_dict total_words  返回制作好的词表以及词表中的单词总数

最后word_dict字典的格式为：{"单词"：2,... 'UNK': 0, 'PAD': 1}

#索引->单词
inv_word_dict = {v:k for v,k in word_dict.items()}
print(inv_word_dict[2],inv_word_dict[6])

将数据集中的句子级数据转换成对应字典中的数字

a_sentences = [[word_dict.get(w,0) for w in sentence] for sentence in sentences]

将句子长度按照长度进行排序，保证长度相近的句子放在一起
def len_argsort(seq):
    return sorted(range(len(seq)),key=lambda x: len(seq[x]))

sorted_index = len_argsort(sentences_1)
sentences_2 = [sentences_1[i] for i in sorted_index]

需要掌握如何将数据切分成batch
==============================================================================

torch.cat(a,b)    把两个矩阵拼接在一起
'''