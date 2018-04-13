# decisionTree #
## 功能 ##
> 在Mushroom Data Set上实现C4.5算法  

## 数据集 ##
> 数据集：UCI-Mushroom Data Set  
> * Number of Instance: 8124  
> * Number of Attributes: 22  
> * Number of Labels: 2  

## 算法步骤 ##
> 1. 读取数据，得到训练集和测试集  
> 2. 计算得到所有的类别、所有属性所对应的值  
> 3. 根据信息增益比构建数据集的决策树  
> 4. 在测试集上测试决策树的准确率  

## 结果 ##
> 在测试集上得到82.1165%的准确率  

## API ##
`void initLabels(string& labels, vector<string> dataset)`:  
> 初始化labels  
> @param: string& labels: 包含所有类别的string变量  
> @param: vector<string> dataset: 所有数据  
> @return:  

`void initValues(vector<string>& values, vector<string> attributes, vector<string> dataset)`:  
> 初始化values  
> @param: vector<string>& values: 包含所有属性对应值的vector容器  
> @param: vector<string> attributes: 包含所有属性的vector容器  
> @param: vector<string> dataset: 所有数据  
> @return:  

`void initMap(map<string, string>& m, vector<string> attributes, vector<string> values)`:  
> 初始化attributes和values的map容器  
> @param: map<string, string>& m: 属性与对应值的map容器  
> @param: vector<string> attributes: 包含所有属性的vector容器  
> @param: vector<string> values: 包含所有属性对应值的vector容器  
> @return:   

`double computeEntropy(vector<string> dataset)`:  
> 计算对应数据集的信息熵  
> @param: vector<string>: 计算信息熵的数据集  
> @return: 计算得到的信息熵  

`double computeFeatureEntropy(vector<string> dataset, int feature)`:  
> 计算给定特征的熵  
> @param: vector<string> dataset: 计算信息熵的数据集  
> @param: int feature: 用于分类的给定特征  
> @return: 计算得到的特征熵  

`vector<string> splitDataset(vector<string> dataset, int axis, char value)`:  
> 根据给定属性及其对应值对数据集分割  
> @param: vector<string> dataset: 进行分割的数据集  
> @param: int axis: 用于分割的属性值对应的索引值  
> @param: char value: 属性对应的某个值  
> @return: 分割后的数据集  

`int chooseBestFeature(vector<string> dataset)`:  
> 根据给定数据集选定最佳属性  
> @param: vector<string> dataset: 进行属性选择的数据集  
> @return: 根据信息增益比选择出的最佳属性  

`bool allTheSameLabel(vector<string> dataset)`:  
> 判断数据集是否属于同一类别  
> @param: vector<string> dataset: 用于判断的数据集  
> @return: 属于同一类返回true，否则返回false  

`char mostCommonLabel(vector<string> dataset)`:  
> 计算数据集中样本最多的类别  
> @param: vector<string> dataset: 用于计算的数据集  
> @return: 数据集中样本最多的类别  

`Node* createDecisionTree(Node* parent, vector<string> dataset, vector<string> attributes)`:  
> 创建决策树  
> @param: Node* parent: 父节点  
> @param: vector<string> dataset: 用于创建节点的数据集  
> @param: vector<string> attributes: 还未利用的属性值  
> @return: 决策树的根节点  

`char classify(Node* root, string item)`:  
> 对测试样本分类  
> @param: Node* root: 决策树根节点  
> @param: string item: 测试样本  
> @return: 测试样本的类别  

`pair<vector<string>, vector<string>> read(string file, double ratio)`:  
> 读取数据，分为训练集和测试集  
> @param: string file: 读取数据的文件  
> @param: double ratio: 训练集占总数据的比例  
> @return: 训练集和测试集构成的pair变量  