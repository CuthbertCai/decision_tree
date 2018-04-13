/*
    决策树C4.5算法的实现
    数据集：UCI-Mushroom Data Set
           Number of Instance: 8124
           Number of Attributes: 22
           Number of Labels: 2
    作者： 蔡冠羽
*/
#include "iostream"
#include "vector"
#include "fstream"
#include "cassert"
#include "string"
#include "map"
#include "cmath"
#include "algorithm"
#include "set"
#define MAX_LEN_PER_LINE 23
#define BLANK  "";

using namespace std;

map<string, string> map_attribute_values;
// 属性值集合
const vector <string> attributes{"cap-shape", "cap-surface", "cap-color", "bruises", "odor", 
"gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
"stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
"stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
"ring-number", "ring-type", "spore-print-color", "population", "habitat"};
vector<string> values;
string labels;

// 定义决策树节点的数据结构
struct Node{
    string attribute;
    string values;
    char label;
    vector<Node *> childs;
    Node(){
        attribute = BLANK;
        values = BLANK;
        label = ' ';
    }
};
// 声明根节点
Node* root;

// 初始化labels
void initLabels(string& labels, vector<string> dataset){
    string _labels;
    for(auto item:dataset){
        _labels.push_back(item[0]);
    }
    set<char> labelSet(_labels.begin(), _labels.end());
    for(auto label:labelSet){
        labels.push_back(label);
    }
}

// 初始化values的vector容器
void initValues(vector<string>& values, vector<string> attributes, vector<string> dataset){
    for(int i = 0; i < attributes.size(); i++){
        string value;
        for(int j = 0; j < dataset.size(); j++){
            value.push_back(dataset[j][i+1]);
        }
        set<char> valueSet(value.begin(), value.end());
        string temp;
        for(auto value:valueSet){
            temp.push_back(value);
        }
        values.push_back(temp);
    }
}

// 初始化attributes和values的map容器
void initMap(map<string, string>& m, vector<string> attributes, vector<string> values){
    // cout << attributes.size() << values.size() << endl;
    if(attributes.size() != values.size())
        cout << "Number of instance is wrong!" << endl;
    for(int i = 0; i < attributes.size(); i++){
        m[attributes[i]] = values[i];
    }
}

// 计算信息增益
double computeEntropy(vector<string> dataset){
    unsigned int numItem = dataset.size();
    map<char, unsigned int> labelCount;
    for(auto item = dataset.begin(); item != dataset.end(); item++){
        char label = (*item)[0];
        if(labelCount.count(label) == 0) labelCount[label] = 0;
        labelCount[label] += 1;
    }
    double entropy = 0.0;
    for(int i = 0; i < labelCount.size(); i++){
        double prob = double(labelCount[labels[i]]) / numItem;
        if(prob==0) continue;
        entropy -= prob * log2(prob);
    }
    return entropy;
}

// 计算给定特征feature的熵
double computeFeatureEntropy(vector<string> dataset, int feature){
    unsigned int numItem = dataset.size();
    map<char, int> featureCount;
    for(auto item:dataset){
        char value = item[feature];
        if(featureCount.count(value)==0) featureCount[value] = 0;
        featureCount[value]++;
    }
    double featureEntropy = 0.0;
    for(auto item:featureCount){
        double prob = double(item.second) / numItem;
        if(prob==0) continue;
        featureEntropy -= prob * log2(prob);
    }
    return featureEntropy;
}

// 根据给定特征将数据集分割
vector<string> splitDataset(vector<string> dataset, int axis, char value){
    vector<string> retDataset;
    for(auto item = dataset.begin(); item != dataset.end(); item ++){
        if((*item)[axis] == value){
            string temp = (*item).substr(0, axis);
            temp += (*item).substr(axis+1, (*item).size()-axis-1);
            retDataset.push_back(temp);
        }
    }
    return retDataset;
}

// 根据数据集选取信心增益比最大的特征
int chooseBestFeature(vector<string> dataset){
    unsigned int numFeatures = dataset[0].size();
    int bestFeature = -1;
    double baseEntropy = computeEntropy(dataset), bestGain = 0.0;
    for(int i = 1; i < numFeatures; i++){
        vector<char> feaVec;
        for(auto item = dataset.begin(); item != dataset.end(); item++){
            feaVec.push_back((*item)[i]);
        }
        set<char> uniqueVals(feaVec.begin(), feaVec.end());
        double newEntropy = 0.0;
        for(auto value = uniqueVals.begin(); value != uniqueVals.end(); value++){
            vector<string> subDataset = splitDataset(dataset, i, *value);
            double prob = (double)subDataset.size() / dataset.size();
            newEntropy += prob * computeEntropy(subDataset);
        }
        double infoGain = (baseEntropy - newEntropy)/computeFeatureEntropy(dataset, i);
        if(infoGain > bestGain){
            bestGain = infoGain;
            bestFeature = i;
        }
    }
    return bestFeature;
}

// 判断数据集是否全部属于同一类别
bool allTheSameLabel(vector<string> dataset){
    for(auto item = dataset.begin()+1; item != dataset.end(); item++){
        if((*item)[0] != dataset[0][0]) return false;
        else continue;
    }
    return true;
}

bool cmpWithValue(const pair<char, int>& a, const pair<char, int>& b){
    return a.second > b.second;
}

// 计算数据集中样本最多的类别
char mostCommonLabel(vector<string> dataset){
    map<char, int> labelCount;
    for(auto item = dataset.begin(); item != dataset.end(); item++){
        char label = (*item)[0];
        if(labelCount.count(label) == 0) labelCount[label] = 0;
        labelCount[label] += 1;
    }
    vector<pair<char, int>> labelCountVec(labelCount.begin(), labelCount.end());
    sort(labelCountVec.begin(), labelCountVec.end(), cmpWithValue);
    return labelCountVec[0].first;
}

// 创建决策树
Node* createDecisionTree(Node* parent, vector<string> dataset, vector<string> attributes){
    if(parent == NULL) parent = new Node();
    if(allTheSameLabel(dataset)){
        parent->label = dataset[0][0];
        return parent;
    }
    if(attributes.size()==0){
        parent->label = mostCommonLabel(dataset);
        return parent;
    }
    int bestFeature = chooseBestFeature(dataset);
    string bestAttribute = attributes[bestFeature];
    string bestValues = map_attribute_values[bestAttribute];
    attributes.erase(attributes.begin()+bestFeature);
    parent->attribute = bestAttribute;
    for(auto value : bestValues){
        vector<string> subAttributes = attributes;
        Node* child = new Node();
        vector<string> subDataSet = splitDataset(dataset, bestFeature, value);
        if(subDataSet.size()==0) continue;
        child = createDecisionTree(child, subDataSet, subAttributes);
        parent->values.push_back(value);
        parent->childs.push_back(child);
    }
    return parent;
}

// 根据决策树和测试样本进行分类
char classify(Node* root, string item){
    if(root->label != ' ') return root->label;
    for(int i = 0; i < attributes.size(); i++){
        if(root->attribute == attributes[i]){
            for(int j = 0; j < root->values.size(); j++){
                if(item[i+1] == root->values[j]){
                    if(root->childs[j]->label != ' ') return root->childs[j]->label;
                    return classify(root->childs[j], item);
                }
            }
            cout << "Cannot find value!" << endl;
            return ' ';
        }
    } 
    cerr << "Cannot find attribute." << endl;
    return ' ';
}

// 读取数据，并将数据分为训练集和测试
pair<vector<string>, vector<string>> read(string file, double ratio){
    ifstream infile;
    infile.open(file.data());
    assert(infile.is_open());

    vector <string> dataset;
    string temp;
    while(getline(infile, temp)){
        string store;
        for(auto c = temp.begin(); c != temp.end(); c++){
            if(*c != ',')
                store.push_back(*c);
        }
        assert(store.size() == MAX_LEN_PER_LINE);
        dataset.push_back(store);
    }
    infile.close();
    int index = (int)dataset.size() * ratio;
    vector<string> trainData, testData;
    for(int i = 0; i < dataset.size(); i++){
        if(i < index) trainData.push_back(dataset[i]);
        else testData.push_back(dataset[i]);
    }
    return pair<vector<string>, vector<string>>(trainData, testData);
}

int main(){
    vector<string> dataset = read("data.txt", 1).first;
    vector<string> trainData = read("data.txt", 0.7).first;
    vector<string> testData = read("data.txt", 0.7).second;
    initLabels(labels, dataset);
    initValues(values, attributes, dataset);
    initMap(map_attribute_values, attributes, values);
    root = createDecisionTree(root, trainData, attributes);
    int count = 0;
    for(auto item:testData){
        char ressult = classify(root, item);
        if(ressult == item[0]) count++;
    }
    double accuracy = (double)count/testData.size();
    cout << "Accuracy of test data is: " << accuracy << endl;
    return 0;
}