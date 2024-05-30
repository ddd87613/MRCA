# 多源数据根本原因定位原型代码说明

### 项目说明
多源数据根本原因定位的算法研究，此原型代码说明了运行环境以及运行命令等。

### 运行环境
* PyTorch version: 1.13.1

### 数据集介绍
'Data'中包含的'normal data'和'normal log'是系统正常运行时生成的数据，'suffer anomaly inject data'和'anomaly log'是注入异常后收集到的数据（包括trace，log，metric以及异常注入时间和位置等信息）
'anomaly log classification'是按照服务分类log得到的结果
'initial ranking by VAE'是进过VAE之后得到的初始排名

### 运行命令
根本原因定位
```
Casual analysis.py
```
