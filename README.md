# Paper
MRCA: Metric-level Root Cause Analysis for Microservices via
Multi-Modal Data

### Operating environment
* PyTorch version: 1.13.1

### Dataset
You can loaddown: 
* The raw data: https://cuhko365-my.sharepoint.com/:f:/g/personal/223040053_link_cuhk_edu_cn/Enk7u1UdrLNBsCuDuVfeRpYBYCGs-mfQZAk8vVg5F8Cisw?e=ffnK8s
* The processed data: https://cuhko365-my.sharepoint.com/:f:/g/personal/223040053_link_cuhk_edu_cn/EsZRFG_39ItArg-q9gyJFUwB-ofuAFu8LQj1DtEEDOVnSA?e=mGZwDI
* AD results: https://cuhko365-my.sharepoint.com/:f:/g/personal/223040053_link_cuhk_edu_cn/Euq7B-g812NKuFra71XOe64BxmNgD5u_P8TYAOjGF7qsCA?e=Ud2Oo1
Put them in the main directory.

### Run command
Feature learning for trace and log.
```
trace_processing.py
log_processing.py
```
Anomaly detection.
```
anomaly_detection.py
```
Root cause localization
```
root_cause_localization.py
```
Evaluation MRCA
```
evaluation_at_service-level.py
evaluation_at_metric-level.py
```
