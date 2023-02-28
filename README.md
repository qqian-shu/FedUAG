# FedUAG
This experiment is based on the scenario of overlapping decentralized nodes. It not only uses secure multi-party computation to obtain global nodes adjacency relationship, but also uses differential privacy to perturb model parameters. There are six clients of this experiment, and the client communicaiton was through socketIO-client. 

# Dataset
The graph datasets used in this study, including Cora,Citeseer, and PubMed, are shown in Table 1.  
| Dataset  | Nodes | Edges | Features | classes |
|----------|-------|-------|----------|---------|
| Cora     | 2708  | 5429  | 1433     | 7       |
| Citeseer | 3327  | 4327  | 3703     | 6       |
| Pubmed   | 19717 | 44324 | 500      | 3       |


# Experimental environment
* numpy
* matplotlib
* pandas
* flask
* flask_socketio
* tensorflow

# Acknowledgement
This work was sponsored by the National Key Research and Development Program of China (No. 2018YFB0704400), Key Program of Science and Technology of Yunnan Province (No. 202002AB080001-2, 202102AB080019-3), Key Research Project of Zhejiang Laboratory (No.2021PE0AC02), Key Project of Shanghai Zhangjiang National Independent Innovation Demonstration Zone(No. ZJ2021-ZD-006). The authors gratefully appreciate the anonymous reviewers for their valuable comments.
