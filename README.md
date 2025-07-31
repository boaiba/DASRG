# A Dual Asynchronous-synchronous Relation Graph Method for Sensor-based Group Activity Recognition

>Ruohong Huan, Ai Bo, Ke Wang, Peng Chen*, Ronghua Liang

<div style="text-align: justify;">
Group activity recognition (GAR) aims to understand complex activities composed of multiple individual actions. However, existing methods primarily focus on recognizing synchronous actions and exhibit significant limitations in handling asynchronous actions, which are prevalent in real-world scenarios. In asynchronous actions, temporal-delayed individuals perform or respond to actions with temporal lags that are not synchronized with those of real-time individuals, forming complex temporal-delayed relations. To address this issue, this paper proposes a dual asynchronous-synchronous relation graph method (DASRG) for sensor-based GAR. By introducing an asynchronous relation graph, the temporal-delayed relations between the temporal-delayed individuals and real-time individuals are captured, and the features of temporal-delayed individuals are synchronized with those of the real-time individuals through feature shifting. Meanwhile, a synchronous relation graph is constructed to enhance the modeling of interactions and action differences between individuals by integrating spatial interaction relation features and residual relation features, thereby reducing the interference of non-critical individuals on recognition results. To achieve the coupling between asynchronous and synchronous relation graphs, a graph aggregation module based on the multi-head attention mechanism is designed. By leveraging weighted feedback from attention heads, this module optimizes node information updating and enables the mutual feedback of information between the two graphs in the spatial domain, improving the model’s ability to handle asynchronous actions and mitigate interference from non-critical individuals. Experimental results on two self-constructed datasets, UT-Data-gar and Garsensors, demonstrate the superiority of the proposed method in complex scenarios involving asynchronous actions and interfering individuals.
</div>


## Installation

```shell
# Install PyTorch
$ pip install torch==1.8.0 torchvision==0.9.0

# Install other python libraries
$ pip install -r requirements.txt
```



## Requirements

- python>=3.6,<3.10
- torch==1.8.0
- torchvision==0.9.0
- numpy>=1.19.0
- pandas>=1.3.0
- scipy>=1.7.0
- scikit-learn>=0.24.0
- matplotlib>=3.3.0
- seaborn>=0.11.0



## Prepare Datasets

Download publicly available datasets from following links:

-  [UT-Data-gar dataset](https://www.dropbox.com/scl/fo/vc8o6ciawhhwin687umxt/AH8bSMjamLFxxtkN6rE0fPs?rlkey=v8euihfog6wnrw0ozlj2se6ok&st=7amtmm63&dl=0)
- [Garsensors dataset](https://www.dropbox.com/scl/fo/vc8o6ciawhhwin687umxt/AH8bSMjamLFxxtkN6rE0fPs?rlkey=v8euihfog6wnrw0ozlj2se6ok&st=7amtmm63&dl=0)

