# An Imperative Leaning (IL) control-variate approach for MTSP
iMTSP is a novel self-supervised framework to solve the multiply traveling salesmen problem (MTSP). By introducing a surrogate network as a control variate, iMTSP can efficiently train the allocation network through the non-differentiable TSP solver and the discrete decision space.
<img src='imgs/iMTSP_framework.pdf' width=500>
##Experimental results
The figures demonstrate the advantages of iMTSP on two specific MTSP instances. We compare iMTSP with a reinforcement learning(RL)-based approach and Google OR-Tools routes module.
<img src='imgs/routes.png' width=500>
We also explicitly record the history of gradient variance of iMTSP and the RL baseline. Our method converges 20 times faster than the baseline with the help of the surrogate network.
<img src='imgs/var_hist.pdf' width=500>
## Paper
Our paper "iMTSP: Solving Min-Max Multiple Traveling Salesman Problem with Imperative Learning" has been submitted to 2024 International Conference on Intelligent Robots and Systems (IROS) and you can find a pdf [here](https://github.com/sair-lab/iMTSP/files/14891988/iMTSP.1.pdf).
## To get start
### Dependencies
python == 3.8.15

PyTorch == 1.13.1

torch-geometric == 2.2.0

wandb == 0.15.2

ortools == 9.5.2237
### Generate data for training, validation, and testing
Open data_generator.py, and modify the flag (training, validation, or testing), the number of nodes, and the batch size as needed. Run the script to generate data.
## Training
The validation process is embedded in the training files. 

train.py contains the learning-based baseline.

my_train.py contains the proposed approach.
## Testing
test.py

## More information
Trained models: Please refer to [here](https://github.com/sair-lab/iMTSP/releases/tag/v1.0) for trained allocation networks and corresponding surrogate networks.
