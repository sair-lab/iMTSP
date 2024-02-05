# A Imperative Leaning (IL) control-variate approach for MTSP

# To get start
## Install the dependencies
python == 3.8.15
PyTorch == 1.13.1
torch-geometric == 2.2.0
wandb == 0.15.2
ortools == 9.5.2237
## Generate data for training, validation, and testing
Open data_generator.py, and modify the flag (training, validation, or testing), the number of nodes, and the batch size as needed. Run the script to generate data.
# Training
The validation process is embedded in the training files. 
train.py contains the learning-based baseline.
my_train.py contains the proposed approach.
# Testing
test.py
