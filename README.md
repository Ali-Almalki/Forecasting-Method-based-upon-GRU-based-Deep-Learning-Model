## World-Model

In this World Model, we specifically modified the RNN model by replacing the LSTM model with Bidirectional GRU. A typical advantage of using GRU is that it has less training parameters as compared to LSTM which eventually trains faster.

### 1- Model description and training steps
Input to the system is an observation 64×64 pixels color image of a car track environment depicted in Figure 1 and output of the system is the set of actions: 1. Steering direction (-1,1), 2. Acceleration (0,1) and 3. Brake (0,1).

### 2- Training Variational Autoencoder (VAE)
VAE model takes the input color image of size 64×64×3 and generates the 32-dimensional vector (z). VAE is based on convolutional neural network (CNN) model. Combination of reconstruction and KL loss has been used in training the model. The first step is to generate the data (rollouts) for VAE training using the following command:

    python 01_generate_data.py car_racing --total_episodes 2000 --time_steps 300
This will produce 2000 rollouts (saved in ten batches of 200), starting with batch number 0. Each rollout will be a maximum of 300 time-steps long.
Now train the controller using following command:

    python 02_train_vae.py --N 500 --time_steps 300 --epochs 20 --new_model

#### This figure presents the training loss for VAE and the table presents all the three losses.

![alt text](https://github.com/Ali-Almalki/World-Model/blob/master/Images/vae_losses.PNG)

#### Training Loss for VAE
![alt text](https://github.com/Ali-Almalki/World-Model/blob/master/Images/Training_loss_for_VAE.PNG)

### 3- Training RNN (Bidirectional GRU)
In this work LSTM model has been replaced with Bidirectional GRU. Input to RNN is compressed vector (z_t) from VAE and actions (a_t) and it output next time-step vector (z_(t+1)). The first step is to generate the data for RNN training using following command:

    python 03_generate_rnn_data.py --N 10000
    
Now run the RNN training:

    python 04_train_rnn.py --N 10000 --steps 1000 --batch_size 100 --model_type GRU --new_model
    
RNN has been trained for one epoch and 1000 steps. The follwoing tble presents the RNN loss after training.
#### RNN training loss for one epoch and 1000 steps.

![alt text](https://github.com/Ali-Almalki/World-Model/blob/master/Images/RNN%20training%20loss.PNG)

### 4- Training Controller
It is trained with the evolutionary algorithm known as Covariance Matrix Adaptation-Evolution Strategy (CMA-ES). This algorithm creates the copies of the parameters of this model with multiple random initializations (population), test each population inside environment and record the average score. Controller is trained with following command:

     python 05_train_controller.py car_racing --num_worker 16 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25

This figure shows the average score of the controller at each generation of the training. We train the controller for 100 generations.
![alt text](https://github.com/Ali-Almalki/World-Model/blob/master/Images/Controller.png)

Training output of controller with scores is depicted in Table. Each best score is computed after 25 evaluations.
![alt text](https://github.com/Ali-Almalki/World-Model/blob/master/Images/Controller%20scores.png)

### 4- Results
The follwoing table presents the comparison of various models on CarRacing-v0 environment. Compared to the Full world model our model achieved a reasonable score in about 26h with smaller dataset.
2000 rollouts. However, full world model works on 10,000 rollouts and take many days to achieve the score of 906. Table 5 finally presents the loss comparison of LSTM with Bidirectional GRU. 
#### CarRacing-v0 scores using various models
![alt text](https://github.com/Ali-Almalki/World-Model/blob/master/Images/CarRacing%20Scores.PNG)

#### Comparison of RNN models: LSTM vs. Bidirectional GRU.
![alt text](https://github.com/Ali-Almalki/World-Model/blob/master/Images/Comparison.PNG)




