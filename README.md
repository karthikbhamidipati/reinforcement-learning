## Overview

Implement reinforcement learning algorithm to find policies for frozenlake environment.

Reinforcement learning algorithms implemented:
1. Model based tabular algorithms:
   * Policy iteration
   * Value iteration
   * Policy improvement
2. Model free tabular algorithms:
   * SARSA control
   * Q-learning control
3. Model free non-tabular algorithms:
   * Linear approximation with SARSA control
   * Linear approximation with Q-learning control
   
## Requirements

Unzip the folder and run ```pip install -r requirements.txt``` to install the required dependencies.         
* python version should be 3.7 or above
* numpy version should be 1.19.2 or above
* matplotlib version should be 3.3.2 or above


## Execution
      
   * Execute the ```main_implementation.py``` file to run the reinforcement learning algorithms on the environments.
   * Execute the ```env\run_env.py``` file to manually run the environments.

## Explanation
       
   * main_implementation displays the policy and values for the small frozenlake, comment out line 135 if you want to execute only for small frozenlake
   * big_implementation displays the policy and values for the big frozenlake, comment out line 134 if you want to execute only for big frozenlake
