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
    * SARSA control with Linear function approximation
    * Q-learning with Linear function approximation

## Requirements

Unzip the folder, `cd` onto the project directory and run ```pip install -r requirements.txt``` from terminal to install
the required dependencies.

* python version should be 3.7 or above
* numpy version should be 1.19.2 or above
* matplotlib version should be 3.3.2 or above

## Execution

* Run ```python main_implementation.py``` file to run the reinforcement learning algorithms on the environments.
* Run ```python run_env.py``` file to manually run the small frozenlake environment.

## Additional Information

### Main implementation

* `small_lake_implementation()` method in `main_implementation.py` displays the policy and values for the small
  frozenlake, comment out line 135 if you want to execute only for small frozenlake.
* `big_lake_implementation()` method in `main_implementation.py` displays the policy and values for the big frozenlake,
  comment out line 136 if you want to execute only for big frozenlake.

### Run environment

* ```run_env.py``` runs the small frozenlake environment by default.
* To manually run the big frozenlake environment, comment line 85 and uncomment line 86

### Data Collector

* Data collector is used to visualize the convergence of the algorithms.
* It computes mean squared error for value when compared to optimal value.
* It computes the classification error for policy when compared to optimal policy.
* It has been commented out as it affects performance. Follow the below instructions to use data collector.

```
Add below to the algorithm to store the error for visualization.
    
    DataCollectorSingleton.instance().calculate_error("Algorithm name", policy, value)
   
Add the below to the main_implementation.py to visualize the errors

before running algorithms:

    DataCollectorSingleton.instance().set_optimal_policy_value("env name", "env optimal policy value path <.npy file>")
    
after running algorithms:

    plot_errors(*DataCollectorSingleton.instance().get_errors()) 
```

### Contributors

- [Omer Faruk Yildirim](https://github.com/farukyld)
- [Karthik Bhamidipati](https://github.com/vamshikarthik)
- [Sudipta Mondal](https://github.com/sudiptamondal1802)
