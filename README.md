Reinforcement Learning for Profiled Side-Channel Analysis
========
The code for the experiments generating SCA CNNs can be found in the `metaqnn` folder.
The definitions for the experiments are in the `models` folder.
To run an experiment, use `python -m metaqnn.main <model>`, where model can be any of the folders in the `models` folder.
To generate an overview of the results and to create the scatter plots, use `python -m metaqnn.display_results -h` for instructions.
To generate the GE graph or use the simple ensemble, use `python -m metaqnn.plot_top_ges -h` for instructions.

requirements.txt includes all requirements including the exact dependencies used, while requirements.minimal.txt only includes the explicitly installed requirements (generated with `pip-chill`)

**NB:** at most tensorflow 2.1, cuda 10.1 and cudnn 10.1-7.6.0.64 are required. Higher tensorflow versions have some breaking changes that cause the code crash (mostly the One-Cycle LR code). 
#
**This code is partially based on:**

**[Designing Neural Network Architectures Using Reinforcement Learning](https://arxiv.org/pdf/1611.02167.pdf)**   
Bowen Baker, Otkrist Gupta, Nikhil Naik, Ramesh Raskar  
*International Conference on Learning Representations*, 2017

The source code of which can be found on [Github](https://github.com/bowenbaker/metaqnn/) under the MIT License.
