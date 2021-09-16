DDPG-hgru4rec
=============

Source Code generated during Masters Studies Thesis: *Exploiting Inter-session Dynamics for Long Intra-Session Sequences of Interactions with Deep Reinforcement Learning for Session-Aware Recommendation [link](https://www.teses.usp.br/teses/disponiveis/55/55134/tde-23062021-105306/en.php)*

This repository includes a Pytorch implementation of the hgru4rec architecture as close as possible to the [original implementation](https://github.com/mquad/hgru4rec) and an adaptation of the Deep Deterministic Policy Gradient Algorithm to adjust the hidden state of the intra-session level network by introducing inter-session based perturbations.  


Requirements
============
- Cuda 10.1 (Driver 430.64 Ubuntu 16.04)
- Pytorch 1.3.1
- Pandas 0.24
- PyTables 3.4.4
- Numpy 1.15.4

Additionally you will need to create several folders inside the **root** folder to store models and preprocessed data 
- Datasets
- Models
- Results
 
Datasets
========
This repository includes preprocessing files for the following datasets: 

- [30 Music Dataset](http://recsys.deib.polimi.it/datasets/)
- [Repeat buyers challenge](https://tianchi.aliyun.com/competition/entrance/231576/information)
To build datasets execute `/data/BuildDataset.ipynb`

Execution
=========
In your terminal execute the file adding the `split.sh`
