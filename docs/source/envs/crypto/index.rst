Environments for Crypto Trading Task
====================================

.. toctree::
   :maxdepth: 1

   finrlmeta
   trademaster


Evaluation Pipeline for Crypto Market Environments
--------------------------------------------------

To isolate the effect of environment design on agent performance, the evaluation is conducted on a single-asset crypto trading task, with all non-environmental factors held constant (data, agent, training protocol, seeds, and evaluation metrics).  
The transaction cost for both buy and sell operations is set to **0.1%**.


Data Preprocessing
------------------

We load five-minute OHLCV data of **Bitcoin (BTC)** from Binance and apply a standardized preprocessing pipeline.  
This includes:

- Filling missing values.  
- Adding technical indicators such as **Moving Average Convergence Divergence (MACD)** and **Relative Strength Index (RSI)**.  
- Adding **turbulence indexes** to the dataset.  

The dataset is divided into **training**, **validation**, and **testing** sets, as illustrated in Figure 1.  

- Training set: 06/01/2024 00:00 - 06/30/2024 23:59
- Validation set: 07/01/2024 00:00 - 07/05/2024 23:59  
- Testing set: 07/06/2024 00:00 - 07/20/2024 23:59

.. figure:: ../../image/Data_split.png
   :width: 100%
   :align: center
   :name: fig_datasplit

   Illustration of data splitting



Training, Validation, and Testing
---------------------------------

The same training and evaluation procedure is used for the crypto market.  
We employ the **Proximal Policy Optimization (PPO)** algorithm (Schulman et al., 2017) from **Stable-Baselines3** (Raffin et al., 2021) to train the agent.  
The validation set is used to verify correctness and select hyperparameters, and the final performance is reported on the testing set.


Comparison
----------

Trading results are compared against established baselines, including:

- **S&P Cryptocurrency Broad Digital Market (S&P BDM) Index**  
- **Equal-weight strategy**  
- **Mean-Variance Portfolio Allocation Strategy**

Performance is evaluated using the following metrics:

- **Cumulative return**  
- **Annualized return**  
- **Annualized volatility**  
- **Sharpe ratio**  
- **Maximum drawdown**
