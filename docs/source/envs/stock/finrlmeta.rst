FinRL-Meta
==========

Action Space
------------
The action space is a ``Box(-1,1,(stock_dim,),float32)``. 
Each element *a* is in the range [-1,1], representing the normalized number
of shares to buy or sell for the corresponding stock. 

- ``a > 0``: Buy ``a*hmax`` shares
- ``a = 0``: Hold
- ``a < 0``: Sell ``a*hmax`` shares

State Space
-----------
The state space consists of the following components:

- **Account balance**: 1 element
- **Stock prices**: ``stock_dim`` elements
- **Holding positions**: ``stock_dim`` elements
- **Feature vector**: ``I × stock_dim`` elements (``I`` features per asset)

Thus, the total state dimension is ``1 + (I + 2) × stock_dim``.

Transition Dynamics
-------------------


Reward Design
-------------
The goal is to maximize the gain in asset value, thus the reward is defined as the change in asset value, multiplied by a scaling multiplier.
It is calculated by ``(end_total_asset - begin_total_asset)*reward_scaling``

Initial State
-------------
At the start of each episode:

- **Account balance** is initialized to ``1,000,000``.
- **Holding positions** are initialized to ``0`` for all assets.
- **Stock prices** are set to the prices from the first trading day.
- **Feature vector** is initialized using the feature values for the first day.


Ending Condition
----------------
The episode terminates when the trading day would exceed the available data range.

Documentation
-------------
.. currentmodule:: finai_contest.env_stock_trading.env_stock_trading_meta

.. autoclass:: StockTradingEnv_FinRLMeta
   :members:
   :inherited-members:
   :show-inheritance: