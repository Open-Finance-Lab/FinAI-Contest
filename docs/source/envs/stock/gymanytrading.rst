StockTradingEnv_gym_anytrading
==============================
Action Space
------------
The action space is discrete, with two actions:

- 0 : Sell 
- 1 : Buy

State Space
-----------

The state space consists of the following components:


- **Account balance** (1 element): current account balance
- **Stock price** (1 element): current price for the stock
- **Holding position** (1 element): number of shares held for the stock
- **Price difference** (1 element): the change in price since the previous timestep
- **Position** (1 element): 0 for short, 1 for long


Observation Space
-----------------

The agent can observe the following components:

- **Stock price**: 1 element
- **Price difference**: 1 element

Each observation contains a rolling history of these elements over the most recent ``window_size`` time steps.

Thus, the total observation space dimension is ``2*window_size``.

Transition Dynamics
-------------------

At each timestep, the environment performs the following updates:

1. **Decode Actions (Discrete)**
   
   The action for each asset is a discrete flag:
   ``0`` = **SELL/close long**, ``1`` = **BUY/open long**.
   Current position flags are kept in ``self.position`` (``0`` = short, ``1`` = long).

   .. code-block:: python

      acts = np.asarray(actions, dtype=int).reshape(-1)
      pos  = np.asarray(self.position, dtype=int).reshape(-1)

2. **Execute Trades**

   We resolve the intended transitions into two disjoint sets:

   .. code-block:: python

      sell_hits = np.where((acts == 0) & (pos == 1))[0]
      buy_hits  = np.where((acts == 1) & (pos == 0))[0]

   - **BUY (open long)** — from short:

     Calls ``_buy_stock(i, qty)`` with ``qty = self.hmax`` (the function **ignores**
     the requested quantity and buys the **maximum affordable** number of shares):

     .. math::

        \text{unit_cost}^{(i)} = \text{price}^{(i)} \bigl(1 + c^{\text{buy}}_i\bigr),\qquad
        \text{actual}^{(i)} = \left\lfloor \frac{\text{cash}}{\text{unit_cost}^{(i)}} \right\rfloor.

     Internal accounting:

     - Cash decreases by ``price * actual + fee`` where ``fee = price * actual * buy_cost_pct[i]``.
     - Holdings increase by ``actual`` shares.
     - ``self.last_trade_price[i]`` is set to the entry price for later reward.
     - Trade counter and cumulative cost are updated.
     - Position flag becomes ``1`` (long).

   - **SELL (close long)** — from long:

     Calls ``_sell_stock(i, qty)`` with ``qty = min(hmax, holding)`` but the function
     **ignores** the request and sells **all** current shares:

     - Cash increases by ``price * actual - fee`` where ``fee = price * actual * sell_cost_pct[i]``.
     - Holdings drop to zero.
     - Trade counter and cumulative cost are updated.
     - Position flag becomes ``0`` (flat).

3. **Advance Time & Update State / Observation**

   - Advance the market clock: ``self.day += 1`` and load the new row: ``self.data = self.df.loc[self.day, :]``.
   - Recompute turbulence if applicable.
   - Rebuild the internal state (cash, price, holdings, price_diff, etc.) via ``_update_state()``.
   - Append the **observation frame** `[price, price_diff]` to the rolling buffer:

     .. code-block:: python

        self._frames.append(np.array([self.state[1], self.state[3]], dtype=np.float32))

   - ``_get_obs()`` concatenates the last ``window_size`` frames to produce the agent’s observation.

**Constraints**

- **Costs:** proportional buy/sell costs are applied via ``buy_cost_pct[i]`` and ``sell_cost_pct[i]``.



Reward Design
-------------

Reward is given **only** when a SELL closes an existing LONG. For each such asset ``i``:

.. math::

   r^{(i)}_t \;=\; \bigl(\text{sell_price}^{(i)}_t - \text{last_buy_price}^{(i)}\bigr).

The step reward is the sum over assets that closed a long this step:

.. code-block:: python

   price_diff[i] = current_sell_price - self.last_trade_price[i]
   reward = float(sum(price_diff))

All other actions (opening a long, staying flat, or forced no-trade) yield **0** reward.

Initial State
-------------
At the start of each episode:

- **Account balance** is initialized to ``1,000,000``.
- **Holding position** is initialized to ``0``.
- **Stock price** is set to the price from the first trading day.
- **Price difference** is set to the change in price since the last day in the observation window.

Ending Condition
----------------
The episode terminates when the trading day would exceed the available data range.

Documentation
-------------
.. currentmodule:: finai_contest.env_stock_trading.env_stock_trading_gym_anytrading

.. autoclass:: StockTradingEnv_gym_anytrading
   :members:
   :inherited-members:
   :show-inheritance: