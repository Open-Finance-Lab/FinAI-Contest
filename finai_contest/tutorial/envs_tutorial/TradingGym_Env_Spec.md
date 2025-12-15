# Standardize the environment

## Action Space
The action space includes three discrete actions:
- **0 — Hold**
- **1 — Buy**
- **2 — Sell**

Internally, the environment sets:
```python
self.action_space = np.array([3,])
self.gym_actions = range(3)
```

Action constraints:
The agent **cannot exceed** the maximum allowed position `max_position`.
If attempting to buy while already at `+max_position`, action becomes **hold**.
If attemping to sell while already at `-max_postion`, action becomes **hold**.

## State Space
### **Market Data**
- **Market data segment**(`df_sample`): defines the time range over which the episode is executed.
- **Stock price**(`price`): is used as the reference for all reward and P&L calculations.
- **Market feature matrix**(`obs_features`): A two-dimensional array containing the selected market features at each time step.
### **Position State**
- **Market position**(`posi_arr`): A one-dimensional array representing the agent's position at each time step, measured in number of shares of contracts.
- **Position variation**(`posi_variating_arr`): records how the position changes due to trading actions, such as increasing or decreasing exposure.
- **Position entry/ cover indicator**(`posi_entry_cover_arr`): A one-dimensional array encoding whether a position change corresponds to an entry or a cover operation.
### **Price Statistics**
- **Average position price**(`price_mean_arr`): A one-dimensional array representing the average entry price of current position at each time step.
### **Reward and Profit & Loss**
- **Unrealized reward**(`reward_fluctuant_arr`): defined as the difference between the current price and the average position price, multiplied by the current position size.
- **Realized reward indicator**(`reward_makereal_arr`): A one-dimensional arry indicating whether the reward at a given time step is realized.
- **Realized reward**(`reward_arr`): A one-dimensional array representing the realized profit or loss at each time step.
## **Control and Bookkeeping**
- **Step pointer**(`step_st`): An integer indicating the starting index of the current observation window within the trading session.
- **Additional information**(`info`): A placeholder variable for returning auxiliary information when the episode terminates.
- **Transaction log**(`transaction_details`): An empty DataFrame used to record detailed information about all trading action, positions, and rewards during the episode.

## Observation Space
`ob_state` - **rolling market window**
- Shape: `(obs_len, feature_len)`
- Definition: a slice of 'obs_features' starting at `step_st`:
```python
obs_state = obs_features[step_st : step_st + obs_len]
```
- Each row correspond to one time step in the recent past.
- Each column corresponds to one of the chosen `feature_names`(e.g. `price`, `volume`, etc)

## Reward Design
The environment has two types of reward:
- **Unrealized (fluctuating) reward**

If the agent does not close a position:
```python
reward_ret = reward_fluctuant / fluc_div
```
- **Realized reward**

When the agent close a long or short position ("cover" action):
```python
reward = (sell_price - price_mean - fee) * position    # long closing
reward = (price_mean - buy_price - fee) * (-position)  # short closing
```
This reward is added directly to `reward_sum`.

Notes:
- Rewards are positive for profitable closes, negative for losses.
- Unrealized rewards are scaled by `fluc_div` to avoid exploding values.

## Transition Dynamics
At each environment step, the following process occur:
1. **Interpret Agent Action**
- Action 1 -> attempt to **buy**
- Action 2 -> attempt to **sell**
- Action 0 -> **hold**

Position constraints are enforced (cannot exceed ± `max_position`)

2. **Execute Trade Logic**

The environment handles:
- **Opening** long/short positions (`transact_type = "new"`)
- **Adding to existing** long/short positions
- **Closing (covering)** opposite-side positions (`transact_type = "cover"`)

For each trade:
- Position count is updated
- Average price (`price_mean`) is recalculated
- Fees are applied
- Profit/los is recorded into `reward_sum`
- A new row is appended to `transaction_details`

3. **Advance Market Timeline**

After trade evaluation:
```python
self.step_st += self.step_len
```
The new observation features:
```python
self.obs_res = self.obs_features[self.step_st : self.step_st + self.obs_len]
```

4. **Check Termination Conditions**
Episode ends if:
- The observation window reaches the end of available data (`step_st + obs_len + step_len ≥ total_length`)
- The agent hits the game-over loss threshold (`reward_sum + reward_fluctuant < -gameover_limit`)

When ending:
- Any open position is automatically closed
- Final realized reward is added to `reward_sum`

## Initial Condition
- **Market data segment** (`df_sample`) is set to `_random_choice_section`.
- **Stock price** (`price`) extracted from the selected sesion and stored as a NumPy array.
- **Market feature matrix** (`obs_features`) extraced from `df_sample` according to `feature_names`.
- **Market position** (`posi_arr`) initialized to an array of zeros.
- **Position variation** (`posi_variation_arr`) is initialized to zeros.
- **Position entry/ cover indicator** (`posi_entry_cover_arr`) is initialized to zeros.
- **Average position price** (`price_mean_arr`) is initialized as a copy of stock price array.
- **Unrealized reward** (`reward_fluctuation_arr`) is initialized to zero for all time steps.
- **Realized reward indicator** (`reward_makereal_arr`) is initialized to `0`.
- **Realized reward** (`reward_arr`) is initialized to `0`.
- **Step pointer** (`step_st`) is set to `0`.
- **Additional information** (`info`) is initialized to `None`.
- **Transaction log** (`transaction_details`) is initalized as an empty DataFrame.
