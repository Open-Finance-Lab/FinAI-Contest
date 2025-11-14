# Trading_gym
original code: https://github.com/Yvictor/TradingGym/blob/master/trading_env/envs/training_v1.py

To evaluate the stock market environment from gym-anytrading, follow the following steps.
## Installation
```
git clone https://github.com/Yvictor/TradingGym.git
cd TradingGym
python setup.py install
```
Requirements are met if you ran the pipeline files in the same virtual environment.


## Run Original Code
We choose the script from README.md at https://github.com/Yvictor/TradingGym/blob/master/README.md

1. Use the same YahooFinance Data

Run 1_Data.ipynb, and copy the three files generated to the same directory. To align the data, use the following function to convert our data to the expected format.
```
import numpy as np
import pandas as pd

def ohlcv_to_tradinggym(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = df_ohlcv.copy()

    df["datetime"] = pd.to_datetime(df["date"], utc=False)
    df = df.sort_values("datetime").dropna(subset=["close", "volume"])

    df["Price"]  = df["close"].astype(float)
    df["Volume"] = df["volume"].astype(float)

    spread = df["Price"] * 0.0001
    df["Ask_price"] = (df["Price"] + spread/2).astype(float)
    df["Bid_price"] = (df["Price"] - spread/2).astype(float)

    updown = np.sign(df["Price"].diff()).fillna(0.0)
    df["Updown"] = updown.astype(float)
    df["Updown_Cum"] = df["Updown"].cumsum()

    ask_ratio = np.where(df["Updown"] > 0, 0.7, np.where(df["Updown"] < 0, 0.3, 0.5))
    df["Ask_deal_vol"] = (df["Volume"] * ask_ratio).astype(float)
    df["Bid_deal_vol"] = (df["Volume"] - df["Ask_deal_vol"]).astype(float)

    df["Bid/Ask_deal"] = (df["Bid_deal_vol"] - df["Ask_deal_vol"]) / (
        (df["Bid_deal_vol"] + df["Ask_deal_vol"]).replace(0, np.nan)
    )
    df["Bid/Ask_deal"] = df["Bid/Ask_deal"].fillna(0.0)

    df["serial_number"] = np.arange(len(df), dtype=int)
    df["Comm"] = "CUSTOM"

    cols = [
        "serial_number","datetime","Comm","Price","Volume",
        "Ask_price","Bid_price","Ask_deal_vol","Bid_deal_vol",
        "Bid/Ask_deal","Updown","Updown_Cum"
    ]
    return df[cols]

train = pd.read_csv('../data/train_data.csv')
df = ohlcv_to_tradinggym(train)
df
```

2. Modify training pipeline

The current file runs the training and trading stage with the same dataset, which makes results unsuitable for meaningful comparison with our pipeline. To address, create an environment for each stage, training and testing.
```
env = trading_env.make(env_id='backtest_v1', obs_data_len=10, step_len=1,
                       df=df, fee=0.1, max_position=5, deal_col_name='Price', 
                       feature_names=['Price'])
```

3. Modify training_v1.py

The current function would cause some errors due to the as_matrix() function. To fix it, change all as_matrix() function to to_numpy().
```
# define the price to calculate the reward
self.price = self.df_sample[self.price_name].to_numpy()
# define the observation feature
self.obs_features = self.df_sample[self.using_feature].to_numpy()
```

4. Save training data

```
class RandomAgent:
    def __init__(self):
        pass
        
    def choice_action(self, state):
        return np.random.randint(3)


agent = RandomAgent()

transactions = []
while not env.backtest_done:
    state = env.reset()
    done = False
    while not done:
        state, reward, done, info = env.step(agent.choice_action(state))
        print(reward)
        #env.render()
        if done:
            transactions.append(info)
            break
transaction = pd.concat(transactions)
os.makedirs('../results', exist_ok=True)
transaction.to_csv("../results/Trading_gym.csv")


realized = env.info["reward_fluctuant"] * (env.info["reward_makereal"] != 0).astype(float)
initial_cash   = 100_000.0
contract_size  = 1.0

equity_curve = initial_cash + (realized.fillna(0.0) * contract_size).cumsum()
start_money  = equity_curve.iloc[0]
end_money    = equity_curve.iloc[-1]

print(f"Start: {start_money:.2f}")
print(f"End:   {end_money:.2f}")
print(f"Return: {(end_money/start_money - 1)*100:.2f}%")

print("Full transaction saved to Trading_gym.csv")
```
The complete modified file can be found [here]().
## Reproduce in our pipeline

## Standardize the environment
