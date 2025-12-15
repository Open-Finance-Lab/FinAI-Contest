# Adapted from Trading_Gym (https://github.com/Yvictor/TradingGym)
# Original author: Yvictor (MIT License)
# Refactored by Xiangyu Ji, 2025

from __future__ import annotations

from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class TradingEnvStandardized(gym.Env):
    """
    Single-asset trading environment with discrete actions and rolling-window observations.

    ----------------------------------------------------
    Action Space
    ----------------------------------------------------
    Discrete(3):
        0 -> Hold (do nothing)
        1 -> Buy  (open / increase long)
        2 -> Sell (open / increase short)

    The agent cannot exceed the maximum allowed position `max_position`.
    - If a Buy at +max_position is attempted, it becomes Hold.
    - If a Sell at -max_position is attempted, it becomes Hold.

    ----------------------------------------------------
    State Space (internal)
    ----------------------------------------------------
    Market Data
    -----------
    - df_sample
        Sub-DataFrame representing one continuous trading session.
    - price
        1-D array of prices used for reward & PnL calculations.
    - obs_features
        2-D array of selected market features at each time step.

    Position State
    --------------
    - posi_arr
        1-D array of position (number of shares/contracts) at each time step.
    - posi_variation_arr
        1-D array of position changes due to trades.
    - posi_entry_cover_arr
        1-D array encoding type of trade:
            +1 -> new entry
            +2 -> increase existing position
            -1 -> partial cover
            -2 -> full close
             0 -> no trade

    Price Statistics
    ----------------
    - price_mean_arr
        1-D array of average entry price of current position at each time.

    Reward and P&L
    --------------
    - reward_fluctuant_arr
        Unrealized P&L:
            (price - price_mean_arr) * posi_arr
    - reward_makereal_arr
        Indicator whether P&L at a step is realized (1) or not (0).
    - reward_arr
        1-D array of realized reward per time step.

    Control / Bookkeeping
    ---------------------
    - step_st
        Start index of current observation window.
    - info
        Auxiliary info (e.g. joined df + transaction log) at episode end.
    - transaction_details
        DataFrame logging position, PnL, etc. at the end of the episode.

    ----------------------------------------------------
    Observation Space (agent-facing)
    ----------------------------------------------------
    The agent sees a rolling window of length `obs_len`.

    Base market window:
        obs_state = obs_features[step_st : step_st + obs_len]
        shape: (obs_len, feature_len)

    If `return_transaction=False`:
        observation = obs_state

    If `return_transaction=True`:
        For each timestep in the window, we concatenate:
          [market features,
           posi_arr,
           posi_variation_arr,
           posi_entry_cover_arr,
           price,
           price_mean_arr,
           reward_fluctuant_arr,
           reward_makereal_arr,
           reward_arr]
        so the number of columns becomes:
          feature_len + 8

    In both cases, the Gym observation space is a Box with shape:
        (obs_len, obs_feature_len)

    ----------------------------------------------------
    Reward (Option A)
    ----------------------------------------------------
    At each step, after trades and updates, the environment returns:

        step_reward = obs_reward.sum()

    where `obs_reward` is the slice of `reward_arr` over the current
    observation window. (Unrealized reward is not directly returned.)

    ----------------------------------------------------
    Initial Condition (reset)
    ----------------------------------------------------
    At reset():
    - df_sample           <- random session from full df
    - step_st             <- 0
    - price               <- df_sample[price_name].values
    - obs_features        <- df_sample[feature_names].values
    - posi_arr            <- zeros
    - posi_variation_arr  <- zeros
    - posi_entry_cover_arr<- zeros
    - price_mean_arr      <- copy of price
    - reward_fluctuant_arr<- zeros
    - reward_makereal_arr <- zeros
    - reward_arr          <- zeros
    - info                <- None
    - transaction_details <- empty DataFrame
    - t_index             <- 0
    - First observation window is taken from index 0 and padded if needed.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        obs_len: int,
        step_len: int,
        fee: float,
        max_position: int = 5,
        deal_col_name: str = "price",
        feature_names: List[str] | None = None,
        return_transaction: bool = False,
        fluc_div: float = 100.0,
        gameover_limit: float | None = None,
    ) -> None:
        super().__init__()

        assert obs_len >= 1, "obs_len must be >= 1"
        assert step_len >= 1, "step_len must be >= 1"
        feature_names = feature_names or ["price", "volume"]

        self.df = df.reset_index(drop=True)
        self.obs_len = int(obs_len)
        self.step_len = int(step_len)
        self.fee = float(fee)
        self.max_position = int(max_position)
        self.price_name = deal_col_name
        self.using_feature = feature_names
        self.feature_len = len(self.using_feature)
        self.return_transaction = return_transaction
        self.fluc_div = float(fluc_div)
        self.gameover_limit = gameover_limit

        # Precompute session boundaries:
        # we assume 'serial_number' marks session start (==0)
        if "serial_number" not in self.df.columns:
            raise ValueError("DataFrame must contain 'serial_number' column.")
        self.begin_fs = self.df[self.df["serial_number"] == 0]
        self.date_leng = len(self.begin_fs)

        # ---- Gym spaces ----
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation: rolling window with optional transaction features
        self.transaction_feature_len = 8 if self.return_transaction else 0
        self.obs_feature_len = self.feature_len + self.transaction_feature_len

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_len, self.obs_feature_len),
            dtype=np.float32,
        )

        # Internal episode state placeholders
        self.df_sample: pd.DataFrame | None = None
        self.price: np.ndarray | None = None
        self.obs_features: np.ndarray | None = None

        self.posi_arr: np.ndarray | None = None
        self.posi_variation_arr: np.ndarray | None = None
        self.posi_entry_cover_arr: np.ndarray | None = None
        self.price_mean_arr: np.ndarray | None = None
        self.reward_fluctuant_arr: np.ndarray | None = None
        self.reward_makereal_arr: np.ndarray | None = None
        self.reward_arr: np.ndarray | None = None

        self.step_st: int = 0
        self.t_index: int = 0
        self.info: Any = None
        self.transaction_details: pd.DataFrame = pd.DataFrame()

        # Observation caches
        self.obs_state: np.ndarray | None = None
        self.obs_posi: np.ndarray | None = None
        self.obs_posi_var: np.ndarray | None = None
        self.obs_posi_entry_cover: np.ndarray | None = None
        self.obs_price: np.ndarray | None = None
        self.obs_price_mean: np.ndarray | None = None
        self.obs_reward_fluctuant: np.ndarray | None = None
        self.obs_makereal: np.ndarray | None = None
        self.obs_reward: np.ndarray | None = None
        self.obs_return: np.ndarray | None = None
    # ------------------------------------------------------------------
    # Session sampler
    # ------------------------------------------------------------------
    def _random_choice_section(self) -> pd.DataFrame:
        """Randomly choose a session from df."""
        random_int = np.random.randint(self.date_leng)
        if random_int == self.date_leng - 1:
            begin_point = self.begin_fs.index[random_int]
            end_point = None
        else:
            begin_point, end_point = self.begin_fs.index[random_int : random_int + 2]
        df_section = self.df.iloc[begin_point:end_point]
        return df_section

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Sample one session
        self.df_sample = self._random_choice_section()
        self.step_st = 0

        # Define price and features
        self.price = self.df_sample[self.price_name].to_numpy(dtype=float)
        self.obs_features = self.df_sample[self.using_feature].to_numpy(dtype=float)

        # Position & PnL arrays
        self.posi_arr = np.zeros_like(self.price, dtype=float)
        self.posi_variation_arr = np.zeros_like(self.posi_arr, dtype=float)
        self.posi_entry_cover_arr = np.zeros_like(self.posi_arr, dtype=float)

        self.price_mean_arr = self.price.copy()
        self.reward_fluctuant_arr = (self.price - self.price_mean_arr) * self.posi_arr
        self.reward_makereal_arr = np.zeros_like(self.posi_arr, dtype=float)
        self.reward_arr = np.zeros_like(self.posi_arr, dtype=float)

        self.info = None
        self.transaction_details = pd.DataFrame()
        self.t_index = 0

        # Build initial observation window
        self._update_obs_window()

        return self.obs_return.astype(np.float32), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Parameters
        ----------
        action : int
            0=Hold, 1=Buy (long), 2=Sell (short).

        Returns
        -------
        obs : np.ndarray
            Next observation window of shape (obs_len, obs_feature_len).
        reward : float
            Per-step reward (here: sum of realized rewards over the window).
        terminated : bool
            True if episode ended (end of session or gameover).
        truncated : bool
            False (no time-limit truncation implemented).
        info : dict
            Extra info (contains `info` only at episode end).
        """
        assert self.df_sample is not None and self.price is not None

        current_index = self.step_st + self.obs_len - 1
        current_price_mean = float(self.price_mean_arr[current_index])
        current_mkt_position = float(self.posi_arr[current_index])

        self.t_index += 1
        self.step_st += self.step_len

        # Refresh observation window & change slices
        self._update_obs_window()

        # Change buffers: last `step_len` rows of the window
        self.chg_posi = self.obs_posi[-self.step_len:]
        self.chg_posi_var = self.obs_posi_var[-self.step_len:]
        self.chg_posi_entry_cover = self.obs_posi_entry_cover[-self.step_len:]
        self.chg_price = self.obs_price[-self.step_len:]
        self.chg_price_mean = self.obs_price_mean[-self.step_len:]
        self.chg_reward_fluctuant = self.obs_reward_fluctuant[-self.step_len:]
        self.chg_makereal = self.obs_makereal[-self.step_len:]
        self.chg_reward = self.obs_reward[-self.step_len:]

        terminated = False

        # ----------------- End-of-session forced close -----------------
        if self.step_st + self.obs_len + self.step_len >= len(self.price):
            terminated = True
            # No new action; force close on last segment if position != 0
            if current_mkt_position != 0:
                self.chg_price_mean[:] = current_price_mean
                self.chg_posi[:] = 0.0
                self.chg_posi_var[:1] = -current_mkt_position
                self.chg_posi_entry_cover[:1] = -2.0
                self.chg_makereal[:1] = 1.0
                # Realized reward for closing all
                self.chg_reward[:] = (
                    (self.chg_price - self.chg_price_mean) * current_mkt_position
                    - abs(current_mkt_position) * self.fee
                ) * self.chg_makereal

            # Build transaction_details & info at the end
            self.transaction_details = pd.DataFrame(
                [
                    self.posi_arr,
                    self.posi_variation_arr,
                    self.posi_entry_cover_arr,
                    self.price_mean_arr,
                    self.reward_fluctuant_arr,
                    self.reward_makereal_arr,
                    self.reward_arr,
                ],
                index=[
                    "position",
                    "position_variation",
                    "entry_cover",
                    "price_mean",
                    "reward_fluctuant",
                    "reward_makereal",
                    "reward",
                ],
                columns=self.df_sample.index,
            ).T
            self.info = self.df_sample.join(self.transaction_details)

        # ----------------- Trading logic (no forced close) --------------
        if not terminated:
            # Use next tick inside the change segment as entry price
            enter_price = float(self.chg_price[0])

            # Respect position limits (convert impossible actions to Hold)
            if action == 1 and current_mkt_position == self.max_position:
                action = 0
            if action == 2 and current_mkt_position == -self.max_position:
                action = 0

            # Long / short / cover / stay
            if action == 1 and self.max_position > current_mkt_position >= 0:
                # open/increase long
                open_posi = (current_mkt_position == 0)
                self._long(open_posi, enter_price, current_mkt_position, current_price_mean)

            elif action == 2 and -self.max_position < current_mkt_position <= 0:
                # open/increase short
                open_posi = (current_mkt_position == 0)
                self._short(open_posi, enter_price, current_mkt_position, current_price_mean)

            elif action == 1 and current_mkt_position < 0:
                # covering short
                self._short_cover(current_price_mean, current_mkt_position)

            elif action == 2 and current_mkt_position > 0:
                # covering long
                self._long_cover(current_price_mean, current_mkt_position)

            elif action == 0 and current_mkt_position != 0:
                # stay on with current position
                self._stayon(current_price_mean, current_mkt_position)

        # ----------------- Update unrealized P&L ------------------------
        self.chg_reward_fluctuant[:] = (
            (self.chg_price - self.chg_price_mean) * self.chg_posi
            - np.abs(self.chg_posi) * self.fee
        )

        # At this point, reward_arr & reward_fluctuant_arr windows are updated
        # and obs_return should be recomputed with latest arrays
        self._update_obs_window()

        # Here we simply return the sum of reward over the observation window,
        # consistent with your current implementation.
        step_reward = float(self.obs_reward.sum())

        # Optional gameover condition, if configured
        if self.gameover_limit is not None:
            # approximate equity = sum(realized) + last unrealized
            current_idx = self.step_st + self.obs_len - 1
            realized_total = float(self.reward_arr[: current_idx + 1].sum())
            unrealized = float(self.reward_fluctuant_arr[current_idx])
            equity = realized_total + unrealized
            if equity < -self.gameover_limit:
                terminated = True

        return self.obs_return.astype(np.float32), step_reward, terminated, False, (
            {} if self.info is None else {"info": self.info}
        )

    # ------------------------------------------------------------------
    # Trading helpers (array-based)
    # ------------------------------------------------------------------
    def _long(
        self,
        open_posi: bool,
        enter_price: float,
        current_mkt_position: float,
        current_price_mean: float,
    ) -> None:
        if open_posi:
            self.chg_price_mean[:] = enter_price
            self.chg_posi[:] = 1.0
            self.chg_posi_var[:1] = 1.0
            self.chg_posi_entry_cover[:1] = 1.0
        else:
            after_act_mkt_position = current_mkt_position + 1.0
            self.chg_price_mean[:] = (current_price_mean * current_mkt_position + enter_price) / after_act_mkt_position
            self.chg_posi[:] = after_act_mkt_position
            self.chg_posi_var[:1] = 1.0
            self.chg_posi_entry_cover[:1] = 2.0

    def _short(
        self,
        open_posi: bool,
        enter_price: float,
        current_mkt_position: float,
        current_price_mean: float,
    ) -> None:
        if open_posi:
            self.chg_price_mean[:] = enter_price
            self.chg_posi[:] = -1.0
            self.chg_posi_var[:1] = -1.0
            self.chg_posi_entry_cover[:1] = 1.0
        else:
            after_act_mkt_position = current_mkt_position - 1.0
            self.chg_price_mean[:] = (
                current_price_mean * abs(current_mkt_position) + enter_price
            ) / abs(after_act_mkt_position)
            self.chg_posi[:] = after_act_mkt_position
            self.chg_posi_var[:1] = -1.0
            self.chg_posi_entry_cover[:1] = 2.0
    
    def _short_cover(self, current_price_mean: float, current_mkt_position: float) -> None:
        self.chg_price_mean[:] = current_price_mean
        self.chg_posi[:] = current_mkt_position + 1.0
        self.chg_makereal[:1] = 1.0
        self.chg_reward[:] = (
            (self.chg_price - self.chg_price_mean) * (-1.0) - self.fee
        ) * self.chg_makereal
        self.chg_posi_var[:1] = 1.0
        self.chg_posi_entry_cover[:1] = -1.0

    def _long_cover(self, current_price_mean: float, current_mkt_position: float) -> None:
        self.chg_price_mean[:] = current_price_mean
        self.chg_posi[:] = current_mkt_position - 1.0
        self.chg_makereal[:1] = 1.0
        self.chg_reward[:] = (
            (self.chg_price - self.chg_price_mean) * (1.0) - self.fee
        ) * self.chg_makereal
        self.chg_posi_var[:1] = -1.0
        self.chg_posi_entry_cover[:1] = -1.0

    def _stayon(self, current_price_mean: float, current_mkt_position: float) -> None:
        self.chg_posi[:] = current_mkt_position
        self.chg_price_mean[:] = current_price_mean

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _update_obs_window(self) -> None:
        """Update all obs_* slices and obs_return from current step_st."""
        s = self.step_st
        e = self.step_st + self.obs_len

        # Clip window if near the end
        e = min(e, len(self.price))

        self.obs_state = self.obs_features[s:e]
        self.obs_posi = self.posi_arr[s:e]
        self.obs_posi_var = self.posi_variation_arr[s:e]
        self.obs_posi_entry_cover = self.posi_entry_cover_arr[s:e]
        self.obs_price = self.price[s:e]
        self.obs_price_mean = self.price_mean_arr[s:e]
        self.obs_reward_fluctuant = self.reward_fluctuant_arr[s:e]
        self.obs_makereal = self.reward_makereal_arr[s:e]
        self.obs_reward = self.reward_arr[s:e]

        if self.return_transaction:
            self.obs_return = np.concatenate(
                (
                    self.obs_state,
                    self.obs_posi[:, np.newaxis],
                    self.obs_posi_var[:, np.newaxis],
                    self.obs_posi_entry_cover[:, np.newaxis],
                    self.obs_price[:, np.newaxis],
                    self.obs_price_mean[:, np.newaxis],
                    self.obs_reward_fluctuant[:, np.newaxis],
                    self.obs_makereal[:, np.newaxis],
                    self.obs_reward[:, np.newaxis],
                ),
                axis=1,
            )
        else:
            self.obs_return = self.obs_state

        # If we're at the end and window is shorter than obs_len, pad
        if self.obs_return.shape[0] < self.obs_len:
            pad_rows = self.obs_len - self.obs_return.shape[0]
            pad = np.repeat(self.obs_return[-1:, :], pad_rows, axis=0)
            self.obs_return = np.concatenate([self.obs_return, pad], axis=0)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self) -> np.ndarray:
        """Simple render: return current observation (for debugging)."""
        return self.obs_return.copy()

    # ------------------------------------------------------------------
    # SB3 helper
    # ------------------------------------------------------------------
    def get_sb_env(self) -> Tuple[DummyVecEnv, Any]:
        """Wrap this env in SB3 DummyVecEnv and return (env, obs)."""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
