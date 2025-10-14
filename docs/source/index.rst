.. FinAI Contest documentation master file, created by
   sphinx-quickstart on Tue Jan 21 19:38:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FinAI Contest Documentation
==================================================

.. Add your content using ``reStructuredText`` syntax. See the
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
   documentation for details.

FinAI Contest is a series of contests focusing on applying artificial intelligence in various financial tasks. The contest aims to promote open finance and encourage the development of advanced financial technologies.

Financial reinforcement learning (FinRL) applies reinforcement learning algorithms to financial tasks, such as order execution, portfolio management, high-frequency trading, option pricing and hedging, and market making. The recent breakthroughs of large language models (LLMs) is driving open finance, which provides affordable and scalable solutions for customers to make intelligent decisions and created personalized financial search and robo-advisors.

From 2023 to 2025, we have organized three FinRL Contests featuring different financial tasks, leveraging reinforcement learning and LLMs. These tasks allow contestants to participate in various financial tasks and contribute to open finance using state-of-the-art technologies.

From FinRL and FinRL-Meta, we select stable market environments to define tasks. We provide diverse datasets, including OHLCV data, limit order books (LOB) data, and financial news. We also provide environments that incorporate LLM-generated signals and support massively parallel simulation. We provide an ensemble learning approach with GPU-optimized parallel environments to mitigate policy instability and address sampling bottleneck. 

.. image:: ./image/FinRL_Contest_Tasks.png
   :width: 100%
   :align: center

FinRL Contest 2025: `Contest Website <https://open-finance-lab.github.io/FinRL_Contest_2025/>`__; `Starter Kit <https://github.com/Open-Finance-Lab/FinRL_Contest_2025>`__

FinRL Contest 2024: `Contest Website <https://open-finance-lab.github.io/finrl-contest-2024.github.io/>`__; `Starter Kit <https://github.com/Open-Finance-Lab/FinRL_Contest_2024>`__

FinRL Contest 2023: `Contest Website <https://open-finance-lab.github.io/finrl-contest.github.io/>`__; `Starter Kit <https://github.com/Open-Finance-Lab/FinRL_Contest_2023>`__


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   intro/overview
   intro/data
   intro/env
   intro/agent

.. toctree::
   :caption: Market Environments
   :hidden:
   :maxdepth: 2

   envs/overview
   envs/stock/index
   envs/crypto/index

.. toctree::
   :maxdepth: 2
   :caption: Baseline

   baseline/market
   baseline/regression

.. toctree::
   :maxdepth: 2
   :caption: SecureFinAI Contest 2025

   securefinai2025/overview
   securefinai2025/task1
   securefinai2025/task2
   securefinai2025/task3

.. toctree::
   :maxdepth: 2
   :caption: FinAI Contest 2025

   finai2025/overview
   finai2025/task1
   finai2025/task2
   finai2025/task3

.. toctree::
   :maxdepth: 2
   :caption: FinRL Contest 2025

   finrl2025/overview
   finrl2025/task1
   finrl2025/task2
   finrl2025/task3
   finrl2025/task4
   finrl2025/accepted_paper

.. toctree::
   :maxdepth: 2
   :caption: FinRL Contest 2024

   finrl2024/overview
   finrl2024/task1
   finrl2024/task2 

.. toctree::
   :maxdepth: 2
   :caption: FinRL Contest 2023

   finrl2023/overview
   finrl2023/task1
   finrl2023/task2

.. toctree::
   :maxdepth: 2
   :caption: Resources

   resource/resource

