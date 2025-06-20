---
tags: []
parent: ""
collections:
    - 'A Reconfiguration'
$version: 73640
$libraryID: 1
$itemKey: VLICNBHC

---
全文翻译：A multi-agent reinforcement learning method for distribution system restoration considering dynamic network reconfiguration

> # A multi-agent reinforcement learning method for distribution system restoration considering dynamic network reconfiguration

# 一种考虑动态网络重构的配电网恢复多智能体强化学习方法

> Ruiqi ${\mathrm{{Si}}}^{\mathrm{a}}$ ,Siyuan Chen ${}^{\mathrm{a}}$ ,Jun Zhang ${}^{\mathrm{a}, * }$ ,Jian ${\mathrm{{Xu}}}^{\mathrm{a}}$ ,Luxi Zhang ${}^{\mathrm{b}}$

睿琪 ${\mathrm{{Si}}}^{\mathrm{a}}$，陈思远 ${}^{\mathrm{a}}$，张俊 ${}^{\mathrm{a}, * }$，建 ${\mathrm{{Xu}}}^{\mathrm{a}}$，张璐茜 ${}^{\mathrm{b}}$

> a School of Electrical Engineering and Automation, Wuhan University, Wuhan, 430072, China

武汉大学电气工程与自动化学院，武汉，430072，中国

> ${}^{\mathrm{b}}$ Brandeis University,Waltham,02454,USA

${}^{\mathrm{b}}$ 布兰迪斯大学，沃尔瑟姆，02454，美国

> ## ARTICLE INFO

## 文章信息

> Keywords:

关键词：

> Deep reinforcement learning

深度强化学习

> Multi-agent reinforcement learning

多智能体强化学习

> Distribution system restoration

配电系统恢复

> Distribution network

配电网络

> Microgrid

微电网

> ## A B S T R A C T

## 摘要

> Extreme weather, chain failures, and other events have increased the probability of wide-area blackouts, which highlights the importance of rapidly and efficiently restoring the affected loads. This paper proposes a multi-agent reinforcement learning method for distribution system restoration. Firstly, considering that the topology of the distribution system may change during network reconfiguration, a dynamic agent network (DAN) architecture is designed to address the challenge of input dimensions changing in neural network. Two encoders are created to capture observations of the environment and other agents respectively, and an attention mechanism is used to aggregate an arbitrary-sized neighboring agent feature set. Then, considering the operation constraints of the DSR, an action mask mechanism is implemented to filter out invalid actions, ensuring the security of the strategy. Finally, an IEEE 123-node test system is used for validation, and the experimental results showed that the proposed algorithm can effectively assist agents in accomplishing collaborative DSR tasks.

极端天气、连锁故障等事件增加了大面积停电的可能性，这凸显了快速高效恢复受影响负荷的重要性。本文提出了一种用于配电系统恢复的多智能体强化学习方法。首先，考虑到配电系统在网络重构过程中拓扑结构可能发生变化，设计了一种动态智能体网络（DAN）架构，以应对神经网络输入维度变化的挑战。创建了两个编码器，分别用于捕捉环境和其他智能体的观测信息，并使用注意力机制来聚合任意大小的邻近智能体特征集。然后，考虑到配电系统恢复的操作约束，实施了动作掩码机制以过滤无效动作，确保策略的安全性。最后，使用IEEE 123节点测试系统进行验证，实验结果表明，所提出的算法能够有效协助智能体完成协作配电系统恢复任务。

> ## 1. Introduction

## 1. 引言

> Extreme weather, deliberate attacks, and other events pose significant threats to the reliable operation of the power system, resulting in major power outages and economic losses. In 2008, the ice disaster in China caused power outages affecting more than 170 areas and economic losses amounting to tens of billions \[1,2]. In 2012, Hurricane Sandy left 8 million residents in America without an electricity supply $\left\lbrack {3,4}\right\rbrack$ . Meanwhile,the rising penetration rate of renewable energy and power electronics has accentuated the vulnerability of new type power system. In 2016, Southern Australia, where renewable energy sources constituted ${48.36}\%$ of power generation,was struck by typhoons and rainstorms. The extreme weather conditions resulted in a significant disconnection of renewable energy sources, ultimately culminating in a widespread blackout \[5,6]. The distribution network, situated at the tail end of the power system, has the characteristics of light-boned network topologies, making it more susceptible to extreme events. Nowadays, the widespread integration of renewable energy and advanced control technologies has made it possible for the distribution network to operate as isolated microgrids (MGs). Therefore, the development of an efficient source-net-load collaborative distribution system restoration (DSR) strategy is crucial for protecting power system utilities and improving power user satisfaction \[7].

极端天气、人为攻击和其他事件对电力系统的可靠运行构成了重大威胁，导致大规模停电和经济损失。2008年，中国的冰灾导致170多个地区停电，经济损失达数百亿\[1,2]。2012年，飓风桑迪使美国800万居民断电$\left\lbrack {3,4}\right\rbrack$。与此同时，可再生能源和电力电子设备渗透率的上升加剧了新型电力系统的脆弱性。2016年，可再生能源发电占比${48.36}\%$的南澳大利亚遭遇台风和暴雨袭击。极端天气条件导致大量可再生能源发电中断，最终引发了大规模停电\[5,6]。配电网络位于电力系统的末端，具有轻量化网络拓扑结构的特点，使其更容易受到极端事件的影响。如今，可再生能源和先进控制技术的广泛集成使得配电网络能够作为孤立的微电网（MGs）运行。因此，开发高效的源-网-荷协同配电系统恢复（DSR）策略对于保护电力系统设施和提高电力用户满意度至关重要\[7]。

> The DSR problem is essentially a mixed-integer non-linear programming (MINLP). It involves a large number of integer variables (e.g., nodes, lines, load energization status) and non-linear physical coupling constraints (e.g., energization status relationship and several MGs radial topology). Currently, the majority of research focuses on solving this problem based on mixed-integer linear programming (MILP) \[8-14], mixed-integer second-order cone programming (MISOCP) \[15-18], and mixed-integer semidefinite programming (SDP) \[19]. \[8] proposes the "bus block" for aggregating buses and uses linear DistFlow constraints to cut down the problem complexity. \[19] uses a two-stage method with the first stage deciding the post-restoration topology by minimum diameter spanning tree algorithm, eliminating integer variables and constraints related to energization status and radial topology. \[20,21] uses the alternating direction method of multipliers (ADMM) to decompose the problem by relaxing binary variables, which reduces computational scale. Considering the singlestep optimization method only generates a final configuration, \[8] forms a feasible restoration sequence for system operators, avoiding manual decomposition and verification of each step operation. However, in the sequential DSR problem, dynamic switching operations make variables and constraints more complex. \[10] does not allow switches to close if both ends of the switch are already energized to avoid forming a loop. But it ignores the situation where two energized MGs form a large network through a closed switch. \[22] assumes each islanded MG is controlled by only one diesel generator (DG), which makes the number of MGs and DGs equal, simplifying the topology constraint. Overall, two aspects limit the model-based methods in source-net-load collaborative DSR. (1) With the more complex network structure and abundant controllable resources, the complexity of the DSR problem increases, posing challenges for quick solving in practical application. (2) The majority of multi-step DSR studies have certain assumptions and model simplifications, making it difficult to achieve maximal connectivity between sources and loads for multi-source coordination. And different simplification and mathematical processing techniques can greatly influence the result and efficiency.

DSR问题本质上是一个混合整数非线性规划（MINLP）问题。它涉及大量的整数变量（例如节点、线路、负荷通电状态）和非线性物理耦合约束（例如通电状态关系和多个微电网的辐射状拓扑）。目前，大多数研究集中在基于混合整数线性规划（MILP）\[8-14]、混合整数二阶锥规划（MISOCP）\[15-18]和混合整数半定规划（SDP）\[19]来解决该问题。\[8]提出了“总线块”用于聚合总线，并使用线性DistFlow约束来降低问题的复杂性。\[19]采用了两阶段方法，第一阶段通过最小直径生成树算法决定恢复后的拓扑，消除了与通电状态和辐射状拓扑相关的整数变量和约束。\[20,21]使用交替方向乘子法（ADMM）通过松弛二进制变量来分解问题，从而减少了计算规模。考虑到单步优化方法仅生成最终配置，\[8]为系统操作员形成了一个可行的恢复序列，避免了手动分解和验证每个步骤的操作。然而，在顺序DSR问题中，动态切换操作使变量和约束更加复杂。\[10]不允许在开关两端都已通电的情况下关闭开关，以避免形成环路。但它忽略了两个通电的微电网通过关闭的开关形成一个大型网络的情况。\[22]假设每个孤岛微电网仅由一个柴油发电机（DG）控制，这使得微电网和柴油发电机的数量相等，简化了拓扑约束。总体而言，基于模型的方法在源-网-荷协同DSR中存在两个方面的限制。（1）随着网络结构更加复杂和可控资源的丰富，DSR问题的复杂性增加，对实际应用中的快速求解提出了挑战。（2）大多数多步DSR研究都有一定的假设和模型简化，难以实现源与负荷之间的最大连接以进行多源协调。不同的简化和数学处理技术会极大地影响结果和效率。

***

> ☆ This work was supported by the National Key R\&D Program of China under Grant No. 2022YFB2403500.

☆ 本工作得到了国家重点研发计划（项目编号：2022YFB2403500）的资助。

> *   Corresponding author.

*   通讯作者。

> E-mail addresses: <ruiqi.si@whu.edu.cn> (R. Si), <wddqcsy@whu.edu.cn> (S. Chen), <jun.zhang.ee@whu.edu.cn> (J. Zhang), <xujian@whu.edu.cn> (J. Xu), <Luxizhang@brandeis.edu> (L. Zhang).

电子邮件地址：<ruiqi.si@whu.edu.cn> (R. Si)，<wddqcsy@whu.edu.cn> (S. Chen)，<jun.zhang.ee@whu.edu.cn> (J. Zhang)，<xujian@whu.edu.cn> (J. Xu)，<Luxizhang@brandeis.edu> (L. Zhang)。

***

| <!-- --> | <!-- --> |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Nomenclature                                        |                                                                                            |
| $\mathcal{E}$                                       | Set of all branches, including line and switch                                             |
| ✓                                                   | Set of all nodes                                                                           |
| ${\mathcal{N}}_{i}$                                 | Set of nodes in the same MG as node $i$                                                    |
| $T$                                                 | Total number of steps of the recovery process                                              |
| ${G}_{ij},{B}_{ij}$                                 | Conduction and susceptance of branch between node $i$ and node $j$                         |
| ${n}_{b}$                                           | Total number of nodes in the distribution network                                          |
| ${S}_{i}^{G}$                                       | Rated capacity of DG connected to node $i$                                                 |
| ${V}_{i,\max },{V}_{i,\min }$                       | Max,min permissive voltage of node $i$                                                     |
| ${I}_{{ij},\max },{I}_{{ij},\min }$                 | Max, min permissive current of branch ij                                                   |
| $\alpha$                                            | Max permissive restored load power per step                                                |
| ${c}_{i}$                                           | Weight factor of load connected to node $i$                                                |
| ${x}_{i,t}^{\text{load }}$                          | Energization status of load connected to node $i$ at step $t$ ,1-energized,0-de-energized  |
| ${x}_{i,t}^{\text{node }}$                          | Energization status of node $i$ at step $t$ , 1-energized, 0-de-energized                  |
| ${x}_{{ij},t}^{BR}$                                 | Energization status of branch ij at step $t$ , 1-energized, 0-de-energized                 |
| ${x}_{{ij},t}^{SW}$                                 | Energization status of switch connected to node $i$ and $j$ at step $t$ ,1-closed,0-opened |
| ${p}_{i,t}^{G},{q}_{i,t}^{G}$                       | Active and reactive power output of DG connected to node $i$ at step $t$                   |
| ${p}_{i,t}^{\text{load }},{q}_{i,t}^{\text{load }}$ | Active and reactive power demand of load connected to node $i$ at step $t$                 |
| ${p}_{i,t}^{PV}$                                    | Active power output of PV connected to node $i$ at step $t$                                |
| ${p}_{i,t,\max }^{PV}$                              | Max permissive active power output of PV connected to node $i$ at step $t$                 |
| ${v}_{i,t},{\theta }_{i,t}$                         | Voltage magnitude and phase at node $i$ at step $t$                                        |
| ${n}_{t,{\mathcal{N}}_{i}}^{G}$                     | The number of DGs in the microgrid where node $i$ is located at step $t$                   |
| ${n}_{s,t}$                                         | The number of microgrids at step $t$                                                       |
| ${\rho }_{{ij},t}$                                  | Load ratio of branch ${ij}$ at step $t$                                                    |
| ${i}_{{ij},t}$                                      | Current of branch ${ij}$ at step $t$                                                       |


> Deep reinforcement learning (DRL) is used to solve sequential decision-making problems. It generates control decisions through a feed-forward calculation, reducing the computation time. DRL can avoid modeling some complex physical constraints in power systems (e.g., NetworkX in Python can conveniently calculate the number of roots and energization status). In recent years, it has been widely used in the operation and control of power systems, such as real-time power dispatch \[23,24], load management \[25-27], voltage-var control \[28-30], and emergency control \[31,32]. \[33] formulates DSR problems into Markov decision process (MDP) and uses DRL to search for optimal control policy. However, DRL is vulnerable to the curse of dimensionality and not suitable for only obtained partial observation situations. As an extension of DRL, multi-agent deep reinforcement learning (MADRL) introduces multi-agent game theory, extending MDPs as partially observable stochastic games (POSGs). By facilitating agents to reach a Nash equilibrium, a collaborative strategy is obtained. The distribution network reconfiguration is essentially a switch combination optimization problem. The action space of DRL increases exponentially with the number of switches. MADRL methods mitigate this optimization complexity by segmenting high-dimensional action spaces, which ensures effective training within a limited training time. \[34] develops a MADRL-based DSR approach, but it assumes that agents can only observe local generator and load information. This makes it difficult to model the interaction relationships between agents. \[35] models black-start DGs as agents in the DSR task, and it assumes each agent can observe a fixed number of neighbors. This assumption is not suitable for the situation where the number of neighbors changes due to MGs reconfiguration.

深度强化学习（DRL）用于解决序列决策问题。它通过前馈计算生成控制决策，从而减少计算时间。DRL可以避免对电力系统中一些复杂的物理约束进行建模（例如，Python中的NetworkX可以方便地计算根节点数量和通电状态）。近年来，DRL已广泛应用于电力系统的运行与控制，例如实时电力调度\[23,24]、负荷管理\[25-27]、电压-无功控制\[28-30]以及紧急控制\[31,32]。\[33]将DSR问题表述为马尔可夫决策过程（MDP），并使用DRL搜索最优控制策略。然而，DRL容易受到维度灾难的影响，并且不适合仅获得部分观测信息的情况。作为DRL的扩展，多智能体深度强化学习（MADRL）引入了多智能体博弈论，将MDP扩展为部分可观测随机博弈（POSG）。通过促使智能体达到纳什均衡，获得协作策略。配电网重构本质上是一个开关组合优化问题。DRL的动作空间随着开关数量的增加呈指数增长。MADRL方法通过分割高维动作空间来缓解这种优化复杂性，从而确保在有限的训练时间内进行有效训练。\[34]开发了一种基于MADRL的DSR方法，但假设智能体只能观测到局部发电机和负荷信息。这使得智能体之间的交互关系难以建模。\[35]将黑启动分布式发电机（DG）建模为DSR任务中的智能体，并假设每个智能体可以观测到固定数量的邻居。这一假设不适用于由于微电网重构导致邻居数量变化的情况。

> Three challenges have not been fully addressed when using DRL to solve DSR problems. (1) The load recovery process involves numerous security constraints (e.g., voltage and current limits). Currently, the main idea for dealing with security constraints in DRL is reward shaping, which is adding penalty terms to the reward. In \[36], voltage constraint is converted into a barrier function and the necessity of designing an appropriate barrier function is demonstrated from experimental results. This reward-guidance method makes it difficult to construct the reward signal reasonably and achieve zero constraint violations. (2) In the MADRL, it is necessary for each agent to observe and model other agents. Existing methods usually concatenate the feature of neighbors as input to neural networks (NNs). However, in the DSR task, the action of the switch leads to dynamic MG boundaries and neighboring agent sets, making variations in observation dimensions. Traditional NNs with fixed input dimensions are not applicable. (3) Modern distribution networks often face scenarios involving the integration of new generators and loads. However, training a new policy becomes challenging due to the absence of historical data for these new devices. And considering the problem of low sample efficiency in DRL, the training time required for agents to reach the same level as humans generally exceeds ${60}\mathrm{\;h}$ in the game tasks \[37]. If the agent needs to be retrained every time the environment changes, it will consume a lot of time and computational resources.

在使用深度强化学习（DRL）解决配电系统恢复（DSR）问题时，有三个挑战尚未得到充分解决。（1）负荷恢复过程涉及众多安全约束（如电压和电流限制）。目前，DRL中处理安全约束的主要思路是奖励塑造，即在奖励中添加惩罚项。在\[36]中，电压约束被转化为障碍函数，并通过实验结果证明了设计适当障碍函数的必要性。这种奖励引导方法使得合理构建奖励信号并实现零约束违规变得困难。（2）在多智能体深度强化学习（MADRL）中，每个智能体需要观察并建模其他智能体。现有方法通常将邻居的特征连接起来作为神经网络的输入。然而，在DSR任务中，开关的动作会导致微电网边界和邻居智能体集的动态变化，从而导致观察维度的变化。传统的具有固定输入维度的神经网络不适用。（3）现代配电网络经常面临新发电机和负荷接入的场景。然而，由于这些新设备缺乏历史数据，训练新策略变得具有挑战性。考虑到DRL中样本效率低的问题，智能体在游戏任务中达到与人类相同水平所需的训练时间通常超过${60}\mathrm{\;h}$\[37]。如果每次环境变化都需要重新训练智能体，将会消耗大量时间和计算资源。

> Considering the aforementioned issues, we propose a multi-agent reinforcement learning method in distribution system restoration, as shown in Fig. 1. This method controls switches, distributed PVs, and load pickup to maximize weighted load recovery rate. The DSR problem is modeled as a POSG and solved by an improved QMIX algorithm with the action mask and dynamic agent network. To verify the effectiveness of the proposed method, a DSR simulation verification environment for the IEEE 123-node distribution system was constructed. The main contributions are summarized as follows:

鉴于上述问题，我们提出了一种用于配电系统恢复的多智能体强化学习方法，如图1所示。该方法通过控制开关、分布式光伏和负荷接入，以最大化加权负荷恢复率。配电系统恢复问题被建模为部分可观察随机博弈（POSG），并通过改进的QMIX算法结合动作掩码和动态智能体网络进行求解。为了验证所提出方法的有效性，构建了基于IEEE 123节点配电系统的恢复仿真验证环境。主要贡献总结如下：

> (1) Considering the difficulty in modeling and fast solving for source-net-load collaborative sequential DSR problems, we formulate the problem as a POSG and introduce a model-free MADRL method to obtain the control strategy.

(1) 考虑到源-网-荷协同时序DSR问题的建模和快速求解难度，我们将该问题表述为部分可观测随机博弈（POSG），并引入一种无模型的MADRL方法来获取控制策略。

> (2) An action mask technique is used to filter out actions that violate safety constraints. Furthermore, considering the combinatorial nature of DSR, it can reduce the feasible domain of agent action space and improve learning performance.

(2) 使用动作掩码技术来过滤违反安全约束的动作。此外，考虑到DSR的组合性质，它可以减少智能体动作空间的可行域，从而提高学习性能。

![\<img src="attachments/XCJPCW3Y.jpg" alt="" data-attachment-key="XCJPCW3Y" width="723.4468937875752" height="500" ztype="zimage"> | 723.4468937875752](attachments/XCJPCW3Y.jpg)

> Fig. 1. The framework of DSR based on MADRL method.

图1. 基于MADRL方法的DSR框架。

> (3) Facing the dynamic MG boundaries and newly added devices, an attention-based dynamic agent network (DAN) architecture is proposed, which enables aggregate feature representations of arbitrary-sized neighboring agents.

(3) 面对动态的MG边界和新添加的设备，提出了一种基于注意力的动态代理网络（DAN）架构，该架构能够实现对任意大小的邻近代理的聚合特征表示。

> The rest of this paper is organized as follows. The DSR model is described in Section 2. The details of the proposed DRL algorithm are explained in Section 3. Case studies and conclusions are provided in Section 4 and Section 5.

本文的其余部分组织如下。第2节描述了DSR模型。第3节详细解释了所提出的DRL算法。案例研究和结论分别在第4节和第5节中提供。

> ## 2. Problem formulation

## 2. 问题表述

> ### 2.1. Sequential optimal DSR model

### 2.1. 顺序最优DSR模型

> To better verify the effectiveness of the proposed strategy, the following assumptions are made:

为了更好地验证所提出策略的有效性，做出以下假设：

> (1) After the occurrence of extreme events, the distribution network is unable to obtain power from the transmission network through substations and requires the use of local resources to restore loads.

(1) 在极端事件发生后，配电网无法通过变电站从输电网络获取电力，需要使用本地资源来恢复负荷。

> (2) At the onset of the fault, the faulty line has been isolated, with all switches set to an open position. All loads and PVs are disconnected from the network, and DGs are used as black-start sources.

(2) 故障发生时，故障线路已被隔离，所有开关均处于断开状态。所有负载和光伏系统均与网络断开，分布式发电设备被用作黑启动电源。

> (3) During the recovery process, to facilitate the system operator, only one switch is allowed to be operated per step. All loads and distributed power sources can be controlled in each step.

(3) 在恢复过程中，为了便于系统操作员操作，每一步只允许操作一个开关。每一步都可以控制所有负载和分布式电源。

> (4) According to the response time of current power system automation devices, the time interval of each step is measured in seconds. Throughout the recovery process, the load power remains constant, while the random fluctuations of PV power adhere to a normal distribution.

(4) 根据当前电力系统自动化设备的响应时间，每个步骤的时间间隔以秒为单位进行测量。在整个恢复过程中，负载功率保持不变，而光伏功率的随机波动遵循正态分布。

> Taking load priority into account, the objective function is defined as maximizing the restored power across the entire time horizon. The mathematical model is as follows:

在考虑负荷优先级的情况下，目标函数定义为在整个时间范围内最大化恢复的电力。数学模型如下：

$$
\max \mathop{\sum }\limits_{{t = 1}}^{T}\mathop{\sum }\limits_{{i \in  \mathcal{N}}}{c}_{i} \cdot  {x}_{i,t}^{\text{load }} \cdot  {p}_{i,t}^{\text{load }} \tag{1}
$$

> s.t.

使得

$$
{p}_{i} = {v}_{i,t}\mathop{\sum }\limits_{{\left( {i,j}\right)  \in  \mathcal{E}}}{x}_{{ij},t}^{BR} \cdot  {v}_{j,t} \cdot  \left( {{G}_{ij}\cos {\theta }_{{ij},t} + {B}_{ij}\sin {\theta }_{{ij},t}}\right)  \tag{2}
$$

$$
{p}_{i} = {p}_{i,t}^{G} + {p}_{i,t}^{PV} - {x}_{i,t}^{\text{load }} \cdot  {p}_{i,t}^{\text{load }} \tag{3}
$$

$$
{q}_{i} = {v}_{i,t}\mathop{\sum }\limits_{{\left( {i,j}\right)  \in  \mathcal{E}}}{x}_{{ij},t}^{BR} \cdot  {v}_{j,t} \cdot  \left( {{G}_{ij}\sin {\theta }_{{ij},t} - {B}_{ij}\cos {\theta }_{{ij},t}}\right)  \tag{4}
$$

$$
{q}_{i} = {q}_{i,t}^{G} - {x}_{i,t}^{\text{load }} \cdot  {q}_{i,t}^{\text{load }} \tag{5}
$$

$$
{i}_{{ij},t} = {x}_{{ij},t}^{BR} \cdot  \left( {{v}_{i,t} - {v}_{j,t}}\right)  \cdot  \left( {{G}_{ij} + j{B}_{ij}}\right)  \tag{6}
$$

$$
\mathop{\sum }\limits_{{\left( {ij}\right)  \in  \mathcal{E}}}{x}_{{ij},t}^{BR} = {n}_{b} - {n}_{s,t} \tag{7}
$$

$$
{x}_{i,t}^{\text{node }} \cdot  {V}_{\min } \leq  {v}_{i,t} \leq  {x}_{i,t}^{\text{node }} \cdot  {V}_{\max } \tag{8}
$$

$$
{x}_{{ij},t}^{BR} \cdot  {I}_{{ij},\min } \leq  {i}_{{ij},t} \leq  {x}_{{ij},t}^{BR} \cdot  {I}_{{ij},\max } \tag{9}
$$

$$
{\left( {p}_{i,t}^{G}\right) }^{2} + {\left( {q}_{i,t}^{G}\right) }^{2} \leq  {\left( {S}_{i}^{G}\right) }^{2} \tag{10}
$$

$$
0 \leq  {p}_{i,t}^{PV} \leq  {x}_{i,t - 1}^{\text{node }} \cdot  {p}_{i,t,\max }^{PV} \tag{11}
$$

$$
{x}_{i,t - 1}^{\text{load }} \leq  {x}_{i,t}^{\text{load }} \leq  {x}_{i,t - 1}^{\text{node }} \tag{12}
$$

$$
{x}_{{ij},t - 1}^{SW} \leq  {x}_{{ij},t}^{SW} \leq  {x}_{i,t - 1}^{\text{node }} + {x}_{j,t - 1}^{\text{node }} \tag{13}
$$

$$
\mathop{\sum }\limits_{{i \in  \mathcal{N}}}{x}_{i,t}^{\text{load }} \cdot  {p}_{i,t}^{\text{load }} - \mathop{\sum }\limits_{{i \in  \mathcal{N}}}{x}_{i,t}^{\text{load }} \cdot  {p}_{i,t - 1}^{\text{load }} \leq  \alpha  \tag{14}
$$

> Eq. (1) is the objective function. Eqs. (2)-(6) are power flow constraints for distribution networks. Eq. (7) is the radial topology constraint. The radial distribution network can be regarded as a tree structure in graph theory. It must satisfy the condition that the total number of edges is equal to the number of nodes minus the number of roots, and the number of root nodes is equal to the number of MGs. Eq. (8) and (9) are node voltage and line current constraints. Eq. (10) and (11) are DG and PV output constraints. Eq. (12) is the load recovery status constraint. The load can only be restored when the connected nodes are energized, and after restoration, it is not allowed to power off again. Eq. (13) is the switch operation constraint. It is not allowed to close when the both ends of the switch are not energized. And after closing, to prevent the load from powering off, it is not allowed to open again. Eq. (14) is the maximum recovery power allowed per step constraint.

式（1）为目标函数。式（2）-（6）为配电网的潮流约束。式（7）为辐射状拓扑约束。辐射状配电网可视为图论中的树结构，必须满足边总数等于节点数减去根节点数，且根节点数等于微电网（MG）数量的条件。式（8）和（9）为节点电压和线路电流约束。式（10）和（11）为分布式电源（DG）和光伏（PV）输出约束。式（12）为负荷恢复状态约束。负荷只能在连接的节点通电时恢复，且恢复后不允许再次断电。式（13）为开关操作约束。开关两端未通电时不允许闭合，且闭合后为防止负荷断电，不允许再次断开。式（14）为每步允许的最大恢复功率约束。

> It can be seen that the DSR is a mixed-integer non-linear and non-convex dynamic programming problem, which makes it difficult to find solutions efficiently through mathematical optimization methods. Therefore, we convert the sequential DSR model into a cooperative multi-agent game and employ data-driven MADRL methods to solve it.

可以看出，DSR是一个混合整数非线性非凸动态规划问题，这使得通过数学优化方法高效求解变得困难。因此，我们将顺序DSR模型转换为协作多智能体博弈，并采用数据驱动的MADRL方法来解决它。

> ### 2.2. Basics of POSGs

### 2.2. POSG 基础知识

> POSGs model the dynamic interactions among multiple agents and their environment. For POSGs with $N$ agents,a tuple $\left( {\mathcal{N},\mathcal{S},{\left\lbrack {\mathcal{O}}_{i}\right\rbrack }_{N},{\left\lbrack {\mathcal{A}}_{i}\right\rbrack }_{N},{\left\lbrack {\mathcal{R}}_{i}\right\rbrack }_{N},\gamma }\right)$ can be used to define them. Among them, $\mathcal{N}$ is the set of agents; $\mathcal{S}$ is the set of states; ${\mathcal{O}}_{i}$ is the observation space of agent $i;{\mathcal{A}}_{i}$ is the action space of agent $i;{\mathcal{R}}_{i} : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ is an immediate reward for agent $i;\gamma \in \left\lbrack {0,1}\right\rbrack$ is a discount factor used by agents to balance immediate and future rewards.

POSGs 模拟了多个智能体与其环境之间的动态交互。对于具有 $N$ 个智能体的 POSGs，可以使用元组 $\left( {\mathcal{N},\mathcal{S},{\left\lbrack {\mathcal{O}}_{i}\right\rbrack }_{N},{\left\lbrack {\mathcal{A}}_{i}\right\rbrack }_{N},{\left\lbrack {\mathcal{R}}_{i}\right\rbrack }_{N},\gamma }\right)$ 来定义它们。其中，$\mathcal{N}$ 是智能体的集合；$\mathcal{S}$ 是状态的集合；${\mathcal{O}}_{i}$ 是智能体 $i;{\mathcal{A}}_{i}$ 的观察空间；$i;{\mathcal{R}}_{i} : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ 是智能体 $i;\gamma \in \left\lbrack {0,1}\right\rbrack$ 的即时奖励；**8** 是智能体用于平衡即时奖励和未来奖励的折扣因子。

> Each state $s \in \mathcal{S}$ describes the global information of the environment. ${o}_{i} \in {\mathcal{O}}_{i}$ describes the local observation of agent $i$ ,which can be written as follows: ${o}_{i,t} = \left\lbrack {{o}_{i,t}^{env},{o}_{i,t}^{1},\ldots ,{o}_{i,t}^{i - 1},{o}_{i,t}^{i + 1},\ldots ,{o}_{i,t}^{n}}\right\rbrack .{o}_{i,t}^{env}$ describes the observation of agent $i$ on the environment,and ${o}_{i,t}^{j}$ describes the unique observation of agent $i$ on agent $j$ (for example,in the game task, it can be distance and orientation between two agents).

每个状态$s \in \mathcal{S}$描述了环境的全局信息。${o}_{i} \in {\mathcal{O}}_{i}$描述了智能体$i$的局部观察，可以表示如下：${o}_{i,t} = \left\lbrack {{o}_{i,t}^{env},{o}_{i,t}^{1},\ldots ,{o}_{i,t}^{i - 1},{o}_{i,t}^{i + 1},\ldots ,{o}_{i,t}^{n}}\right\rbrack .{o}_{i,t}^{env}$描述了智能体$i$对环境的观察，而${o}_{i,t}^{j}$描述了智能体$i$对智能体$j$的独特观察（例如，在游戏任务中，可以是两个智能体之间的距离和方向）。

> State-action value function represents the expected reward received by agent $i$ when executing a joint action $\mathbf{a} = \left\lbrack {{\mathbf{a}}_{1},\ldots ,{\mathbf{a}}_{n}}\right\rbrack$ . As the expected reward depends on the actions chosen by agents, the value function is related to the strategy.

状态-动作价值函数表示代理$i$在执行联合动作$\mathbf{a} = \left\lbrack {{\mathbf{a}}_{1},\ldots ,{\mathbf{a}}_{n}}\right\rbrack$时获得的预期奖励。由于预期奖励取决于代理选择的动作，价值函数与策略相关。

$$
{Q}_{i}\left( \mathbf{a}\right)  = E\left\lbrack  {\left. {\mathop{\sum }\limits_{{t = 1}}^{\infty }{\gamma }^{t - 1}{r}_{i,t}}\right| \;{\mathbf{a}}_{t} = \mathbf{a}}\right\rbrack   \tag{15}
$$

> If the joint strategy ${\pi }^{ * } = \left\lbrack {{\pi }_{1}^{ * },\ldots ,{\pi }_{n}^{ * }}\right\rbrack$ meets

如果联合策略${\pi }^{ * } = \left\lbrack {{\pi }_{1}^{ * },\ldots ,{\pi }_{n}^{ * }}\right\rbrack$满足

$$
\mathop{\sum }\limits_{{{a}_{1},\cdots ,{a}_{n} \in  {\mathcal{A}}_{1},\cdots ,{\mathcal{A}}_{n}}}{Q}_{i}\left( \mathbf{a}\right) {\pi }_{1}^{ * }\left( {\mathbf{a}}_{1}\right) \cdots {\pi }_{i}^{ * }\left( {\mathbf{a}}_{i}\right) \cdots {\pi }_{n}^{ * }\left( {\mathbf{a}}_{n}\right)  \geq
$$

$$
\mathop{\sum }\limits_{{{a}_{1},\cdots ,{a}_{n} \in  {\mathcal{A}}_{1},\cdots ,{\mathcal{A}}_{n}}}{Q}_{i}\left( \mathbf{a}\right) {\pi }_{1}^{ * }\left( {\mathbf{a}}_{1}\right) \cdots {\pi }_{i}\left( {\mathbf{a}}_{i}\right) \cdots {\pi }_{n}^{ * }\left( {\mathbf{a}}_{n}\right)  \tag{16}
$$

$$
i = 1,\ldots ,n
$$

> where ${\pi }_{i}^{ * }\left( {\mathbf{a}}_{i}\right)$ is the probability of agent $i$ choosing action ${\mathbf{a}}_{i}$ under the Nash equilibrium (NE) strategy; ${\pi }_{i}$ is any strategy of agent $i$ from the strategy space.

其中 ${\pi }_{i}^{ * }\left( {\mathbf{a}}_{i}\right)$ 是代理 $i$ 在纳什均衡（NE）策略下选择动作 ${\mathbf{a}}_{i}$ 的概率；${\pi }_{i}$ 是代理 $i$ 从策略空间中选择的任何策略。

> Formula (16) represents that the multi-agent system achieves a Nash equilibrium in stochastic games, where all agents cannot obtain higher rewards by changing its own policy given that other agent continue using their NE policy. The purpose of multi-agent reinforcement learning is to find the optimal policy ${\pi }_{i}^{ * }$ for each agent to maximize ${Q}_{i}\left( \mathbf{a}\right)$ and reach a Nash equilibrium.

公式（16）表示多智能体系统在随机博弈中达到了纳什均衡，其中所有智能体在给定其他智能体继续使用其纳什均衡策略的情况下，无法通过改变自身策略来获得更高的奖励。多智能体强化学习的目的是为每个智能体找到最优策略 ${\pi }_{i}^{ * }$，以最大化 ${Q}_{i}\left( \mathbf{a}\right)$ 并达到纳什均衡。

> ### 2.3. Model the DSR problem as a POSG

### 2.3. 将DSR问题建模为POSG

> We model a POSG for DSR by making sequential decisions on switches, distributed PVs, and load pickup. Its main components are defined as follows:

我们通过依次对开关、分布式光伏和负荷恢复做出决策，为DSR建模了一个POSG。其主要组成部分定义如下：

> (1) Agent: Each PV and load are modeled as an agent. All switches are integrated into a switch agent.

(1) 代理：每个光伏和负载都被建模为一个代理。所有开关被集成到一个开关代理中。

> (2) State: The state includes the node voltage, line loading rate, switch status,restored load power,PV and DG generation at step $t$ ,and the load demand power and maximum PV generation at step $t + 1$ ,as well as faulty line ID $\left( {\mathbf{F}}_{t}\right)$ and load priority factor(C).

(2) 状态：状态包括节点电压、线路负载率、开关状态、恢复的负载功率、步骤$t$时的光伏和分布式发电出力，以及步骤$t + 1$时的负载需求功率和最大光伏出力，还包括故障线路ID $\left( {\mathbf{F}}_{t}\right)$和负载优先级因子(C)。

$$
{\mathbf{s}}_{t} = \left\lbrack  {{\mathbf{v}}_{t},{\mathbf{\rho }}_{t},{\mathbf{x}}_{t}^{SW},{\widetilde{\mathbf{p}}}_{t}^{\text{load }},{\mathbf{p}}_{t}^{PV},{\mathbf{p}}_{t}^{G},{\mathbf{p}}_{t + 1}^{\text{load }},{\mathbf{p}}_{t + 1,\max }^{PV},{\mathbf{F}}_{t},\mathbf{C}}\right\rbrack   \tag{17}
$$

$$
\text{And}{\widetilde{\mathbf{p}}}_{t}^{\text{load }} = {\left( {\mathbf{x}}_{t}^{\text{load }}\right) }^{T} \cdot  {\mathbf{p}}_{t}^{\text{load }}\text{.}
$$

> (3) Observation: Due to switches determining the topology of the network and power capacity of each MG, for the switch agent, use global information as observation,that is ${o}_{t}^{SW} = {s}_{t}$ .

(3) 观察：由于开关决定了网络的拓扑结构和每个微电网的功率容量，对于开关代理，使用全局信息作为观察，即 ${o}_{t}^{SW} = {s}_{t}$。

> For PV and load agents, only observations from the local environment and other agents in the same MG need to be obtained.

对于光伏和负载代理，只需获取来自本地环境以及同一微电网中其他代理的观测数据。

$$
{o}_{i,t}^{env} = \left\lbrack  \begin{array}{l} \mathop{\sum }\limits_{{i \in  {\mathcal{N}}_{i}}}{\widetilde{p}}_{i,t}^{load},\mathop{\sum }\limits_{{i \in  {\mathcal{N}}_{i}}}{p}_{i,t}^{PV},\mathop{\sum }\limits_{{i \in  {\mathcal{N}}_{i}}}{p}_{i,t}^{G}, \\  \mathop{\sum }\limits_{{i \in  {\mathcal{N}}_{i}}}{p}_{i,t + 1}^{load},\mathop{\sum }\limits_{{i \in  {\mathcal{N}}_{i}}}{p}_{i,t + 1,\max }^{PV},{n}_{t,{\mathcal{N}}_{i}}^{G} \end{array}\right\rbrack   \tag{18}
$$

$$
{o}_{i,t}^{j} = \left\lbrack  {{v}_{j,t},{p}_{j,t}^{PV},{p}_{j,t + 1,\max }^{PV}}\right\rbrack  ,j \in  {\mathcal{N}}_{i}\text{and}{PV} \tag{19}
$$

$$
{o}_{i,t}^{j} = \left\lbrack  {{v}_{j,t},{\widetilde{p}}_{j,t}^{\text{load }},{p}_{j,t + 1}^{\text{load }},{c}_{j}}\right\rbrack  ,j \in  {\mathcal{N}}_{i}\text{and load} \tag{20}
$$

> (4) Action: For the switch agent, it determines which switch will be closed in the current step. Assuming there are $n$ switches in the network,the action dimension of the switch agent is $n + 1$ . The last dimension corresponds to the situation where the states of all switches remain unchanged. This ensures that only one switch is allowed to be operated per step, and reduces the action space from the original switch combination ${2}^{n}$ to $n + 1$ .

(4) 动作：对于开关代理，它决定在当前步骤中关闭哪个开关。假设网络中有 $n$ 个开关，开关代理的动作维度为 $n + 1$。最后一个维度对应于所有开关状态保持不变的情况。这确保了每一步只允许操作一个开关，并将动作空间从原始的开关组合 ${2}^{n}$ 减少到 $n + 1$。

> For the PV agent,the action space of $\left\lbrack {0,1}\right\rbrack$ is divided into 11 dimensions with a granularity of 0.1 . The PV output at each step is ${p}_{i,t}^{PV} = {a}_{i,t}^{PV} \cdot {p}_{i,t,\max }^{PV}$ ,and ${a}_{i,t}^{PV}$ is the agent action generated by the neural network. It can handle dynamic upper and lower limits to satisfy the PV output constraint.

对于PV代理，$\left\lbrack {0,1}\right\rbrack$的动作空间被划分为11个维度，粒度为0.1。每一步的PV输出为${p}_{i,t}^{PV} = {a}_{i,t}^{PV} \cdot {p}_{i,t,\max }^{PV}$，${a}_{i,t}^{PV}$是由神经网络生成的代理动作。它可以处理动态的上下限以满足PV输出约束。

> For load agents, it determines whether to restore the load at each step.

对于负载代理，它决定是否在每一步恢复负载。

> (5) Reward: DSR is a collaborative task, so all agents share the same reward. The goal of DSR is to efficiently restore the critical load while adhering to security constraints during the recovery process. Therefore, rewards must consider the recovered load power ratio at each step based on load priority.

(5) 奖励：DSR 是一项协作任务，因此所有代理共享相同的奖励。DSR 的目标是在恢复过程中高效恢复关键负载，同时遵守安全约束。因此，奖励必须基于负载优先级考虑每一步的恢复负载功率比。

$$
{r}_{t}^{\text{restore }} = \mathop{\sum }\limits_{{i \in  \mathcal{N}}}{c}_{i} \cdot  \left( {{x}_{i,t}^{\text{load }} - {x}_{i,t - 1}^{\text{load }}}\right)  \cdot  {p}_{i,t}^{\text{load }}/\mathop{\sum }\limits_{{i \in  \mathcal{N}}}{c}_{i} \cdot  {p}_{i,t}^{\text{load }} \tag{21}
$$

> At the same time, the reward needs to impose certain penalties for actions resulting in node voltage exceeding limits and line overload. The corresponding penalties are as follows:

同时，奖励需要对导致节点电压超限和线路过载的行为施加一定的惩罚。相应的惩罚如下：

$$
{r}_{t}^{\text{voltage }} =  - \mathop{\sum }\limits_{{i \in  \mathcal{N}}}{x}_{i,t}^{\text{node }} \cdot  \max \left( {0,{v}_{i,t} - {1.05},{0.95} - {v}_{i,t}}\right)  \tag{22}
$$

$$
{r}_{t}^{rho} =  - \mathop{\sum }\limits_{{\left( {i,j}\right)  \in  \mathcal{E}}}{x}_{{ij},t}^{BR} \cdot  \max \left( {0,{\rho }_{{ij},t} - 1}\right)  \tag{23}
$$

> If the action causes the output of DGs to exceed the limit, it will be considered a failure of the task. The current episode will be terminated and a negative reward will be given to agents. In summary, the reward is

如果该操作导致分布式发电机的输出超过限制，则视为任务失败。当前回合将被终止，并向代理给予负奖励。总之，奖励是

$$
{r}_{t} = \left\{  \begin{array}{l} {r}^{\text{done }},\text{ if done } \\  {\alpha }_{r} \cdot  {r}_{t}^{\text{restore }} + {\alpha }_{v} \cdot  {r}_{t}^{\text{voltage }} + {\alpha }_{l} \cdot  {r}_{t}^{\text{rho }},\text{ else } \end{array}\right.  \tag{24}
$$

> ${\alpha }_{r},{\alpha }_{v},{\alpha }_{l}$ is the weight coefficient of each item.

${\alpha }_{r},{\alpha }_{v},{\alpha }_{l}$ 是每个项目的权重系数。

> ## 3. Technical methods

## 3. 技术方法

> ### 3.1. Dynamic agent number network

### 3.1. 动态代理数量网络

> Compared to game tasks, the uncertainty of initial faults and distributed resources in DSR make the scenarios more diverse. Also, the training complexity of DRL will increase exponentially with the growing number of agents \[38]. However, \[39] indicates that there is sparse interaction between agents. Thus, it is crucial to establish a concise and effective observation space for agents. In the DSR, agents in the same MG work together to maintain power balance and do not have any electrical connections with agents outside the MG. Therefore, there is no requirement to observe the agents outside the MG. Simultaneously, owing to the actions of switches, the group of agents will dynamic alterations. On the other hand, with the continuous development of modern distribution networks, new load and PV connections are often encountered. In the POSG, this corresponds to variations in agent scale, observation space, and action space. However, it is challenging for the optimal strategy approximated by deep neural networks to adapt to dynamic changes in the group size. Considering the aforementioned scenarios, wherein the input dimension of the neural network varies due to changes in topology and device quantity. We propose a dynamic agent network that maps the variable input dimension to a fixed-dimensional latent space based on the attention mechanism. This structure ensures that each agent only interacts with others within the same MG during each decision-making step. It can alleviate the situation of input dimension explosion and information redundancy caused by excessive observations on all agents.

与游戏任务相比，DSR中初始故障和分布式资源的不确定性使得场景更加多样化。此外，随着代理数量的增加，DRL的训练复杂性将呈指数级增长\[38]。然而，\[39]指出代理之间的交互是稀疏的。因此，为代理建立一个简洁且有效的观察空间至关重要。在DSR中，同一微电网（MG）内的代理共同协作以维持电力平衡，并且与MG外的代理没有任何电气连接。因此，无需观察MG外的代理。同时，由于开关的操作，代理组会动态变化。另一方面，随着现代配电网络的不断发展，新的负载和光伏连接经常出现。在POSG中，这对应于代理规模、观察空间和动作空间的变化。然而，由深度神经网络近似的最优策略难以适应组规模的动态变化。考虑到上述场景，其中神经网络的输入维度由于拓扑结构和设备数量的变化而变化，我们提出了一种基于注意力机制的动态代理网络，将可变输入维度映射到固定维度的潜在空间。该结构确保每个代理在每个决策步骤中仅与同一MG内的其他代理交互。它可以缓解由于对所有代理进行过多观察而导致的输入维度爆炸和信息冗余的情况。

> Fig. 2 shows the network architecture of DAN. The agent observation can be divided into two parts: information about the environment and other agents. The left half of DAN is the environmental information encoder, which is used to extract the environmental features. The dimension of ${o}_{i,t}^{env}$ does not dynamically change. The right half of DAN is the agent interaction information encoder, which is used to extract the features of interactive agents. The dimension of ${o}_{i,t}^{j}$ is generally fixed, but the number of interactive agents is dynamically changing in the DSR task.

图2展示了DAN的网络架构。代理的观察可以分为两部分：关于环境的信息和其他代理的信息。DAN的左半部分是环境信息编码器，用于提取环境特征。${o}_{i,t}^{env}$的维度不会动态变化。DAN的右半部分是代理交互信息编码器，用于提取交互代理的特征。${o}_{i,t}^{j}$的维度通常是固定的，但在DSR任务中，交互代理的数量是动态变化的。

> In the environmental information encoder,the output ${\mathbf{h}}_{i} = {g}_{i}\left( {o}_{i,t}^{\text{env }}\right)$ represents the embedding of the environmental information by agent $i$ .

在环境信息编码器中，输出${\mathbf{h}}_{i} = {g}_{i}\left( {o}_{i,t}^{\text{env }}\right)$表示代理$i$对环境信息的嵌入。

> In the interaction information encoder,the output ${f}_{i}\left( {\mathbf{o}}_{i,t}^{j}\right) ,i \neq j$ represents the embedding of the observation information on agent $j$ by agent $i$ . The attention mechanism is used to learn the importance weights for each interactive agent through a data-driven approach. It computes the weighted sum of agent features to obtain a representation that remains invariant dimensions. The attention mechanism aggregates the embedding of agent $i$ for each agent $j$ in the same MG.

在交互信息编码器中，输出${f}_{i}\left( {\mathbf{o}}_{i,t}^{j}\right) ,i \neq j$表示代理$i$对代理$j$的观察信息的嵌入。注意力机制通过数据驱动的方法学习每个交互代理的重要性权重。它计算代理特征的加权和，以获得保持维度不变的表示。注意力机制为同一多代理游戏（MG）中的每个代理$j$聚合代理$i$的嵌入。

$$
{\mathbf{v}}_{i,t} = \mathop{\sum }\limits_{{\forall j \in  {\mathcal{N}}_{i},j \neq  i}}{\alpha }_{i,t}^{j} \cdot  {f}_{i}\left( {\mathbf{o}}_{i,t}^{j}\right)  \tag{25}
$$

![\<img src="attachments/5L88J3F4.jpg" alt="" data-attachment-key="5L88J3F4" width="659.6715328467153" height="500" ztype="zimage"> | 659.6715328467153](attachments/5L88J3F4.jpg)

> Fig. 2. The network structure of DAN.

图2. DAN的网络结构。

> The formula for calculating the weighting factor is as follows:

计算权重因子的公式如下：

$$
{\alpha }_{i,t}^{j} = {\operatorname{softmax}}_{j}\left( {\beta }_{i,t}^{j}\right)  = \frac{\exp \left( {\beta }_{i,t}^{j}\right) }{\mathop{\sum }\limits_{{\forall j \in  {\mathcal{N}}_{i},j \neq  i}}\exp \left( {\beta }_{i,t}^{j}\right) } \tag{26}
$$

$$
{\beta }_{i,t}^{j} = {f}_{i}^{T}\left( {o}_{i,t}^{j}\right) {\mathbf{W}}_{k}^{T}{\mathbf{W}}_{q}{f}_{i}\left( {o}_{i,t}^{i}\right)  \tag{27}
$$

> ${\mathbf{W}}_{k}$ and ${\mathbf{W}}_{q}$ are parameters for automatic learning. ${\mathbf{W}}_{k}$ maps ${f}_{i}\left( {\mathbf{o}}_{i,t}^{j}\right)$ to key,and ${\mathbf{W}}_{q}$ maps ${f}_{i}\left( {\mathbf{o}}_{i,t}^{i}\right)$ to query. ${\beta }_{i,t}^{j}$ calculates the correlation between the embedding vectors of agent $i$ and agent $j$ through the inner product. Then, ${\beta }_{i,t}^{j}$ is normalized to obtain attention weights ${\alpha }_{i,t}^{j}$ through softmax function. Using attention mechanisms, it becomes possible to model the interaction between agent $i$ and any number of other agents, allowing for the transformation of the variable-length dimensional observation into the fixed-length feature vector. The interactive information encoder ${f}_{i}\left( \cdot \right)$ is shared for different agent $j,i \neq j$ . Thus,the network parameters that need to be learned remain constant regardless of the number of agents. This not only simplifies the training process but also enhances the adaptability of agents to dynamic environments.

${\mathbf{W}}_{k}$ 和 ${\mathbf{W}}_{q}$ 是自动学习的参数。${\mathbf{W}}_{k}$ 将 ${f}_{i}\left( {\mathbf{o}}_{i,t}^{j}\right)$ 映射为键，${\mathbf{W}}_{q}$ 将 ${f}_{i}\left( {\mathbf{o}}_{i,t}^{i}\right)$ 映射为查询。${\beta }_{i,t}^{j}$ 通过内积计算代理 $i$ 和代理 $j$ 的嵌入向量之间的相关性。然后，${\beta }_{i,t}^{j}$ 通过 softmax 函数进行归一化，得到注意力权重 ${\alpha }_{i,t}^{j}$。使用注意力机制，可以建模代理 $i$ 与任意数量其他代理之间的交互，从而将可变长度的维度观察转换为固定长度的特征向量。交互信息编码器 ${f}_{i}\left( \cdot \right)$ 在不同代理 $j,i \neq j$ 之间共享。因此，无论代理数量如何，需要学习的网络参数保持不变。这不仅简化了训练过程，还增强了代理对动态环境的适应性。

> Finally, DAN will concatenate the embedding of the environment and other agents as input to the subsequent layer of the neural network, outputting the final Q value $Q\left( {o}_{i,t}\right) = {F}_{i}\left( {{\mathbf{h}}_{i,t}\parallel {\mathbf{v}}_{i,t}}\right)$ or action $P\left( {\cdot \mid {o}_{i,t}}\right) =$ ${F}_{i}\left( {{\mathbf{h}}_{i,t}\parallel {\mathbf{v}}_{i,t}}\right)$ .

最后，DAN 会将环境和其他智能体的嵌入连接起来，作为神经网络的下一层的输入，输出最终的 Q 值 $Q\left( {o}_{i,t}\right) = {F}_{i}\left( {{\mathbf{h}}_{i,t}\parallel {\mathbf{v}}_{i,t}}\right)$ 或动作 $P\left( {\cdot \mid {o}_{i,t}}\right) =$ ${F}_{i}\left( {{\mathbf{h}}_{i,t}\parallel {\mathbf{v}}_{i,t}}\right)$。

> ### 3.2. Solve the POSG via QMIX

### 3.2. 通过QMIX解决POSG

> Currently, MADRL is mainly divided into two categories: learning cooperation and learning communication \[40]. The former uses a framework of centralized training and decentralized execution. It enables agents to learn collaborative control strategies by sharing global information during the training process. In the execution phase, each agent independently makes decisions based on their local observation. The latter maximizes their shared utility by designing communication mechanisms between agents. This paper employs the QMIX algorithm, which is based on learning cooperation, and considers information from other agents by using the DAN architecture to better achieve collaboration between agents.

目前，MADRL主要分为两类：学习合作和学习通信\[40]。前者采用集中训练和分散执行的框架，通过在训练过程中共享全局信息，使智能体能够学习协作控制策略。在执行阶段，每个智能体根据其局部观察独立做出决策。后者通过设计智能体之间的通信机制来最大化它们的共享效用。本文采用基于学习合作的QMIX算法，并通过使用DAN架构考虑其他智能体的信息，以更好地实现智能体之间的协作。

> Fig. 3 illustrates the network structure created by the QMIX algorithm, comprising three networks.

图3展示了由QMIX算法创建的网络结构，包含三个网络。

> Due to the DSR belonging to a cooperative task, all agents share the same reward. The QMIX algorithm based on value decomposition is utilized to address multi-agent credit assignment issues. During the training, it acquires global information to calculate the joint action value ${Q}_{\text{tot }}\left( {o,a}\right)$ . During the execution,the agent chooses the action with the highest local action value ${Q}_{i}\left( {{o}_{i},{a}_{i}}\right)$ based on its local observations, respectively. The QMIX algorithm adheres to the fundamental paradigm of the POSG by training agents to reach a Nash equilibrium state. Selecting the optimal joint action based on the global action value is equivalent to each agent individually choosing their best actions based on their local action value and combining them. That is

由于DSR属于协作任务，所有智能体共享相同的奖励。基于价值分解的QMIX算法被用于解决多智能体信用分配问题。在训练过程中，它获取全局信息以计算联合动作值${Q}_{\text{tot }}\left( {o,a}\right)$。在执行过程中，智能体根据各自的局部观察选择具有最高局部动作值${Q}_{i}\left( {{o}_{i},{a}_{i}}\right)$的动作。QMIX算法遵循POSG的基本范式，通过训练智能体达到纳什均衡状态。基于全局动作值选择最优联合动作等同于每个智能体根据其局部动作值独立选择最佳动作并将其组合。即

$$
\arg \mathop{\max }\limits_{\mathbf{a}}{Q}_{\text{tot }}\left( {\mathbf{o},\mathbf{a}}\right)  = \left( \begin{matrix} \arg \mathop{\max }\limits_{{\mathbf{a}}_{1}}{Q}_{1}\left( {{\mathbf{o}}_{1},{\mathbf{a}}_{1};{\mathbf{\phi }}_{1}}\right) \\  \vdots \\  \arg \mathop{\max }\limits_{{\mathbf{a}}_{n}}{Q}_{n}\left( {{\mathbf{o}}_{n},{\mathbf{a}}_{n};{\mathbf{\phi }}_{n}}\right)  \end{matrix}\right)  \tag{28}
$$

![\<img src="attachments/LWUJHVJZ.jpg" alt="" data-attachment-key="LWUJHVJZ" width="1215.4882154882155" height="500" ztype="zimage"> | 1215.4882154882155](attachments/LWUJHVJZ.jpg)

> Fig. 3. The structure of the QMIX network.

图3. QMIX网络的结构。

> ${\phi }_{i}$ is the neural network parameter of agent $i$ .

${\phi }_{i}$ 是代理 $i$ 的神经网络参数。

> The QMIX algorithm deduces the essential conditions for the aforementioned assumptions and translates it into monotonicity constraints.

QMIX算法推导出上述假设的基本条件，并将其转化为单调性约束。

$$
\frac{\partial {Q}_{tot}}{\partial {Q}_{i}} \geq  0,\forall i \in  \{ 1,2,\ldots ,n\}  \tag{29}
$$

> Agent network: A Q-network of a single agent that takes local observations and previous actions as inputs,and generates its ${Q}_{i}$ as outputs.

智能体网络：一个单一智能体的Q网络，它以局部观察和先前的动作为输入，并生成其${Q}_{i}$作为输出。

> Mixing network: The mixing network receives the local $Q$ values from each agent network as input and outputs the global $Q$ values, shown as Eq. (30). To ensure compliance with the monotonicity constraints, the weight parameters of the mixing network must be non-negative.

混合网络：混合网络接收来自每个智能体网络的局部$Q$值作为输入，并输出全局$Q$值，如公式(30)所示。为确保满足单调性约束，混合网络的权重参数必须为非负数。

$$
{Q}_{\text{tot }}\left( {\mathbf{o},\mathbf{a}}\right)  = {Q}_{\text{mix }}\left( {{Q}_{1},\ldots ,{Q}_{n};\mathbf{w},\mathbf{b}}\right) ,\mathbf{w} > 0 \tag{30}
$$

> $\mathbf{w},\mathbf{b}$ are the weight and bias of the mixing network,respectively.

$\mathbf{w},\mathbf{b}$ 分别是混合网络的权重和偏置。

> Hypernetwork: The hypernetwork is employed to generate the parameters of the mixing network, including weights and biases. It takes the global state as input. The hypernetwork ensures the nonnegativity of weights by using linear networks and absolute value activation functions. Furthermore, the hypernetwork accepts global states as inputs, enabling the parameters of the mixing network to dynamically adapt to different states. This flexibility facilitates learning the relationship between the states and global Q-values.

超网络：超网络用于生成混合网络的参数，包括权重和偏置。它以全局状态作为输入。超网络通过使用线性网络和绝对值激活函数来确保权重的非负性。此外，超网络接受全局状态作为输入，使混合网络的参数能够动态适应不同的状态。这种灵活性有助于学习状态与全局Q值之间的关系。

> The loss function of the QMIX algorithm is

QMIX算法的损失函数是

$$
L\left( \theta \right)  = \mathop{\sum }\limits_{{i = 1}}^{b}\left\lbrack  {\left( {y}_{i}^{tot} - {Q}_{tot}\left( o,a,s;\phi \right) \right) }^{2}\right\rbrack   \tag{31}
$$

> And ${y}_{i}^{\text{tot }} = r + \gamma \mathop{\max }\limits_{{a}^{\prime }}{Q}_{\text{tot }}\left( {{o}^{\prime },{a}^{\prime },{s}^{\prime };\widehat{\phi }}\right) ,\widehat{\phi }$ is the target network parameter. The parameters of the behavior network are updated using a random gradient descent algorithm through the loss function. The target network periodically copies parameters from the behavior network to improve the stability of the training. In this paper, we use the soft update method.

其中 ${y}_{i}^{\text{tot }} = r + \gamma \mathop{\max }\limits_{{a}^{\prime }}{Q}_{\text{tot }}\left( {{o}^{\prime },{a}^{\prime },{s}^{\prime };\widehat{\phi }}\right) ,\widehat{\phi }$ 是目标网络参数。行为网络的参数通过损失函数使用随机梯度下降算法进行更新。目标网络定期从行为网络复制参数，以提高训练的稳定性。在本文中，我们使用软更新方法。

$$
\widehat{\mathbf{\phi }} \leftarrow  \xi \mathbf{\phi } + \left( {1 - \xi }\right) \widehat{\mathbf{\phi }} \tag{32}
$$

> $\xi$ is the soft update coefficient.

$\xi$ 是软更新系数。

> ### 3.3. Agent action mask mechanism

### 3.3. 代理动作掩码机制

> Compared to the application of DRL in fields such as robot control and games, power systems present greater complexity due to their complex and coupled constraints. The reward shaping method struggles to effectively manage numerous constraints. Excessive penalties tend to make the agent overly conservative in the exploration process, while overly lenient penalties fail to adequately deter illicit actions. While the agent may eventually learn the policy that adheres to the constraints, it cannot ensure that actions remain constrained throughout the training process. This inability to prevent the agent from repeatedly sampling illegal actions within the action space can lead to invalid exploration. \[41] presents a concise method for the illegal action mask technique, which adds a mask layer at the end of the neural network, as shown in Eq. (33).

与深度强化学习（DRL）在机器人控制和游戏等领域的应用相比，电力系统由于其复杂且耦合的约束条件，呈现出更大的复杂性。奖励塑造方法难以有效管理众多约束。过度的惩罚往往会使智能体在探索过程中过于保守，而过轻的惩罚则无法充分阻止非法行为。虽然智能体最终可能学会遵守约束的策略，但它无法确保在整个训练过程中行为始终受到约束。这种无法防止智能体在动作空间中反复采样非法动作的情况可能导致无效的探索。\[41] 提出了一种简洁的非法动作掩码技术方法，该方法在神经网络的末端添加了一个掩码层，如公式（33）所示。

$$
\operatorname{mask}\left( {Q\left( {o,a}\right) }\right)  = \left\{  \begin{array}{l} Q\left( {o,{a}_{i}}\right) ,\text{ if }{a}_{i}\text{ is valid } \\  M,\text{ otherwise } \end{array}\right.  \tag{33}
$$

![\<img src="attachments/VY84Q3CX.jpg" alt="" data-attachment-key="VY84Q3CX" width="1326.5682656826568" height="500" ztype="zimage"> | 1326.5682656826568](attachments/VY84Q3CX.jpg)

> Fig. 4. Design of mask layer.

图4. 掩模层设计。

> Table 1

表1

> The category of constraints.

约束类别。

| <!-- --> | <!-- --> | <!-- --> |
| ------------------------------- | --------------- | --------------- |
| Constraints                     | Hard constraint | Soft constraint |
| Power balance                   | ✓               |                 |
| Radial topology                 | ✓               |                 |
| Node voltage                    |                 | ✓               |
| Line current                    |                 | ✓               |
| DG output                       |                 | ✓               |
| PV output                       | ✓               |                 |
| Load recovery status            | ✓               |                 |
| Switch operation                | ✓               |                 |
| Maximum recovery power per step |                 | ✓               |
|                                 |                 |                 |


> $M$ is a large negative number (for example $M = - 1 \times {10}^{8}$ ).

$M$ 是一个很大的负数（例如 $M = - 1 \times {10}^{8}$）。

> During the computation of the Q-value of each agent, it assigns a value of negative infinity to illegal actions, reducing the sampled probability of illegal action to zero. This approach ensures that agents refrain from sampling invalid actions, as illustrated in Fig. 4.

在计算每个智能体的Q值时，它为非法动作分配一个负无穷大的值，从而将非法动作的采样概率降低到零。这种方法确保智能体避免采样无效动作，如图4所示。

> However, this method requires the pre-specification of the allowable domain of action space within the state ${s}_{t + 1}$ ,to guarantee a safe transition once ${a}_{t}$ is implemented. For example,DG output,node voltage, and line current constraints are related to the joint actions of all agents, making it difficult to quickly and easily project these constraints to the feasible action space of each agent. Hence, we will classify constraints that significantly impact the safe operation of the system and can be easily verified as hard constraints, while categorizing the remaining constraints as soft constraints, as shown in Table 1. Due to DGs being responsible for maintaining the voltage and frequency balance, it is difficult to pre-determine the output of DG before all agents act. Considering DG significantly affects the safety of the system, only violating this constraint will result in task failure in this paper. In this way, guide agents to prioritize satisfying this soft constraint during the training process.

然而，该方法需要预先指定状态${s}_{t + 1}$内允许的动作空间范围，以确保在实施${a}_{t}$时能够安全过渡。例如，分布式发电（DG）输出、节点电压和线路电流约束与所有智能体的联合动作相关，因此很难快速、轻松地将这些约束映射到每个智能体的可行动作空间。因此，我们将对系统安全运行有显著影响且易于验证的约束归类为硬约束，而将其余约束归类为软约束，如表1所示。由于DG负责维持电压和频率平衡，在所有智能体行动之前很难预先确定DG的输出。考虑到DG对系统安全性有显著影响，本文中仅违反此约束将导致任务失败。通过这种方式，引导智能体在训练过程中优先满足这一软约束。

> Algorithm 1 describes the interaction between agents and the environment. In the simulation environment, it receives actions from agents, modifies power flow calculation files, and conducts power flow calculations. After completing the calculation, analyze results to obtain ${s}_{t + 1},{o}_{t + 1},{a}_{t + 1}^{avl},{r}_{t}$ and ${d}_{t}$ ,then return them to agents. Additionally, the interaction trajectory is saved in the reply buffer. Once the agent reaches the designated number of episodes, samples are extracted from the buffer to calculate the loss function, updating neural network parameters. The process of environmental interaction and training is iterated until the specified number of steps is achieved. The framework is shown in Fig. 5.

算法1描述了智能体与环境的交互过程。在仿真环境中，它接收来自智能体的动作，修改潮流计算文件，并执行潮流计算。计算完成后，分析结果以获取${s}_{t + 1},{o}_{t + 1},{a}_{t + 1}^{avl},{r}_{t}$和${d}_{t}$，然后将它们返回给智能体。此外，交互轨迹被保存在回复缓冲区中。当智能体达到指定的回合数时，从缓冲区中提取样本以计算损失函数，并更新神经网络参数。环境交互和训练过程会迭代进行，直到达到指定的步数。该框架如图5所示。

> Algorithm 1 Multi-agent reinforcement learning for DSR

算法1 用于DSR的多智能体强化学习

> Initialize agent network parameters ${\phi }^{SW},{\phi }^{\text{load }},{\phi }^{PV}$ for switches,

初始化交换机代理网络参数 ${\phi }^{SW},{\phi }^{\text{load }},{\phi }^{PV}$。

***

> ```
> loads,and PVs,and mixing network parameters $ {\phi }^{\text{mix }} $
> ```
>
> ${\text{Initialize socre}}_{\text{best }} = 0$ Initialize replay buffer $D$ Initialize DSR environment for Episode $n = 1$ to $I$ do Reset DSR environment and acquire initial observations ${s}_{0}$ and feasible action set ${\mathbf{a}}_{0}^{\text{avl }}$ for Step $t = 1$ to $T$ do Obtain partial observations $\left\lbrack {{\mathbf{o}}_{1,t},\cdots ,{\mathbf{o}}_{i,t},\cdots ,{\mathbf{o}}_{n,t}}\right\rbrack$ for Agent $i = 1$ to $n$ do Generate pseudo-random numbers $k \sim u\left\lbrack {0,1}\right\rbrack$ if $k \leq \varepsilon$ then Randomly select actions ${a}_{i,t}$ from the set of feasible actions ${\mathbf{a}}_{t}^{avl}$ else ${a}_{i,t} = \arg \mathop{\max }\limits_{{a}_{i}}{Q}_{i}\left( {{\mathbf{o}}_{i,t};{\mathbf{\phi }}_{i}}\right)$ end if end for Obtain joint action ${\mathbf{a}}_{t} = \left\lbrack {{a}_{1,t},\cdots ,{a}_{i,t},\cdots ,{a}_{n,t}}\right\rbrack$ Execute the action ${\mathbf{a}}_{t}$ to the DSR environment and return ${\mathbf{s}}_{t + 1}$ , ${\mathbf{o}}_{t + 1},{\mathbf{a}}_{t + 1}^{avl},{r}_{t}$ and ${d}_{t}$ Save transition $\left( {{\mathbf{s}}_{t},{\mathbf{o}}_{t},{\mathbf{a}}_{t}^{avl},{\mathbf{a}}_{t},{\mathbf{s}}_{t + 1},{\mathbf{o}}_{t + 1},{\mathbf{a}}_{t + 1}^{avl},{r}_{t},{d}_{t}}\right)$ in $\mathcal{D}$ Update states,observations,and feasible actions ${s}_{t} = {s}_{t + 1}$ , ${\mathbf{o}}_{t} = {\mathbf{o}}_{t + 1},{\mathbf{a}}_{t}^{avl} = {\mathbf{a}}_{t + 1}^{avl}$ end for Randomly sample batch-size transitions from $\mathcal{D}$ Calculate the loss function according to (31) and update the behavior network parameters ${\phi }^{SW},{\phi }^{\text{load }},{\phi }^{PV}$ ,and ${\phi }^{\text{mix }}$ Soft update target network parameters ${\widehat{\mathbf{\phi }}}^{SW},{\widehat{\mathbf{\phi }}}^{\text{load }},{\widehat{\mathbf{\phi }}}^{PV}$ ,and ${\widehat{\mathbf{\phi }}}^{\text{mix }}$ according to (32) Update $\varepsilon \leftarrow \varepsilon \cdot {\varepsilon }_{\text{decay-rate }}$ Traversing test scenarios and obtain socre ${}_{\text{test }} = \frac{1}{{N}_{\text{test }}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{\text{test }}}\mathop{\sum }\limits_{{t = 1}}^{N}{r}_{t}$ if ${\text{socre}}_{\text{test }} > {\text{socre}}_{\text{best }}$ then ${\text{ socre }}_{\text{best }} = {\text{ socre }}_{\text{test }}$ Update the best agents model end if end for

负载、光伏和混合网络参数 ${\phi }^{\text{mix }}$ ${\text{Initialize socre}}_{\text{best }} = 0$ 初始化回放缓冲区 $D$ 初始化DSR环境 对于Episode $n = 1$ 到 $I$ 执行 重置DSR环境并获取初始观测值 ${s}_{0}$ 和可行动作集 ${\mathbf{a}}_{0}^{\text{avl }}$ 对于Step $t = 1$ 到 $T$ 执行 获取部分观测值 $\left\lbrack {{\mathbf{o}}_{1,t},\cdots ,{\mathbf{o}}_{i,t},\cdots ,{\mathbf{o}}_{n,t}}\right\rbrack$ 对于Agent $i = 1$ 到 $n$ 执行 生成伪随机数 $k \sim u\left\lbrack {0,1}\right\rbrack$ 如果 $k \leq \varepsilon$ 则 从可行动作集中随机选择动作 ${a}_{i,t}$ 否则 ${a}_{i,t} = \arg \mathop{\max }\limits_{{a}_{i}}{Q}_{i}\left( {{\mathbf{o}}_{i,t};{\mathbf{\phi }}_{i}}\right)$ 结束 结束 获取联合动作 ${\mathbf{a}}_{t} = \left\lbrack {{a}_{1,t},\cdots ,{a}_{i,t},\cdots ,{a}_{n,t}}\right\rbrack$ 执行动作 ${\mathbf{a}}_{t}$ 到DSR环境并返回 ${\mathbf{s}}_{t + 1}$ 、 ${\mathbf{o}}_{t + 1},{\mathbf{a}}_{t + 1}^{avl},{r}_{t}$ 和 ${d}_{t}$ 保存转换 $\left( {{\mathbf{s}}_{t},{\mathbf{o}}_{t},{\mathbf{a}}_{t}^{avl},{\mathbf{a}}_{t},{\mathbf{s}}_{t + 1},{\mathbf{o}}_{t + 1},{\mathbf{a}}_{t + 1}^{avl},{r}_{t},{d}_{t}}\right)$ 在 $\mathcal{D}$ 更新状态、观测值和可行动作 ${s}_{t} = {s}_{t + 1}$ 、 ${\mathbf{o}}_{t} = {\mathbf{o}}_{t + 1},{\mathbf{a}}_{t}^{avl} = {\mathbf{a}}_{t + 1}^{avl}$ 结束 从 $\mathcal{D}$ 中随机采样批量转换 根据(31)计算损失函数并更新行为网络参数 ${\phi }^{SW},{\phi }^{\text{load }},{\phi }^{PV}$ 和 ${\phi }^{\text{mix }}$ 根据(32)软更新目标网络参数 ${\widehat{\mathbf{\phi }}}^{SW},{\widehat{\mathbf{\phi }}}^{\text{load }},{\widehat{\mathbf{\phi }}}^{PV}$ 和 ${\widehat{\mathbf{\phi }}}^{\text{mix }}$ 更新 $\varepsilon \leftarrow \varepsilon \cdot {\varepsilon }_{\text{decay-rate }}$ 遍历测试场景并获取分数 ${}_{\text{test }} = \frac{1}{{N}_{\text{test }}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{\text{test }}}\mathop{\sum }\limits_{{t = 1}}^{N}{r}_{t}$ 如果 ${\text{socre}}_{\text{test }} > {\text{socre}}_{\text{best }}$ 则 ${\text{ socre }}_{\text{best }} = {\text{ socre }}_{\text{test }}$ 更新最佳代理模型 结束 结束

***

> ## 4. Case study

## 4. 案例研究

> In order to verify the effectiveness of the proposed method, we developed a Python-based DSR simulation environment that can support various DRL algorithm training. A case study was conducted using the modified balanced IEEE 123 node test system. The power flow calculation of the distribution network is completed based on Pandapower. The main environment configurations are Python 3.8, Pandapower 2.4.0, and PyTorch 1.9.0.

为了验证所提出方法的有效性，我们开发了一个基于Python的DSR仿真环境，该环境能够支持各种DRL算法的训练。使用修改后的平衡IEEE 123节点测试系统进行了案例研究。配电网的潮流计算基于Pandapower完成。主要环境配置为Python 3.8、Pandapower 2.4.0和PyTorch 1.9.0。

> ### 4.1. Setup

### 4.1. 设置

> The topology of the modified IEEE 123-node test system is shown in Fig. 9(a), which is a radial distribution network with a rated voltage of ${4.16}\mathrm{{kV}}$ . The system comprises 85 loads,each assumed to have a power factor of 0.9 . We divide the priority of loads into three levels, with corresponding weight factors of 1,2 , and 3 . The number and power ratio of loads at each level are shown in Table 2. In addition, the system includes 20 switches, 7 DGs, and 9 distributed PVs. The detailed parameters of generators are shown in Table 3. The state dimension of the system is 347 , with 114 decision variables per step.

修改后的IEEE 123节点测试系统的拓扑结构如图9(a)所示，这是一个额定电压为${4.16}\mathrm{{kV}}$的辐射状配电网络。该系统包含85个负荷，每个负荷的功率因数假设为0.9。我们将负荷的优先级分为三个等级，对应的权重因子分别为1、2和3。每个等级的负荷数量和功率比见表2。此外，系统还包括20个开关、7个分布式发电机（DG）和9个分布式光伏（PV）。发电机的详细参数见表3。系统的状态维度为347，每步有114个决策变量。

![\<img src="attachments/9LZ3YL2G.jpg" alt="" data-attachment-key="9LZ3YL2G" width="1196.461824953445" height="500" ztype="zimage"> | 1196.461824953445](attachments/9LZ3YL2G.jpg)

> Fig. 5. The framework of agent training and application.

图5. 代理训练与应用的框架。

> Table 2

表2

> The information of each level load.

每个层级负载的信息。

| <!-- --> | <!-- --> | <!-- --> |
| -------------- | -------- | --------------- |
|                | Quantity | Power ratio (%) |
| 1st-level load | 12       | 14.90           |
| 2nd-level load | 12       | 13.32           |
| 3rd-level load | 61       | 71.78           |


> Table 3

表3

> The information of power sources.

电源信息。

| <!-- --> | <!-- --> | <!-- --> |
| -- | ----------- | -------------------------- |
|    | Rating (kW) | Connected node             |
| DG | 150         | 9,28,38,67,90,99,108       |
| PV | 150         | 3,23,32,50,60,68,83,87,117 |


> The time series data used in the case is obtained from the public dataset of the Australian distribution network. This dataset includes information on electricity consumption and rooftop PV power generation of residential users \[42]. The characteristics of the scenario are considered from two perspectives. The first is temporality, the PV and load power characteristics vary across different time periods. We extract data every 5 days from the original dataset, covering both weekdays and weekends. The extraction time points are $0 : {00},4 : {00}$ , $8 : {00},{12} : {00},{16} : {00}$ ,and 20:00,including the noon PV power generation peak and the evening load electricity consumption peak. The second is fluctuation, which refers to the changes in PV and load power within an episode. The load demand power typically exhibits relative stability, thus it is assumed to remain constant within the second-level time granularity. The fluctuations in PV generation are assumed to follow a normal distribution $\mu = 0,\sigma = {0.025}$ . This process results in a total of 876 time series scenarios. Besides that, randomly set 3 or 4 lines in fault and create a total of 1000 fault scenarios.

案例中使用的时间序列数据来自澳大利亚配电网络的公开数据集。该数据集包括居民用户的电力消耗和屋顶光伏发电信息\[42]。场景特征从两个角度考虑。第一个是时间性，光伏和负荷功率特性在不同时间段内变化。我们从原始数据集中每5天提取一次数据，涵盖工作日和周末。提取时间点为$0 : {00},4 : {00}$、$8 : {00},{12} : {00},{16} : {00}$和20:00，包括中午光伏发电高峰和晚间负荷用电高峰。第二个是波动性，指的是光伏和负荷功率在一个时段内的变化。负荷需求功率通常表现出相对稳定性，因此假设其在秒级时间粒度内保持不变。光伏发电的波动假设服从正态分布$\mu = 0,\sigma = {0.025}$。此过程共生成876个时间序列场景。除此之外，随机设置3或4条线路故障，共生成1000个故障场景。

> In the DAN, the environmental information encoder comprises two fully connected layers, each with 128 neurons. The interactive information encoder consists of one layer with 16 neurons. The information aggregation layer contains 128 neurons, utilizing ReLu as the activation function. The mixing network layer has 256 neurons and employs ELU as the activation function. The DRL decay rate $\gamma$ is set to 0.99,and the agent chooses random actions for exploration with a decay probability $\varepsilon \left( {{\varepsilon }_{\infty } = {0.05}}\right)$ . The soft update coefficient $\xi$ is 0.001 . Regarding neural network training parameters, the batch size is 32 , the learning rate is 0.0001 , and the RMSprop is employed for the stochastic gradient descent algorithm. About the parameters of the reward function, ${\alpha }_{r} =$ ${20},{\alpha }_{v} = 1,{\alpha }_{l} = 1,{r}^{\text{done }} = - 5$ . The maximum number of steps per episode is set to 15 . When the output of the DG exceeds the limit, the episode terminates, indicating that the DSR task has failed, and agents receive a negative reward.

在DAN中，环境信息编码器由两个全连接层组成，每层有128个神经元。交互信息编码器由一层16个神经元组成。信息聚合层包含128个神经元，使用ReLu作为激活函数。混合网络层有256个神经元，并采用ELU作为激活函数。DRL衰减率$\gamma$设置为0.99，代理以衰减概率$\varepsilon \left( {{\varepsilon }_{\infty } = {0.05}}\right)$选择随机动作进行探索。软更新系数$\xi$为0.001。关于神经网络训练参数，批量大小为32，学习率为0.0001，并使用RMSprop进行随机梯度下降算法。关于奖励函数的参数，${\alpha }_{r} =$ ${20},{\alpha }_{v} = 1,{\alpha }_{l} = 1,{r}^{\text{done }} = - 5$。每回合的最大步数设置为15。当DG的输出超过限制时，回合终止，表示DSR任务失败，代理将收到负奖励。

> ### 4.2. Training capacity of the proposed method

### 4.2. 所提出方法的训练能力

> In the experiment, we considered three metrics to evaluate the learning performance.

在实验中，我们考虑了三个指标来评估学习表现。

> (1) Cumulative Reward (CR): Cumulative rewards achieved by agents in an episode, $\mathop{\sum }\limits_{{t = 1}}^{N}{r}_{t}$ .

(1) 累积奖励 (CR)：智能体在一轮中获得的累积奖励，$\mathop{\sum }\limits_{{t = 1}}^{N}{r}_{t}$。

> (2) Alive Steps (AS): Because of illegal actions executed by agents leading to the termination of the episode, alive steps are employed to indicate the number of successful interaction steps taken by agents in an episode.

(2) 存活步数 (AS)：由于代理执行的非法行为导致情节终止，存活步数用于表示代理在一个情节中成功进行的交互步骤数量。

> (3) Restoring Rate (RR): The proportion of the load that agents successfully restored in an episode. If the episode is terminated prematurely, the restoration rate is equal to 0 . Otherwise, it is equal to $\mathop{\sum }\limits_{{i \in \mathcal{N}}}{c}_{i} \cdot {x}_{i,t}^{\text{load }} \cdot {p}_{i,t}^{\text{load }}/{c}_{i} \cdot {p}_{i,t}^{\text{load }}\left( {t = {15}}\right)$ .

(3) 恢复率 (RR)：代理在某一回合中成功恢复的负载比例。如果回合提前终止，恢复率等于0。否则，它等于$\mathop{\sum }\limits_{{i \in \mathcal{N}}}{c}_{i} \cdot {x}_{i,t}^{\text{load }} \cdot {p}_{i,t}^{\text{load }}/{c}_{i} \cdot {p}_{i,t}^{\text{load }}\left( {t = {15}}\right)$。

> To demonstrate the effectiveness of the proposed algorithm, the attention-based aggregation method and average-based aggregation method in the DAN interactive information encoder are compared.Also, the VDN algorithm \[43] for MADRL is compared. A total of 5 experiments are conducted, including DAN + Attention + QMIX, DAN + Mean + QMIX, QMIX, DAN + Attention + VDN, and DAN + Mean + VDN.

为了验证所提出算法的有效性，对DAN交互信息编码器中的基于注意力的聚合方法和基于平均的聚合方法进行了比较。此外，还比较了用于MADRL的VDN算法\[43]。共进行了5组实验，包括DAN + Attention + QMIX、DAN + Mean + QMIX、QMIX、DAN + Attention + VDN和DAN + Mean + VDN。

> Compared attention-based aggregation method in formula (25), average-based aggregation represents mean embedding, which average aggregate observation information of other agents in the same MG. The formula is

与公式（25）中基于注意力的聚合方法相比，基于平均的聚合表示均值嵌入，该方法对同一MG中其他智能体的观测信息进行平均聚合。公式为

$$
{\mathbf{v}}_{i,t} = \frac{1}{{N}_{i} - 1}\mathop{\sum }\limits_{{\forall j \in  {\mathcal{N}}_{i},j \neq  i}}\left( {o}_{i,t}^{j}\right)
$$

> ${N}_{i}$ is the number of agents in the MG where agent $i$ is located.

${N}_{i}$ 是位于多智能体系统（MG）中智能体 $i$ 所在位置的智能体数量。

> Fig. 6 displays the moving average curves of CR and AS for various algorithms throughout the training process. Because of using greedy strategy, agents choose random actions with a high probability to explore the environment in the early stages. It may lead to the generation of illegal actions, subsequently reducing AS and receiving negative rewards. As training progresses, agents progressively optimize policy and achieve a better balance between exploration and exploitation. After agents complete 25,000 episodes of interaction, all algorithms can converge. However, due to the differences in scenarios and the probability of random actions with ${\varepsilon }_{\infty } = {0.05}$ ,there are still fluctuations in the final training curve, which cannot fully converge to smoothness. Simultaneously, the training curve also serves as validation for the effectiveness of the proposed method.

图6展示了各种算法在整个训练过程中CR和AS的移动平均曲线。由于使用了贪心策略，智能体在早期阶段以高概率选择随机动作来探索环境。这可能导致生成非法动作，从而降低AS并收到负奖励。随着训练的进行，智能体逐渐优化策略，并在探索与利用之间实现更好的平衡。在智能体完成25,000次交互后，所有算法都能收敛。然而，由于场景的差异和随机动作概率为${\varepsilon }_{\infty } = {0.05}$，最终训练曲线仍存在波动，无法完全收敛至平滑状态。同时，训练曲线也验证了所提出方法的有效性。

![\<img src="attachments/Z4UN34I5.jpg" alt="" data-attachment-key="Z4UN34I5" width="368.8029020556227" height="500" ztype="zimage"> | 368.8029020556227](attachments/Z4UN34I5.jpg)

> Fig. 6. Comparison of Training performance.

图6. 训练性能比较。

> i. While the AS of the VDN algorithm may rapidly converge to a level close to the maximum step, the agent struggles to acquire an effective control policy. The CR curve converges to an exceptionally low value.

i. 尽管VDN算法的AS可能迅速收敛到接近最大步长的水平，但智能体难以获得有效的控制策略。CR曲线收敛到一个极低的值。

> ii. The basic QMIX algorithm only takes environment observation as input. It has difficulty in modeling the behavior of neighboring agents, hindering effective collaboration among agents. Consequently, both CR and AS curves tend to converge to lower values.

ii. 基础的QMIX算法仅以环境观测作为输入。它在建模邻近智能体行为方面存在困难，阻碍了智能体之间的有效协作。因此，CR和AS曲线往往趋向于收敛到较低的值。

> iii. For the DAN with attention mechanism, it can dynamically allocate weights to various neighboring agent observations. The attention weight can be interpreted as the relative importance between two agents. Thus, it assists agents in automatically recognizing essential information. Consequently, it can achieve higher convergence levels on both CR and AS, enabling more effective completion of collaborative control tasks in the complex scene.

iii. 对于带有注意力机制的DAN，它可以动态地为各种邻近智能体的观测分配权重。注意力权重可以解释为两个智能体之间的相对重要性。因此，它帮助智能体自动识别关键信息。最终，它可以在CR和AS上实现更高的收敛水平，从而在复杂场景中更有效地完成协作控制任务。

> iv. Based on the analysis of actions, VDN based algorithms converge to a policy that not execute load recovery operations. Consequently, the CR, closely associated with the restoration rate, remains consistently low. The absence of load recovery actions ensures that DG does not surpass its limit, resulting in a relatively large AS. Whether during PV generation peaks or load consumption peaks, VDN based algorithms demonstrate a relatively stable CR across various scenarios, exhibiting smaller fluctuations compared to QMIX-based algorithms.

iv. 根据对行动的分析，基于VDN的算法收敛到一种不执行负载恢复操作的策略。因此，与恢复率密切相关的CR（恢复率）始终保持在较低水平。负载恢复操作的缺失确保了DG（分布式发电）不会超过其限制，从而导致相对较大的AS（可用性）。无论是在光伏发电峰值还是负载消耗峰值期间，基于VDN的算法在各种场景下都表现出相对稳定的CR，与基于QMIX的算法相比，波动较小。

> To illustrate the effectiveness of the action mask mechanism, Fig. 7 presents a comparative analysis between experiments with and without action mask while keeping the same training hyperparameters. The experimental results indicate that the action mask mechanism can dynamically shrink the action space of agents. Due to diminishing inefficient exploration, it assists agents in efficiently converging to a feasible strategy. Faced with the problem characterized by combinatorial explosion, algorithms without action mask make it difficult for agents to find the optimal point. The experimental results show that although agents can converge to a high AS, the low CR indicates that agents choose not to restore loads, to ensure the satisfaction of DG output constraints. The pure reward-guidance method cannot find a truly effective strategy, rendering agents incapable of completing DSR tasks. Thus, the action mask mechanism has unique effectiveness in CR and learning performance.

为了说明动作掩码机制的有效性，图7展示了在保持相同训练超参数的情况下，使用和不使用动作掩码的实验对比分析。实验结果表明，动作掩码机制能够动态缩小智能体的动作空间。由于减少了低效探索，它有助于智能体高效收敛到可行策略。面对组合爆炸问题，没有动作掩码的算法使得智能体难以找到最优解。实验结果显示，尽管智能体能够收敛到较高的AS，但较低的CR表明智能体选择不恢复负载，以确保满足DG输出约束。纯奖励引导方法无法找到真正有效的策略，导致智能体无法完成DSR任务。因此，动作掩码机制在CR和学习性能方面具有独特的效果。

![\<img src="attachments/TS8NR7SU.jpg" alt="" data-attachment-key="TS8NR7SU" width="900.2659574468086" height="500" ztype="zimage"> | 900.2659574468086](attachments/TS8NR7SU.jpg)

> Fig. 7. Results of ablation experiment on action mask.

图7. 动作掩码消融实验结果。

![\<img src="attachments/VP9UXQ9Q.jpg" alt="" data-attachment-key="VP9UXQ9Q" width="947.0588235294117" height="500" ztype="zimage"> | 947.0588235294117](attachments/VP9UXQ9Q.jpg)

> Fig. 8. Test performance of RR.

图8. RR的测试性能。

> ### 4.3. Test performance in DSR problem

### 4.3. DSR问题中的测试性能

> Fig. 8 displays the distribution of RR for three QMIX-based algorithms under 500 test scenarios. This paper focuses on the issue of losing the transmission network power and relying on local resources for load recovery. In various scenarios, the available total local resource capacity varies. Therefore, it may not be feasible to fully restore all loads in every scenario. The experimental results demonstrate that the proposed method achieves a higher load recovery rate, irrespective of whether power resources are abundant or limited. The strategies learned by the other two algorithms are relatively conservative. In scenarios with limited power resources, agents choose to restore only a minimal load to ensure the output of DGs remains below the limit. The conservative strategy results in RR values of approximately 80% and 70% in the situation with abundant power resources, which fail to recover nearly all loads and are significantly lower than the proposed method.

图8展示了三种基于QMIX的算法在500个测试场景下的RR分布。本文重点研究了在失去输电网络电力的情况下，依赖本地资源进行负荷恢复的问题。在不同的场景中，可用的本地资源总容量各不相同。因此，在每个场景中完全恢复所有负荷可能并不可行。实验结果表明，无论电力资源是否充足，所提出的方法都能实现更高的负荷恢复率。其他两种算法学习到的策略相对保守。在电力资源有限的情况下，代理选择仅恢复最小负荷，以确保分布式发电机的输出保持在限制范围内。这种保守策略在电力资源充足的情况下，RR值约为80%和70%，未能恢复几乎所有负荷，且显著低于所提出的方法。

> Scenario I: Occurring at 12:00, when PV power generation reaches its peak, the power supply of the system is adequate to recover nearly all loads. The primary objective of agents is to quickly and efficiently restore the load. When confronted with the loss of the transmission network power supply and accounting for random faults occurring on lines 59-62, 68-69, and 73-74, the reconfigured topology is depicted in Fig. 9(a). In this configuration, the system is divided into three isolated MGs, while retaining a radial topology. Each is equipped with at least one DG serving as a black-start source. For the affected loads, the distribution network is reconfiguration by controlling switches to transfer these loads to other power supply paths. Due to nodes 74, 75, and 76 having only one power supply path, a fault occurring in that particular line will render it incapable of restoring the connected load. Fig. 10(b) illustrates the load recovery ratio for each step, where higher-priority loads are prioritized for restoration, resulting in a recovery rate of 95.87%. In scenarios with abundant power resources, it becomes feasible to restore all loads having transferable power supply paths.

场景一：发生在12:00，当光伏发电达到峰值时，系统的电力供应足以恢复几乎所有负荷。代理的主要目标是快速高效地恢复负荷。当面临输电网络电力供应中断并考虑线路59-62、68-69和73-74上发生的随机故障时，重新配置的拓扑结构如图9(a)所示。在此配置中，系统被划分为三个孤立的微电网，同时保留了辐射状拓扑结构。每个微电网至少配备一个分布式发电装置作为黑启动电源。对于受影响的负荷，通过控制开关重新配置配电网，将这些负荷转移到其他供电路径。由于节点74、75和76仅有一条供电路径，该线路发生故障将导致无法恢复连接的负荷。图10(b)展示了每个步骤的负荷恢复率，其中优先级较高的负荷优先恢复，恢复率达到95.87%。在电力资源充足的情况下，恢复所有具有可转移供电路径的负荷是可行的。

![\<img src="attachments/V2HWNMA9.jpg" alt="" data-attachment-key="V2HWNMA9" width="307.6923076923077" height="500" ztype="zimage"> | 307.6923076923077](attachments/V2HWNMA9.jpg)

> Fig. 9. Results of network formation.

图9. 网络形成的结果。

![\<img src="attachments/BZSR7HRW.jpg" alt="" data-attachment-key="BZSR7HRW" width="500" height="500" ztype="zimage"> | 500](attachments/BZSR7HRW.jpg)

> Fig. 10. Sequential load recovery ratio.

图10. 顺序负荷恢复率。

> Scenario II: Occurring at 20:00, the PV is not generating power, but the load is at its consumption peak. The system relies on DGs as emergency power sources. A limited power source can only restore a part of loads. The primary objective of agents is to prioritize the restoration of critical loads. The reconfigured topology is displayed in Fig. 9(b), following the failure of lines 50-51, 58-59, and 100- 101. To restore load as much as possible, it is necessary to form a large MG and achieve multi-source coordination. However, this could cause the power supply path to be long, resulting in excessive network loss and violation of voltage and current constraints. Therefore, in this scenario, the distribution network is divided into three isolated MGs. Fig. 10(b) displays the load recovery ratio for each step, with the last load recovery rate of 59.22%.

场景二：发生在20:00，光伏系统未发电，但负载处于用电高峰。系统依赖分布式发电（DGs）作为应急电源。有限的电源只能恢复部分负载。代理的主要目标是优先恢复关键负载。线路50-51、58-59和100-101故障后，重新配置的拓扑结构如图9(b)所示。为了尽可能多地恢复负载，需要形成一个大型微电网（MG）并实现多源协调。然而，这可能导致供电路径过长，造成过大的网络损耗并违反电压和电流约束。因此，在此场景中，配电网被划分为三个孤立的微电网。图10(b)显示了每一步的负载恢复比例，最终负载恢复率为59.22%。

> ### 4.4. Verification of algorithm scalability

### 4.4. 算法可扩展性验证

> To test the device quantity scalability of the proposed algorithm, a new 2nd-level load is added at node 54, and a PV is added at node 98 . The increase in the number of control objects leads to changes in the dimensions of the observation space and action space. For the variation of observation space, we use the DAN architecture to adapt the changes in the input dimension. For the variation of action space, the newly added agent reuses the trained strategy by the neural network parameter-sharing method. Fig. 11 illustrates the load recovery progress at each time step. In the initial stage, priority is given to restoring 1st-level loads, and the newly added 2nd-level load is restored at step 8 . Since all recovery operations were completed in 10 steps, the figure displays the first 10 steps. Ultimately, all 1st and 2nd-level loads were successfully restored, resulting in a recovery ratio of 95.08%.

为了测试所提出算法的设备数量可扩展性，在节点54处添加了一个新的二级负载，并在节点98处添加了一个光伏电源。控制对象数量的增加导致观测空间和动作空间的维度发生变化。对于观测空间的变化，我们使用DAN架构来适应输入维度的变化。对于动作空间的变化，新添加的代理通过神经网络参数共享方法重用已训练的策略。图11展示了每个时间步长的负载恢复进度。在初始阶段，优先恢复一级负载，新添加的二级负载在第8步恢复。由于所有恢复操作在10步内完成，图中显示了前10步。最终，所有一级和二级负载均成功恢复，恢复率达到95.08%。

> Simultaneously, we simulate a case where only a PV is added in a situation of power resource shortage, with the new PV positioned at node 98 . Due to the relatively low PV power generation capacity at 16:00, agents are unable to recover all loads. The newly added PV can supply more loads, which increases the power capacity of the system. Fig. 12(a) presents the simulation results. The connection of the new PV led to an increase in the load recovery rate from 79.5% to 94.9%, representing a significant improvement of 15.4%.

同时，我们模拟了一种在电力资源短缺的情况下仅增加光伏发电的情景，新增的光伏发电位于节点98。由于16:00时光伏发电能力相对较低，代理无法恢复所有负荷。新增的光伏发电可以供应更多负荷，从而提高了系统的电力容量。图12(a)展示了模拟结果。新增光伏的接入使负荷恢复率从79.5%提高到94.9%，显著提升了15.4%。

> We also simulate a scenario of adding a load under limited power resources, with the newly added load positioned at node 54. Before adding new loads, the power resources are sufficient, and agents can restore almost all loads. However, the newly added and high-consumption load results in a rise in the total load power, surpassing the allowable generation capacity of the system. Consequently, some loads cannot be fully restored. Fig. 12(b) illustrates the simulation results. The increased number of loads led to a reduction in the load recovery rate, decreasing from 93.0% to 88.8%.

我们还模拟了在有限电力资源下增加负载的场景，新增负载位于节点54。在增加新负载之前，电力资源充足，代理可以恢复几乎所有负载。然而，新增的高耗电负载导致总负载功率上升，超过了系统的允许发电容量。因此，部分负载无法完全恢复。图12(b)展示了模拟结果。负载数量的增加导致负载恢复率下降，从93.0%降至88.8%。

> ### 4.5. Comparison with model-driven method

### 4.5. 与模型驱动方法的比较

> We conduct a comparative analysis of the proposed method against the model-driven mathematical optimization method. Their performance is analyzed in terms of decision optimality and computational efficiency. Considering the difficulty in modeling multi-source coordination sequential DSR optimal model, we transform the sequential DSR problem into a set of one-step DSR problems. For the non-linear DG output constraint, we replace it with maximum active and reactive output constraints to realize linearization,that is ${p}_{i,t}^{G} \leq {0.8} \cdot {S}_{i}^{G}$ and ${q}_{i,t}^{G} \leq {0.6} \cdot {S}_{i}^{G}$ . In addition,we use linear DistFlow constraints to model the DSR as a MILP problem. Meanwhile, the pre-determined set of invalid actions will be transformed into constraints, serving as input for the optimization algorithm. The invalid action set is the action that violates the hard constraints in Table 1. During each optimization step, the DSR optimization model is modified according to the present state of the power system, and an optimization algorithm is invoked for solving.

我们对所提出的方法与模型驱动的数学优化方法进行了比较分析。从决策最优性和计算效率的角度分析了它们的性能。考虑到多源协调顺序DSR优化模型建模的困难，我们将顺序DSR问题转化为一系列单步DSR问题。对于非线性DG输出约束，我们将其替换为最大有功和无功输出约束以实现线性化，即${p}_{i,t}^{G} \leq {0.8} \cdot {S}_{i}^{G}$和${q}_{i,t}^{G} \leq {0.6} \cdot {S}_{i}^{G}$。此外，我们使用线性DistFlow约束将DSR建模为MILP问题。同时，预定义的无效动作集将被转化为约束，作为优化算法的输入。无效动作集是违反表1中硬约束的动作。在每个优化步骤中，根据电力系统的当前状态修改DSR优化模型，并调用优化算法进行求解。

![\<img src="attachments/DY5JPAKM.jpg" alt="" data-attachment-key="DY5JPAKM" width="2637.809187279152" height="500" ztype="zimage"> | 2637.809187279152](attachments/DY5JPAKM.jpg)

> Fig. 11. System load energization status.

图11. 系统负载通电状态。

![\<img src="attachments/IU4GS339.jpg" alt="" data-attachment-key="IU4GS339" width="498.52941176470586" height="500" ztype="zimage"> | 498.52941176470586](attachments/IU4GS339.jpg)

> Fig. 12. Comparison of sequential load recovery ratio for the newly added device.

图12. 新添加设备的顺序负载恢复率比较。

> Regarding the RR, the comparison results are depicted in Fig. 13. The experimental results demonstrate that data-driven methods outperform the comparative method in most scenarios. This is because the DRL method strikes a balance between immediate and future rewards through discount factors $\gamma$ . This equilibrium enables agents to make decisions with foresight, thereby enhancing the coordination between switch and load control in temporal decision-making. It ultimately results in higher load recovery rates. The comparative algorithm decouples the sequential DSR problem. It does not consider the interdependence of temporal decisions, the highest recovery rate is taken as the optimization goal in each step. This approach cannot achieve look-ahead management of resources and can only obtain suboptimal control strategies.

关于RR，比较结果如图13所示。实验结果表明，在大多数情况下，数据驱动方法优于对比方法。这是因为DRL方法通过折扣因子$\gamma$在即时奖励和未来奖励之间取得了平衡。这种平衡使得代理能够以预见性做出决策，从而增强了开关和负载控制在时间决策中的协调性，最终实现了更高的负载恢复率。对比算法将顺序DSR问题解耦，未考虑时间决策的相互依赖性，每一步都以最高恢复率为优化目标。这种方法无法实现资源的超前管理，只能获得次优的控制策略。

> In terms of computational efficiency, the average time required for a complete DSR task is taken as an indicator. The time spent in mathematical optimization methods is ${7.425}\mathrm{\;s}$ . The time-decoupling method simplified the complexity of the dynamic programming problem ${n}^{T}$ to $n \cdot T$ . Hard constraints are taken into account during the calculation of the invalid action set. In the optimization process, these constraints are not considered since the invalid action set has already been incorporated into the optimization model. The time spent in DRL is ${0.39}\mathrm{\;s}$ ,which is only about one-twentieth of the time spent by the model-driven approach. DRL can quickly map system states to control decisions through an end-to-end model, without the need for iterative solving. In addition, in the face of different problem scales, the solving speed of DRL is only related to the complexity of the neural network. Lightweight neural networks can be constructed to meet the time requirements of different power system control and optimization tasks.

在计算效率方面，以完成DSR任务所需的平均时间作为指标。数学优化方法所花费的时间为${7.425}\mathrm{\;s}$。时间解耦方法将动态规划问题的复杂度从${n}^{T}$简化到$n \cdot T$。在计算无效动作集时考虑了硬约束。在优化过程中，由于无效动作集已经纳入优化模型，因此不再考虑这些约束。DRL所花费的时间为${0.39}\mathrm{\;s}$，仅为模型驱动方法所花费时间的二十分之一。DRL可以通过端到端模型快速将系统状态映射到控制决策，而无需迭代求解。此外，面对不同的问题规模，DRL的求解速度仅与神经网络的复杂度相关。可以构建轻量级神经网络，以满足不同电力系统控制和优化任务的时间要求。

![\<img src="attachments/5SAVC54T.jpg" alt="" data-attachment-key="5SAVC54T" width="939.7260273972603" height="500" ztype="zimage"> | 939.7260273972603](attachments/5SAVC54T.jpg)

> Fig. 13. Comparison of Data-driven solutions and Model-driven solutions.

图13. 数据驱动解决方案与模型驱动解决方案的比较。

> ## 5. Conclusion

## 5. 结论

> A distribution system restoration algorithm based on multi-agent reinforcement learning is proposed in this paper, which can restore distribution services that have lost transmission network power supply. We formulate a feasible operational sequence, suitable for diverse initial scenarios, by the collaborative control of switches, distributed power sources, and load pickup. The experimental results illustrate the proposed algorithm can learn efficient load recovery strategies, and it has higher computational efficiency based on the end-to-end approach of neural networks. And mask technique is used to mitigate illegal actions, which ensures the security of the learned policy. Moreover, through the implementation of a dynamic agent network, it not only processes observational information from other agents to achieve effective collaboration among various controllable resources but also adapts to dynamic changes in the number of devices within the power system.

本文提出了一种基于多智能体强化学习的配电系统恢复算法，能够恢复因输电网络供电中断而失去的配电服务。通过开关、分布式电源和负荷接入的协同控制，我们制定了一种适用于多种初始场景的可行操作序列。实验结果表明，所提出的算法能够学习高效的负荷恢复策略，并且基于神经网络的端到端方法具有更高的计算效率。此外，通过使用掩码技术来缓解非法操作，确保了学习策略的安全性。此外，通过动态智能体网络的实现，不仅能够处理来自其他智能体的观测信息以实现各种可控资源的有效协作，还能适应电力系统中设备数量的动态变化。

> Furthermore, the work comes with some limitations. This paper primarily focuses on the scalability of multi-agent strategies. Regarding the transferability of multi-agent policies, fine-tune based on a small amount of training data and the current strategy can be explored. Accelerating learning efficiency and enhancing model performance through transfer learning emerges as a noteworthy issue for future investigation.

此外，这项工作存在一些局限性。本文主要关注多智能体策略的可扩展性。关于多智能体策略的可迁移性，可以探索基于少量训练数据和当前策略的微调。通过迁移学习加速学习效率并提升模型性能，是未来研究中值得关注的问题。

> ## CRediT authorship contribution statement

## CRediT作者贡献声明

> Ruiqi Si: Writing - original draft, Visualization, Software, Methodology, Conceptualization. Siyuan Chen: Methodology, Conceptualization, Writing - review & editing. Jun Zhang: Writing - review & editing, Methodology, Conceptualization. Jian Xu: Writing - review & editing, Methodology, Conceptualization. Luxi Zhang: Writing - review & editing, Methodology, Conceptualization.

司瑞琪：撰写 - 初稿，可视化，软件，方法论，概念化。陈思远：方法论，概念化，撰写 - 审阅与编辑。张军：撰写 - 审阅与编辑，方法论，概念化。徐健：撰写 - 审阅与编辑，方法论，概念化。张璐熙：撰写 - 审阅与编辑，方法论，概念化。

> ## Declaration of competing interest

## 利益冲突声明

> The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

作者声明，他们没有已知的竞争性财务利益或个人关系可能影响本论文所报告的研究工作。

> ## Data availability

## 数据可用性

> No data was used for the research described in the article.

文章中描述的研究未使用任何数据。

> ## References

## 参考文献

> \[1] Chen C, Wang J, Ton D. Modernizing distribution system restoration to achieve grid resiliency against extreme weather events: An integrated solution. Proc IEEE 2017;105(7):1267-88.

\[1] 陈C，王J，Ton D。现代化配电系统恢复以实现电网在极端天气事件中的韧性：一种综合解决方案。《IEEE学报》2017;105(7):1267-88。

> \[2] Küfeoğlu S, Prittinen S, Lehtonen M. A summary of the recent extreme weather events and their impacts on electricity. Int Rev Electr Eng 2014;9(4):821-8.

\[2] Küfeoğlu S, Prittinen S, Lehtonen M. 近期极端天气事件及其对电力影响的综述。国际电气工程评论 2014;9(4):821-8.

> \[3] Bialek J. What does the gb power outage on 9 august 2019 tell us about the current state of decarbonised power systems? Energy Policy 2020;146:111821.

\[3] Bialek J. 2019年8月9日英国停电事件对当前脱碳电力系统的启示是什么？《能源政策》2020;146:111821。

> \[4] Sun H, Xu T, Guo Q, Li Y, Lin W, Yi J, et al. Analysis on blackout in great britain power grid on august 9th, 2019 and its enlightenment to power grid in China. In: Proceedings of the CSEE, vol. 39, 2019, p. 6183-91.

\[4] 孙浩, 徐涛, 郭强, 李勇, 林伟, 易俊, 等. 2019年8月9日英国电网大停电分析及其对中国电网的启示. 中国电机工程学报, 2019, 39: 6183-91.

> \[5] Operator AEM. Preliminary report: black system event in south Australia on 28 September 2016. 2016.

\[5] 运营商AEM。初步报告：2016年9月28日南澳大利亚州黑系统事件。2016年。

> \[6] Zeng H, Sun F, Li T, Zhang Q, Tang J, Zhang T. Analysis of 9. 28 blackout in south Australia and its enlightenment to China. Autom Electr Power Syst 2017;41(13):1-6.

\[6] 曾辉, 孙峰, 李涛, 张强, 唐杰, 张涛. 南澳大利亚9.28大停电事故分析及对中国的启示. 电力系统自动化 2017;41(13):1-6.

> \[7] Liu W, Ding F. Hierarchical distribution system adaptive restoration with diverse distributed energy resources. IEEE Trans Sustain Energy 2020;12(2):1347-59.

\[7] 刘伟，丁峰. 基于多种分布式能源的分层配电系统自适应恢复. IEEE可持续能源汇刊 2020;12(2):1347-59.

> \[8] Chen B, Chen C, Wang J, Butler-Purry KL. Multi-time step service restoration for advanced distribution systems and microgrids. IEEE Trans Smart Grid 2017;9(6):6793-805.

\[8] 陈斌, 陈超, 王杰, Butler-Purry KL. 高级配电系统和微电网的多时间步服务恢复. IEEE 智能电网汇刊 2017;9(6):6793-805.

> \[9] Gazijahani FS, Salehi J, Shafie-Khah M, Catalão JP. Spatiotemporal splitting of distribution networks into self-healing resilient microgrids using an adjustable interval optimization. IEEE Trans Ind Inf 2020;17(8):5218-29.

\[9] Gazijahani FS, Salehi J, Shafie-Khah M, Catalão JP. 基于可调区间优化的配电网时空分割为自愈弹性微电网。《IEEE工业信息学报》2020;17(8):5218-29.

> \[10] Chen B, Chen C, Wang J, Butler-Purry KL. Sequential service restoration for unbalanced distribution systems and microgrids. IEEE Trans Power Syst 2017;33(2):1507-20.

\[10] 陈斌, 陈超, 王杰, Butler-Purry KL. 不平衡配电系统和微电网的顺序服务恢复. IEEE电力系统汇刊 2017;33(2):1507-20.

> \[11] Lei S, Chen C, Li Y, Hou Y. Resilient disaster recovery logistics of distribution systems: Co-optimize service restoration with repair crew and mobile power source dispatch. IEEE Trans Smart Grid 2019;10(6):6187-202.

\[11] 雷松, 陈超, 李勇, 侯宇. 配电系统弹性灾害恢复物流：服务恢复与维修队伍及移动电源调度的协同优化. IEEE 智能电网汇刊 2019;10(6):6187-202.

> \[12] Arif A, Ma S, Wang Z, Wang J, Ryan SM, Chen C. Optimizing service restoration in distribution systems with uncertain repair time and demand. IEEE Trans Power Syst 2018;33(6):6828-38.

\[12] Arif A, Ma S, Wang Z, Wang J, Ryan SM, Chen C. 优化具有不确定修复时间和需求的配电系统中的服务恢复。IEEE 电力系统汇刊 2018;33(6):6828-38.

> \[13] Chen B, Ye Z, Chen C, Wang J. Toward a milp modeling framework for distribution system restoration. IEEE Trans Power Syst 2018;34(3):1749-60.

\[13] 陈斌, 叶志, 陈超, 王杰. 面向配电网恢复的MILP建模框架研究. 电力系统学报 2018;34(3):1749-60.

> \[14] Ye Z, Chen C, Chen B, Wu K. Resilient service restoration for unbalanced distribution systems with distributed energy resources by leveraging mobile generators. IEEE Trans Ind Inf 2020;17(2):1386-96.

\[14] 叶志, 陈晨, 陈波, 吴凯. 利用移动发电机实现含分布式能源的不平衡配电系统弹性服务恢复. IEEE工业信息学报 2020;17(2):1386-96.

> \[15] Ding T, Lin Y, Bie Z, Chen C. A resilient microgrid formation strategy for load restoration considering master-slave distributed generators and topology reconfiguration. Appl Energy 2017;199:205-16.

\[15] 丁涛, 林勇, 别朝红, 陈昌. 考虑主从分布式发电和拓扑重构的弹性微电网负荷恢复策略. 应用能源 2017;199:205-16.

> \[16] Taylor JA, Hover FS. Convex models of distribution system reconfiguration. IEEE Trans Power Syst 2012;27(3):1407-13.

\[16] Taylor JA, Hover FS. 配电系统重构的凸模型. IEEE 电力系统汇刊 2012;27(3):1407-13.

> \[17] Li Y, Xiao J, Chen C, Tan Y, Cao Y. Service restoration model with mixed-integer second-order cone programming for distribution network with distributed generations. IEEE Trans Smart Grid 2018;10(4):4138-50.

\[17] 李毅, 肖杰, 陈超, 谭毅, 曹阳. 含分布式电源的配电网服务恢复模型及其混合整数二阶锥规划方法. 电气与电子工程师协会智能电网汇刊 2018;10(4):4138-50.

> \[18] Romero R, Franco JF, Leão FB, Rider MJ, De Souza ES. A new mathematical model for the restoration problem in balanced radial distribution systems. IEEE Trans Power Syst 2015;31(2):1259-68.

\[18] Romero R, Franco JF, Leão FB, Rider MJ, De Souza ES. 一种用于平衡径向配电系统恢复问题的新数学模型. IEEE 电力系统汇刊 2015;31(2):1259-68.

> \[19] Wang Y, Xu Y, He J, Liu C-C, Schneider KP, Hong M, et al. Coordinating multiple sources for service restoration to enhance resilience of distribution systems. IEEE Trans Smart Grid 2019;10(5):5781-93.

\[19] 王毅, 徐阳, 何杰, 刘成成, Schneider KP, 洪明, 等. 协调多源以实现服务恢复以增强配电系统的韧性. IEEE 智能电网汇刊 2019;10(5):5781-93.

> \[20] Shen F, Lopez JC, Wu Q, Rider MJ, Lu T, Hatziargyriou ND. Distributed self-healing scheme for unbalanced electrical distribution systems based on alternating direction method of multipliers. IEEE Trans Power Syst 2019;35(3):2190-9.

\[20] Shen F, Lopez JC, Wu Q, Rider MJ, Lu T, Hatziargyriou ND. 基于交替方向乘子法的非平衡配电系统分布式自愈方案. IEEE 电力系统汇刊 2019;35(3):2190-9.

> \[21] Nejad RR, Sun W. Enhancing active distribution systems resilience by fully distributed self-healing strategy. IEEE Trans Smart Grid 2021;13(2):1023-34.

\[21] Nejad RR, Sun W. 通过完全分布式自愈策略增强主动配电系统的弹性. IEEE 智能电网汇刊 2021;13(2):1023-34.

> \[22] Huang Y, Li G, Chen C, Bian Y, Qian T, Bie Z. Resilient distribution networks by microgrid formation using deep reinforcement learning. IEEE Trans Smart Grid 2022;13(6):4918-30.

\[22] 黄毅, 李刚, 陈晨, 卞阳, 钱涛, 别朝红. 基于深度强化学习的微电网构建弹性配电网. IEEE 智能电网汇刊 2022;13(6):4918-30.

> \[23] Bai Y, Chen S, Zhang J, Xu J, Gao T, Wang X, et al. An adaptive active power rolling dispatch strategy for high proportion of renewable energy based on distributed deep reinforcement learning. Appl Energy 2023;330:120294.

\[23] 白洋, 陈思远, 张杰, 徐杰, 高天, 王鑫, 等. 基于分布式深度强化学习的高比例可再生能源自适应有功功率滚动调度策略. 应用能源 2023;330:120294.

> \[24] Yi Z, Wang X, Yang C, Yang C, Niu M, Yin W. Real-time sequential security-constrained optimal power flow: A hybrid knowledge-data-driven reinforcement learning approach. IEEE Trans Power Syst 2023.

\[24] 易志, 王晓, 杨超, 杨晨, 牛明, 尹伟. 实时序列安全约束最优潮流：一种混合知识数据驱动的强化学习方法. IEEE 电力系统汇刊 2023.

> \[25] Wang B, Li Y, Ming W, Wang S. Deep reinforcement learning method for demand response management of interruptible load. IEEE Trans Smart Grid 2020;11(4):3146-55.

\[25] 王斌, 李勇, 明伟, 王松. 基于深度强化学习的可中断负荷需求响应管理方法. 智能电网学报 2020;11(4):3146-55.

> \[26] Lu R, Hong SH, Yu M. Demand response for home energy management using reinforcement learning and artificial neural network. IEEE Trans Smart Grid 2019;10(6):6629-39.

\[26] 卢锐, 洪尚浩, 于明. 基于强化学习和人工神经网络的家庭能源管理需求响应. IEEE 智能电网汇刊 2019;10(6):6629-39.

> \[27] Mathew A, Roy A, Mathew J. Intelligent residential energy management system using deep reinforcement learning. IEEE Syst J 2020;14(4):5362-72.

\[27] Mathew A, Roy A, Mathew J. 基于深度强化学习的智能住宅能源管理系统. IEEE 系统杂志 2020;14(4):5362-72.

> \[28] Gao Y, Wang W, Yu N. Consensus multi-agent reinforcement learning for volt-var control in power distribution networks. IEEE Trans Smart Grid 2021;12(4):3594-604.

\[28] 高毅, 王伟, 余宁. 基于共识的多智能体强化学习在配电网络电压无功控制中的应用. IEEE 智能电网汇刊 2021;12(4):3594-604.

> \[29] Liu H, Wu W. Online multi-agent reinforcement learning for decentralized inverter-based volt-var control. IEEE Trans Smart Grid 2021;12(4):2980-90.

\[29] 刘辉, 吴伟. 基于去中心化逆变器的在线多智能体强化学习电压-无功控制. IEEE 智能电网汇刊 2021;12(4):2980-90.

> \[30] Cao D, Zhao J, Hu W, Yu N, Ding F, Huang Q, et al. Deep reinforcement learning enabled physical-model-free two-timescale voltage control method for active distribution systems. IEEE Trans Smart Grid 2021;13(1):149-65.

\[30] 曹东, 赵杰, 胡伟, 余宁, 丁峰, 黄强, 等. 基于深度强化学习的无物理模型双时间尺度电压控制方法在主动配电系统中的应用. IEEE 智能电网汇刊 2021;13(1):149-65.

> \[31] Xu P, Duan J, Zhang J, Pei Y, Shi D, Wang Z, et al. Active power correction strategies based on deep reinforcement learning-Part I: A simulation-driven solution for robustness. CSEE J Power Energy Syst 2021;8(4):1122-33.

\[31] 徐鹏, 段杰, 张杰, 裴洋, 石东, 王震, 等. 基于深度强化学习的有功功率校正策略——第一部分：一种基于仿真驱动的鲁棒性解决方案. 中国电机工程学会电力与能源系统学报 2021;8(4):1122-33.

> \[32] Chen S, Duan J, Bai Y, Zhang J, Shi D, Wang Z, et al. Active power correction strategies based on deep reinforcement learning-Part II: A distributed solution for adaptability. CSEE J Power Energy Syst 2021;8(4):1134-44.

\[32] 陈S, 段J, 白Y, 张J, 石D, 王Z, 等. 基于深度强化学习的有功功率校正策略-第二部分：适应性的分布式解决方案. 中国电机工程学会电力与能源系统学报 2021;8(4):1134-44.

> \[33] Du Y, Wu D. Deep reinforcement learning from demonstrations to assist service restoration in islanded microgrids. IEEE Trans Sustain Energy 2022;13(2):1062-72.

\[33] 杜毅, 吴东. 从示范中学习深度强化学习以辅助孤岛微电网的服务恢复. IEEE可持续能源汇刊 2022;13(2):1062-72.

> \[34] Yao Y, Zhang X, Wang J, Ding F. Multi-agent reinforcement learning for distribution system critical load restoration. In: 2023 IEEE power & energy society general meeting. IEEE; 2023, p. 1-5.

\[34] 姚Y, 张X, 王J, 丁F. 多智能体强化学习在配电系统关键负荷恢复中的应用. 见: 2023年IEEE电力与能源学会大会. IEEE; 2023, 第1-5页.

> \[35] Zhao T, Wang J. Learning sequential distribution system restoration via graph-reinforcement learning. IEEE Trans Power Syst 2021;37(2):1601-11.

\[35] 赵涛, 王杰. 通过图强化学习学习顺序配电系统恢复. IEEE电力系统汇刊 2021;37(2):1601-11.

> \[36] Wang J, Xu W, Gu Y, Song W, Green TC. Multi-agent reinforcement learning for active voltage control on power distribution networks. Adv Neural Inf Process Syst 2021;34:3271-84.

\[36] 王杰, 徐伟, 顾阳, 宋伟, Green TC. 多智能体强化学习在配电网络主动电压控制中的应用. 神经信息处理系统进展 2021;34:3271-84.

> \[37] Hessel M, Modayil J, Van Hasselt H, Schaul T, Ostrovski G, Dabney W, et al. Rainbow: Combining improvements in deep reinforcement learning. In: Proceedings of the AAAI conference on artificial intelligence, vol.32, 2018.

\[37] Hessel M, Modayil J, Van Hasselt H, Schaul T, Ostrovski G, Dabney W, 等. Rainbow: 深度强化学习中的改进组合. 见: AAAI人工智能会议论文集, 第32卷, 2018.

> \[38] Samvelyan M, Rashid T, De Witt CS, Farquhar G, Nardelli N, Rudner TG, et al. The starcraft multi-agent challenge. 2019, arXiv preprint arXiv:1902.04043.

\[38] Samvelyan M, Rashid T, De Witt CS, Farquhar G, Nardelli N, Rudner TG, 等. 星际争霸多智能体挑战. 2019, arXiv预印本 arXiv:1902.04043.

> \[39] Wang W, Yang T, Liu Y, Hao J, Hao X, Hu Y, et al. From few to more: Large-scale dynamic multiagent curriculum learning. In: Proceedings of the AAAI conference on artificial intelligence, vol. 34, 2020, p. 7293-300.

\[39] 王伟, 杨涛, 刘洋, 郝杰, 郝鑫, 胡勇, 等. 从少到多：大规模动态多智能体课程学习. 见: AAAI人工智能会议论文集, 第34卷, 2020年, 第7293-7300页.

> \[40] Hernandez-Leal P, Kartal B, Taylor ME. A survey and critique of multiagent deep reinforcement learning. Auton Agents Multi-Agent Syst 2019;33(6):750-97.

\[40] Hernandez-Leal P, Kartal B, Taylor ME. 多智能体深度强化学习综述与批判. 自主代理与多代理系统 2019;33(6):750-97.

> \[41] Huang S, Ontañón S. A closer look at invalid action masking in policy gradient algorithms. 2020, arXiv preprint arXiv:2006.14171.

\[41] 黄S, Ontañón S. 深入探讨策略梯度算法中的无效动作屏蔽. 2020, arXiv预印本 arXiv:2006.14171.

> \[42] Ratnam EL, Weller SR, Kellett CM, Murray AT. Residential load and rooftop pv generation: an Australian distribution network dataset. Int J Sustain Energy 2017;36(8):787-806.

\[42] Ratnam EL, Weller SR, Kellett CM, Murray AT. 住宅负荷与屋顶光伏发电：澳大利亚配电网络数据集. 国际可持续能源杂志 2017;36(8):787-806.

> \[43] Sunehag P, Lever G, Gruslys A, Czarnecki WM, Zambaldi V, Jaderberg M, et al. Value-decomposition networks for cooperative multi-agent learning. 2017, arXiv preprint arXiv:1706.05296.

\[43] Sunehag P, Lever G, Gruslys A, Czarnecki WM, Zambaldi V, Jaderberg M, 等. 用于协作多智能体学习的价值分解网络. 2017, arXiv预印本 arXiv:1706.05296.
