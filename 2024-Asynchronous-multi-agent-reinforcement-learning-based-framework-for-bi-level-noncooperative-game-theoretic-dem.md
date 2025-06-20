---
tags: []
parent: ""
collections:
    - '3 电网稳定'
$version: 73707
$libraryID: 1
$itemKey: 8ZW8VF6X

---
全文翻译：2024-Asynchronous multi-agent reinforcement learning-based framework for bi-level noncooperative game-theoretic demand response

# 基于异步多智能体强化学习的双级非合作博弈论需求响应框架

Yongxin Nie ， Student Member， IEEE， Jun Liu ， Senior Member， IEEE，

刘晓明 ， 赵宇 ， 任克政 ， 陈晨 ， IEEE高级会员

${Abstract}$ - 提出需求响应 （DR） 以解决配电网络中分布式能源 （DER） 和储能系统的不确定性。然而，目前的研究忽视了需求响应参与者的时间关系和非合作关系。本文提出了一种新的双层非合作 Stackelberg-Nash 动态博弈论框架，专门针对传统需求响应框架中忽视消费者利益和参与者行为的时间关系问题而设计。在这个创新框架中，提出了一种异步多智能体强化学习算法，并将其命名为 Stackelberg-Nash 多智能体近端策略优化算法 （SN-MAPPO）。SN-MAPPO 允许公用事业公司 （UC） 和消费者在拟议的框架内最大限度地利用他们的公用事业，最终导致 UC 和消费者的策略趋同于 Stackelberg-Nash 均衡。为了评估所提出的框架的有效性，使用真实数据进行模拟，涉及 1 个 UC 和 8 个电力消费者。仿真结果证实，所提出的需求响应机制在保障消费者利益的同时，促进了分布式能源的整合，有效缓解了负荷波动。

索引项-Stackelberg-Nash 博弈 异步多智能体强化学习 需求响应 - 多智能体近端策略优化

## 命名法

首字母缩略词 CTDE 集中培训 分散执行 DER 分布式能源 DR 需求响应 ESS 储能系统。LF 领导者追随者 MAPPO 多智能体近端策略优化 MARL 多智能体强化学习 P2P 点对点 PER 优先体验重放。PG 电网 POMG 部分可观察 马尔可夫游戏 SAC 软 演员 评论家 SNE Stackelberg-Nash Equilbrium

TUTT 分时分层关税 UC 公用事业公司集

## 功能

E预期值。

${\mathcal{L}}_{u}/{\mathcal{L}}_{c}$ UC/消费者的损失函数。

✓ 梯度值。

clip 限制策略更新的幅度。

rank 根据 advantage 值对 replay buffer 进行排序时的轨迹排名。

${\pi }_{u}/{\pi }_{c}/{\pi }_{i}$ UC/consumers 策略编码器 $i$ 的策略分布 -th consumer 解码器。

${\mathrm{D}}_{KL}\left( {\pi ,{\pi }^{\prime }}\right)$ KL 新政策 $\pi$ 和旧政策 ${\pi }^{\prime }$ 的分歧。

${A}_{u}/{A}_{c}/{A}_{c,i}$ UC/consumers/ $i$ -th consumer 的优势功能。

${V}_{u}/{V}_{i}$ UC/consumers 的状态值函数。指标

$i/ - i$ 使用者序列号/除 $i$ -th 使用者以外的使用者。

$t$ 时间步长。

${\mathcal{A}}_{u}/{\mathcal{A}}_{c}/{\mathcal{A}}_{i}$ UC/consumers/i th consumer 的作集。

${\mathcal{D}}_{S}/{\mathcal{D}}_{N,i}$ UC/consumers 轨迹的重放缓冲区。

这组使用者。

${\mathcal{O}}_{u}/{\mathcal{O}}_{c}/{\mathcal{O}}_{i}$ UC/consumers/ $i$ -th consumer 的观测集。

PPOMG 的过渡核。

SPOMG 的状态集。

${\mathbf{X}}^{U}/{\mathbf{X}}_{i}^{C}$ 第 $\mathrm{{UC}}/i$ 个消费者策略的可行域。

## 变量

$\Delta {p}_{i,t}$ 第 $i$ 个消费者的需求响应能力。

$\Delta {p}_{i,t}^{l}/\Delta {p}_{i,t}^{s}$ 第 $i$ 个消费者的 DR 可移动/可减少负载。

${\delta }_{t}^{u}/{\delta }_{t}^{c,i}$ 消费者和 UC 的 TD 误差。

${\eta }_{c}/{\eta }_{o}/{\eta }_{s}$ ESS 的充电/放电/衰减效率指数。

${\kappa }_{i}/{\kappa }_{u}$ 体验重播的优先级。

${\lambda }_{p}/{\lambda }_{s}$ SOC/ESS 充电功率的阈值。

${\mathcal{J}}_{u}/{\mathcal{J}}_{c,i}$ UC/i 消费者的效用函数。

${\mathcal{T}}^{a}/{\mathcal{T}}^{s}$ 废弃的 DER 电力/DR 补贴边际效应的价格。

***

这项工作由中国国家自然科学基金 （No. 52177111） 资助。

Yongxin Nie， Jun Liu， Xiaoming Liu， Yu Zhao， Kezheng 任 和 Chen Chen 就职于中国710049安习习安交通大学电气工程学院陕西省智能电网重点实验室。（通讯作者：刘军 <eeliujun@mail.xjtu.edu.cn>）

***

请参阅 https\://www\.ieee.org/publications/rights/index.html

${\mathcal{T}}_{t}^{b}/{\mathcal{T}}_{t}^{d}\left( {p}_{t}^{g}\right)$ 从 PG 购买电力/出售 DER 电力的关税。

${T}^{r}/{T}_{i}^{c}$ DR 灵活性购买/电力舒适度的价格。

${\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right)$ TUTT 代表 $i$ -th 消费者。

${\omega }_{u}/{\omega }_{c}/{\omega }_{i}$ UC/consumers 编码器/解码器的批评网络参数。上标 ${}^{\prime }$ 表示更新前的旧参数。

${\sigma }_{u}/{\sigma }_{i}$ 对 UC 策略更新和消费者策略更新的特定策略轨迹进行采样的概率。

${\mathbf{x}}_{u}/{\mathbf{x}}_{i}$ 第 1 位消费者的策略。星号上标 $*$ 表示 SNE 策略。

${\theta }_{u}/{\theta }_{c}/{\theta }_{i}$ UC/consumers 编码器/解码器的策略网络参数。上标 ${}^{\prime }$ 表示更新前的旧参数。

${\varepsilon }_{u}/{\varepsilon }_{c}/{\varepsilon }_{i}$ UC 策略/-消费者策略编码器/消费者策略解码器的收敛条件。

${a}_{t}^{u}/{a}_{t}^{c}$ UC/消费者的行动。

${C}_{t}^{g}/{C}_{t}^{r}$ 用于 DER 吸收的 UC 利润/用于购买 DR 灵活性服务的 UC 成本。

${C}_{i,t}^{p}/{C}_{i,t}^{d}/{C}_{i,t}^{r}$ 消费者 $i$ -th 公用事业公司购买 UC 电力/DER 弃电/参与 DR。

${C}_{t}^{s}/{C}_{t}^{m}$ UC 向消费者售电的收入/UC 从 PG 购买电力的成本。${E}_{i,t}^{u}$ 本月第 $i$ 个消费者在时间步 $t$ 之前的消耗电量。${o}_{t}^{u}/{o}_{i,t}^{c}$ 对 UC/消费者的观察。${p}_{N}^{c}$ ESS 的额定充电功率。${p}_{t}^{a}$ 在 $t$ 放弃 DER 电源。${p}_{t}^{b}/{p}_{t}^{s}/{p}_{t}^{a}$ 从 PG 购买的电力/以 $t$ 的价格出售给消费者。${p}_{t}^{c}/{p}_{t}^{o}$ ESS 的充电/输出功率。${p}_{t}^{d}/{p}_{t}^{e}$ DER/UC 的 ESS 充电功率。${p}_{t}^{m}$ UC 预期消费者 DR 功率。${p}_{i,t}^{ * }/\Delta {p}_{i,t}^{ * }$ 数据集中的负载数据/最大 DR 功率。${p}_{i,t}^{u}/{p}_{i,t}^{r}$ 从 UC/DER 购买的电力。${R}^{u}/{R}_{i,t}^{c}$ UC/ $i$ -th 消费者的奖励功能。${S}_{t}/{S}_{\max }$ ESS 的充电状态/最大 ESS 充电状态。T 所有时间步长。

## I. 引言

传统的电力能源系统运行结构是自上而下的，发电机通过电网 （PG） 平衡电力需求 \[1]。随着 DER 和储能系统 （ESS） 的比例越来越高，需要更复杂的控制方法来实现平衡 \[2]。利用智能电网中的电力负载作为额外的灵活资源来实现功率和能源平衡，这被称为需求响应 （DR）。DR 在高峰期减少电力需求的能力减轻了传统发电的压力 \[3]。此外，DR 通过其对电力消耗的适应性调节，在提高 DER 消耗方面发挥着关键作用 \[4]。

根据 DR 实施方法，主要的 DR 激励措施可分为基于价格的 DR （PDR） 和基于激励的 DR （IDR） \[5]、\[6]。参考文献 \[5] 从公用事业公司的角度提出了一种新的基于激励的 DR 模型，以实现需求响应资源的系统级调度。参考文献 \[7] 为 UC 提出了一种定制的回扣包定价机制，以奖励支持电力系统的消费者。

根据 DR 框架的架构，现有的 DR 通常分为领导者-跟随者 （LF） 结构和点对点 （P2P） 结构 \[8]、\[9]。LF 结构通常使用分层优化算法来解决 \[10]。一些双层 DR 优化算法经常忽视不同利益相关者的不同目标，导致缺乏消费者的积极参与 \[11]。作为补救措施，研究人员将 Stackelberg 游戏引入 DR 的 LF 形式，消费者在公用事业公司战略下追求最大的效用。Stackelberg 博弈将 DR 目标从解决全局最优策略转变为解决 Stackelberg 均衡策略 \[12]。上述 DR 机制更符合实际场景，但经常引入非凸约束。在一些简单的情况下，可以采用对偶理论将非凸约束转换为具有静止点的可求解 KKT 条件 \[13]。然而，这种方法仍然缺乏通用性，并且通常难以解决。

博弈论用于研究理性市场参与者在电力市场中的战略行为之间的相互作用，通常通过均衡规划模型进行分析 \[14]。参考文献 \[15] 和 \[16] 在博弈论下对 P2P 结构 DR 进行建模，其中 UC 和消费者参与共同博弈，最终 UC 和消费者策略收敛到纳什均衡策略。与静态 LF 结构分层 DR 不同，参考文献 \[17] 引入了基于动态规划的 P2P 结构 DR，其解是 NP-hard \[11]。

随着深度学习的进步，深度强化学习 （DRL） 为动态规划的 DR 提供了一种数据驱动的方法 \[18]。近年来，博弈论和强化学习 （RL） 集成到电力系统中因其在解决该领域内各种挑战方面的潜力而受到广泛关注。这些方法提供了创新的解决方案，用于在面对日益增加的复杂性和分散化时提高系统效率、可靠性和适应性。在 \[14] 中，提出了一种分层的 Nash 分布式框架，以实现市场参与者和独立系统运营商之间的交互。\[19] 提供了一种基于平均场博弈论的 P2P 多能量交易强化学习算法。

此外，考虑到随着 DR 消费者的增加，策略空间呈指数级增长，多智能体强化学习 （MARL） 被应用于提供一种解决 P2P 结构 DR 下 DR 均衡策略的方法 \[20]。然而，大多数现有的基于 MARL 的 DR 主要适用于 P2P 结构的 DR，其中所有参与者都根据 DR 策略同步行动，在作过程中没有策略更改 \[21]，\[22]。在基于 MARL 的 DR 框架中，DR 参与者（UC 和消费者）的时间关系通常通过假设 DR 参与者同时行动来简化，这导致部分代理在学习过程中获得的信息较少，从而导致局部最优 \[23]，\[24]。表 I 总结了关于 DR 的大量学术研究

表 I

DR 的现有文献总结

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------------------- | ---------------------- | --------------------- | ----------------------------- | ----------------------------- |
| Papers              | Coordination Structure | Implementation Method | DR function                   | Algorithm                     |
| \[25], \[5], \[26]  | P2P                    | Nondispatchable       | Electricity Peak Reduction    | Game Theoretical Optimization |
| \[27], \[28], \[17] | P2P                    | Nondispatchable       | Electricity Peak Reduction    | MARL                          |
| \[29]               | LF                     | Dispatchable          | Frequency Regulation          | Hierarchical Optimization     |
| \[30], \[31]        | LF                     | Nondispatchable       | Voltage Regulation            | Single-Agent DRL              |
| This paper          | Mixed                  | Mixed                 | Energy Consumption Scheduling | MARL                          |


从上一段可以明显看出，目前 DR 的研究存在几个关键空白：首先，大多数传统的博弈论 DR 框架都是基于 Stackelberg 博弈或 Nash 博弈的。此外，DR 参与者大多完全合作。基于 Stackelberg-Nash 博弈的非合作 DR 框架缺乏研究。其次，在传统的基于 MARL 的 DR 方法中，基于代理进行同步作的假设，MARL 代理动作的时间关系通常被忽略。在 Stackelberg 博弈的假设下，同步智能体的动作忽略了追随者对领导者策略的最佳响应条件。

针对上述差距，该文提出一种新的双级动态非合作UC-消费者博弈理论DR机制和一种新的异步MARL算法，命名为Stackelberg-Nash多智能体近端策略优化算法（SN-MAPPO）。所提出的 DR 框架的框图如图 1 所示。主要贡献可以总结如下：

1） 提出了一种基于 Stackelberg-Nash 博弈的新型双级非合作博弈论 DR 框架，其中 UC 和消费者在博弈中依次作为独立的上层和下层代理，以最大化其效用。所提出的框架能够增强 DER 与 PG 的整合，并减少配电网中的负载波动。

2） 所提出的 DR 框架被表示为一个部分可观察的马尔可夫博弈，并为 UC 和消费者设计了 2 种不同的奖励函数，以提高所提出的 DR 框架在 DER 弃电和减少电力调峰方面的性能。

3） 提出了一种考虑时间结构的新型异步 MARL 方法，其中 UC 和消费者异步更新他们的策略网络。

4） 拟议的 DER 削减和负载峰值降低的 DR 框架的集中培训和分散执行 （CTDE）。在真实电网上进行了案例研究，证明了所提出的 DR 机制和求解方法的有效性和进步性。

本文的其余部分组织如下。第二部分模型 Stackelberg-Nash 游戏 DR 模型。Section III 将 Section II 的 DR 模型重构为时间顺序部分可观察马尔可夫博弈 （POMG），并提出了一种新的 MARL 算法来解决 POMG 的 SNE。第四部分介绍了案例设置和仿真结果。最后，第五节讨论了这项工作的结论。

![\<img src="attachments/EKGUIYCQ.jpg" alt="" width="785" height="500" data-attachment-key="EKGUIYCQ" ztype="zimage"> | 785](attachments/EKGUIYCQ.jpg)

图 1.拟议的 DR 框架的时间关系。

## 二.基于 STACKELBERG-NASH 博弈的双层动态需求响应框架

尽管 \[31] 中早就提出了多级 DR 模型，但传统的 DR 框架是基于静态多级博弈的。考虑到 UC 与其消费者之间的动态交互，本节提出了一种新的 UC 和消费者 DR 框架，即 Stackelberg-Nash 游戏，其中上层表示 UC 收入优化，即 UC-consumers Stackelberg 游戏，下层是消费者成本优化，即消费者 Nash 游戏。UC 和消费者在提出的动态分层非合作博弈框架中交替进行游戏，并最终收敛到 Stackelberg-Nash 均衡 （SNE）。本文提出的基于 Stackelberg-Nash 博弈的动态双层博弈模型可以更准确地描述时间范围内 DR 的动态博弈过程。

## A. 双层需求响应博弈框架

本文提出的双层动态博弈框架超越了传统的静态博弈模型，在静态博弈模型中，UC 和消费者通过预测累积效用来做出决策。与依赖于 MARL 算法的传统并行博弈框架相比，本节中提出的框架更符合真实的 DR 场景。

在这个框架中，UC 向消费者执行自己的策略 ${\mathbf{x}}_{u}$。UC 在跟随者提供最优响应策略 ${\mathbf{x}}_{i}^{ * } =$ $\arg \min {\mathcal{J}}_{c,i}\left( {{\mathbf{x}}_{i} \mid {\mathbf{x}}_{-i},{\mathbf{x}}_{u}^{ * }}\right)$ 的前提下优化其目标，而消费者倾向于根据 UC 的策略 ${\mathbf{x}}_{u}$ 将其预期成本降至最低。如上所述，双层 Stackelberg-Nash DR 框架被正式化为 \[12]、\[25]、\[28]：

$$
\mathop{\max }\limits_{{\mathbf{x}}_{u}}{\mathcal{J}}_{u}\left( {{\mathbf{x}}_{u} \mid  {\mathbf{x}}_{i}^{ * }}\right) ,{\mathbf{x}}_{u} \in  {\mathbf{X}}^{U} \tag{1}
$$

$$
\mathop{\min }\limits_{{\mathbf{x}}_{i}}{\mathcal{J}}_{c,i}\left( {{\mathbf{x}}_{i} \mid  {\mathbf{x}}_{-i},{\mathbf{x}}_{u}}\right) ,{\mathbf{x}}_{i} \in  {\mathbf{X}}_{i}^{C}\left( {\mathbf{x}}_{u}\right) ,i \in  \mathcal{I} \tag{2}
$$

其中 ${\mathbf{x}}_{u}$ 和 ${\mathbf{x}}_{i}$ 表示 UC 和第 $i$ -th 消费者的策略，这是 UC 和消费者在时间范围内行动的向量。${\mathbf{x}}_{u}^{ * }$ 和 ${\mathbf{x}}_{i}^{ * }$ 表示 UC 和消费者的最佳响应策略。${\mathbf{X}}^{U}$ 表示 UC 的可行域。${\mathbf{X}}_{i}^{C}\left( {\mathbf{x}}_{u}\right)$ 表示 UC 策略下消费者的可行域 ${\mathbf{x}}_{u}.{\mathcal{J}}_{c,i}\left( {{\mathbf{x}}_{i} \mid {\mathbf{x}}_{-i},{\mathbf{x}}_{u}}\right)$ 表示 $i$ -th 消费者在其他消费者策略下的效用 ${\mathbf{x}}_{-i}$ 和 UC ${\mathbf{x}}_{u}.{\mathcal{J}}_{u}\left( {{\mathbf{x}}_{u} \mid {\mathbf{x}}_{i}^{ * }}\right)$ 表示 UC 在消费者最优响应策略下的效用。$\mathcal{I}$ 表示使用者集。索引 $- i$ 表示除第 $i$ 个使用者以外的使用者。（1） 和 （2） 中有 2 个假设对齐：

假设 1.由 ${UC}$ 管理的所有消费者都积极参与 DR，并且对其负载计划完全合理，期望最大限度地发挥其优势。

假设 2.消费者没有配备 DER 或 ESS。DER 和 ESS 的数据对消费者和 UC 完全可访问。

假设 1 保证消费者将调整他们的策略以最大化他们的效用。假设 2 的目的是简化可行域。但是，如果消费者的 ESS 和 DER 数据足够，则可以丢弃假设 2。

值得注意的是，虽然模型表示为 （1） 和 （2），但上下层的相应主体交替作用。此外，UC 和消费者在优化自己的实用程序 ${\mathcal{J}}_{u}\left( {{\mathbf{x}}_{u} \mid {\mathbf{x}}_{i}^{ * }}\right)$ 和 ${\mathcal{J}}_{c,i}\left( {{\mathbf{x}}_{i} \mid {\mathbf{x}}_{-i},{\mathbf{x}}_{u}^{ * }}\right)$ 时，对系统的未来状态一无所知。因此，所提出的 DR 框架应被视为基于模型预测的动态规划问题。统一通信和消费者都需要在整个事件范围 $T$ 中预测他们的效用 ${\mathbb{E}}_{t}\left\lbrack {{\mathcal{J}}_{u}\left( {{\mathbf{x}}_{u} \mid {\mathbf{x}}_{i}^{ * }}\right) }\right\rbrack$ 和 ${\mathbb{E}}_{t}\left\lbrack {{\mathcal{J}}_{c,i}\left( {{\mathbf{x}}_{i} \mid {\mathbf{x}}_{-i},{\mathbf{x}}_{u}^{ * }}\right) }\right\rbrack$，以便做出决策 \[18]。

所提出的 DR 框架的 SNE 意味着在多人游戏中，所有消费者都采用了 UC 的最佳响应策略，没有人可以通过改变策略来提高他们的性能 \[12]，\[13]。UC 可以：

$$
{\mathcal{J}}_{u}\left( {{\mathbf{x}}_{u} \mid  {\mathbf{x}}_{i}^{ * }}\right)  \leq  {\mathcal{J}}_{u}\left( {{\mathbf{x}}_{u}^{ * } \mid  {\mathbf{x}}_{i}^{ * }}\right)  \tag{3}
$$

$$
{\mathcal{J}}_{c,i}\left( {{\mathbf{x}}_{i} \mid  {\mathbf{x}}_{-i},{\mathbf{x}}_{u}}\right)  \leq  {\mathcal{J}}_{c,i}\left( {{\mathbf{x}}_{i}^{ * } \mid  {\mathbf{x}}_{-i},{\mathbf{x}}_{u}}\right)  \tag{4}
$$

## B. 公用事业公司的目标

在提出的 DR 框架中，UC 优化模型的数学描述如下所示：

$$
\mathop{\max }\limits_{{\mathbf{x}}_{u}}{\mathcal{J}}_{u}\left( {{\mathbf{x}}_{u} \mid  {\mathbf{x}}_{i}^{ * }}\right)  = \mathop{\sum }\limits_{{t = 0}}^{T}\left( {{C}_{t}^{s} + {C}_{t}^{m} + {C}_{t}^{g} + {C}_{t}^{r}}\right)  \tag{5}
$$

$$
\text{ s.t. }\left\{  \begin{array}{l} {\mathbf{x}}_{i}^{ * } = \arg \min {\mathcal{J}}_{c,i}\left( {{\mathbf{x}}_{i} \mid  {\mathbf{x}}_{-i},{\mathbf{x}}_{u}^{ * }}\right) \\  {\mathbf{x}}_{u} \in  {\mathbf{X}}^{U} \end{array}\right.  \tag{6}
$$

其中 ${\mathbf{x}}_{u} = \left\{ {{p}_{t}^{b},{p}_{t}^{s},{p}_{t}^{a},{p}_{t}^{c},{p}_{t}^{d},{p}_{t}^{e},{p}_{t}^{o},{p}_{t}^{m}}\right\} ,t = 1,\ldots ,T.{p}_{t}^{b}$ 表示从 PG 购买的电力。${p}_{t}^{s}$ 表示出售给消费者的电力。${p}_{t}^{a}$ 表示废弃的 DER 幂。${p}_{t}^{c}$ 和 ${p}_{t}^{o}$ 表示 PG 的 ESS 的充电功率和输出功率。${p}_{t}^{d}$ 表示 ESS 的充电功率来自 DER。${p}_{t}^{e}$ 表示 ESS 的充电功率来自 UC。${p}_{t}^{o}$ 表示 ESS 输出功率。${p}_{t}^{m}$ 表示预期的 DR 功率。（5） 表示 UC 的收入，由 4 个部分组成：首先，向消费者出售电力的收入可以表示为 ${C}_{t}^{s}$ ：

$$
{C}_{t}^{s} = \mathop{\sum }\limits_{i}^{\left| \mathcal{I}\right| }\left( {{\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right)  \cdot  {p}_{i,t}^{u}}\right)  \tag{7}
$$

其中，分时分层费率 （TUTT） 策略 ${\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right)$ 是与峰谷期 $t$ 和消费者用电量 ${E}_{i,t}^{u} = {\int }_{0}^{t}{p}_{i,t}^{U}\mathrm{\;d}t$ 相关的分段线性函数。${p}_{i,t}^{u}$ 表示消费者从 UC 购买的电力。其次，从 PG 购买电力的成本可以表示为 ${C}_{t}^{m}$ \[13]：

$$
{C}_{t}^{m} =  - \left( {{\mathcal{T}}_{t}^{b} + \varepsilon }\right)  \cdot  {p}_{t}^{b} \tag{8}
$$

其中 UC 以实时市场价格 ${\mathcal{T}}_{t}^{b}$ 从铂族公司购买电力，不确定性为 $\varepsilon$ ，该价格是从实时电价的截断正态分布中抽样的，均值为 0，方差为 0.03，边界为 $3\%$。第三，DER 吸收的利润可以表示为 ${C}_{t}^{g}$ ：

$$
{C}_{t}^{g} = {\mathcal{T}}_{t}^{d}\left( {p}_{t}^{g}\right)  \cdot  \left( {{p}_{t}^{g} - {p}_{t}^{a} - {p}_{t}^{d}}\right)  - {\mathcal{T}}^{a} \cdot  {p}_{t}^{a} \tag{9}
$$

其中 ${\mathcal{T}}_{1}^{d} + {\mathcal{T}}_{2}^{d} \cdot {p}_{t}^{g} \cdot {\mathcal{T}}_{1}^{d}$ 和 ${\mathcal{T}}_{2}^{d}$ 是 DER 吸收的激励关税的因素。UC 通过消费者负载计划和 ESS 充电最大限度地提高 DER 吸收。超额的 DER 生成以成本 ${\mathcal{T}}_{t}^{d}\left( {p}_{t}^{g}\right)$ 丢弃。废弃的 DER 为 ${p}_{t}^{a}.{p}_{t}^{g}$ 表示 DER 代。${C}_{t}^{g}$ 包括 DER 吸收成本和 DER 削减收入的收入。最后，UC 以资费 ${\mathcal{T}}^{r}$ 从消费者那里购买 DR 灵活性服务的成本可以表示为 ${C}_{t}^{r}$ \[10]：

$$
{C}_{t}^{r} = {\mathcal{T}}^{s}\frac{{\left( \mathop{\sum }\limits_{i}^{\left| \mathcal{I}\right| }\Delta {p}_{i,t}\right) }^{2}}{{p}_{t}^{m}} - {\mathcal{T}}^{r}\mathop{\sum }\limits_{i}^{\left| \mathcal{I}\right| }\Delta {p}_{i,t} \tag{10}
$$

其中 ${\mathcal{T}}^{s}$ 表示 DR 福利费率。$\Delta {p}_{i,t}$ 表示消费者的 DR 能力 $\Delta {p}_{i,t}$ 。UC 为消费者的 DR 灵活性服务提供激励，资费 ${\mathcal{T}}^{r} \cdot \mathop{\sum }\limits_{i}^{\left| \mathcal{I}\right| }\Delta {p}_{i,t}/{p}_{t}^{m}$ 表示消费者的 DR 功率 $\Delta {p}_{i,t}$ 与 UC 的目标 DR 功率 ${p}_{t}^{m}$ 的比率。

在消费者的最优反应下，如图 （6） 所示的 UC 策略的可行性域定义如下：

$$
{\mathbf{X}}^{U} = \left\{  \left( {{p}_{t}^{b},{p}_{t}^{s},{p}_{t}^{a},{p}_{t}^{c},{p}_{t}^{d},{p}_{t}^{e},{p}_{t}^{o},{p}_{t}^{m}}\right) \right.
$$

$$
0 \leq  {p}_{t}^{a} < {p}_{t}^{b}, \tag{11}
$$

$$
0 \leq  {p}_{t}^{e} \leq  {p}_{t}^{c},0 \leq  {p}_{t}^{d} \leq  {p}_{t}^{c}, \tag{12}
$$

$$
0 \leq  {p}_{t}^{o} \leq  {p}_{\max }^{o}\text{,} \tag{13}
$$

$$
0 \leq  {p}_{t}^{o} \leq  {p}_{\max }^{m}, \tag{14}
$$

$$
0 \leq  {p}_{t}^{m} < {p}_{t}^{s}, \tag{15}
$$

$$
{p}_{t}^{c} = f\left( {S}_{t}\right) ,{p}_{t}^{c} = {p}_{t}^{e} + {p}_{t}^{d} \tag{16}
$$

$$
{p}_{t}^{b} + {p}_{t}^{g} + {p}_{t}^{o} = {p}_{t}^{a} + {p}_{t}^{c} + {p}_{t}^{s} \tag{17}
$$

$$
\left. {{p}_{t}^{c} \times  {p}_{t}^{o} = 0}\right\}  \text{.} \tag{18}
$$

其中 （12） 和 （13） 表示充电范围和输出 ESS 功率。（14） 表示 DR 预期功率的裕量。（17） 表示配电网的能量平衡。（18） 限制 ESS 同时充电和放电。在（16）中，充电功率是 ESS 充电状态 ${S}_{t - 1}$ 的函数，ESS 可以通过 DER 或 PG 充电。ESS 系统的转换函数定义为 （19） \[32]：

$$
\left\{  \begin{array}{l} {S}_{t} = \left( {1 - {\eta }_{s}}\right) {S}_{t - 1} + {\eta }_{c}{p}_{t - 1}^{c} + {\eta }_{o}{p}_{t - 1}^{o} \\  0 \leq  {S}_{t} \leq  {S}_{\max },0 \leq  {S}_{t - 1} \leq  {S}_{\max } \end{array}\right.  \tag{19}
$$

其中 ${S}_{t}$ 表示 ESS 充电状态 （SOC）。${S}_{\max }$ 表示最大 SOC。${\eta }_{c}$ 和 ${\eta }_{o}$ 是充电和输出功率效率系数。${\eta }_{s}$ 表示 SOC 衰减因子。输出功率是裕量中的连续变量，UC 可以确定 ESS 系统的输入和输出功率。在本文讨论的时间粒度上，充电功率可以表示为分段函数 $f\left( {S}_{t}\right)$，如 （20）：

$$
{p}_{t}^{c} = \left\{  \begin{array}{ll} 0 & {S}_{t} = {S}_{\max } \\  {p}_{N}^{c} & {S}_{t} < {\lambda }_{s}{S}_{\max }, \\  {\lambda }_{p}{p}_{N}^{c} & {S}_{t} \geq  {\lambda }_{s}{S}_{\max } \end{array}\right.  \tag{20}
$$

其中 ${\lambda }_{s}$ 是 ESS SOC 的阈值。${\lambda }_{p}$ 是当 ESS 的 SOC 达到阈值 ${\lambda }_{s}{S}_{\max }$ 时 ESS 的充电功率比，其中 $0 < {\lambda }_{p} < 1.{p}_{N}^{c}$ 表示 ESS 的额定充电功率。

## C. 消费者的目标和约束

第 $i$ 个消费者的优化模型显示为 （21）。如 （22） 所示，策略 ${\mathbf{x}}_{i}$ 需要符合约束集 ${\mathbf{X}}^{C}$ 。

$$
\mathop{\min }\limits_{{\mathbf{x}}_{i}}{\mathcal{J}}_{c,i}\left( {{\mathbf{x}}_{i} \mid  {\mathbf{x}}_{-i},{\mathbf{x}}_{u}}\right)  = \mathop{\sum }\limits_{t}^{T}\left( {{C}_{i,t}^{p} + {C}_{i,t}^{d} + {C}_{i,t}^{r}}\right)  \tag{21}
$$

$$
\text{s.t.}{\mathbf{x}}_{i} \in  {\mathbf{X}}_{i}^{C}\left( {\mathbf{x}}_{u}\right) ,i \in  \mathcal{I} \tag{22}
$$

其中 ${\mathbf{x}}_{i} = \left\{ {{p}_{i,t}^{r},{p}_{i,t}^{u},\Delta {p}_{i,t}^{l},\Delta {p}_{i,t}^{s}}\right\} ,i \in \mathcal{I},t = 1,\ldots ,T$ .消费者收到 UC 策略 ${\mathbf{x}}_{u}$ 作为消费者成本优化的先决条件。（21） 由 3 个部分组成：首先，从 UC 购买电力的支出可以表示为 ${C}_{i,t}^{p}$ ：

$$
{C}_{i,t}^{p} = {\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right)  \cdot  {p}_{i,t}^{u} \tag{23}
$$

其中 ${p}_{i,t}^{U}$ 表示从 UC 购买的第 $i$ 个消费者的电量。其次，消费者在关税 ${\mathcal{T}}^{d}$ 吸收 DER 发电的奖励可以表示为 ${C}_{i,t}^{d}$ ：

$$
{C}_{i,t}^{d} = {\mathcal{T}}_{t}^{d}\left( {p}_{t}^{g}\right)  \cdot  {p}_{i,t}^{r} \tag{24}
$$

消费者优先考虑 DER 电源，因为与 UC 电源相比，它的价格较低，体现了消费者之间的非合作博弈关系。${p}_{i,t}^{r}$ 表示消费者从 DER 购买的电力，DER 关税 ${\mathcal{T}}_{t}^{d}\left( {p}_{t}^{g}\right) = {\mathcal{T}}_{1}^{d} + {\mathcal{T}}_{2}^{d} \cdot {p}_{t}^{g} \cdot {\mathcal{T}}_{1}^{d}$ 和 ${\mathcal{T}}_{2}^{d}$ 表示 DER 关税的斜率和截距参数。第三，参与负载计划和负载转移的消费者效用函数可以表示为 ${C}_{i,t}^{l}$ ：

$$
{C}_{i,t}^{l} = \frac{{\mathcal{T}}_{i}^{c}{\left( \Delta {p}_{i,t}\right) }^{2}}{{p}_{i,t}^{u}} - {\mathcal{T}}^{r}\Delta {p}_{i,t} \tag{25}
$$

其中 ${\mathcal{T}}_{i}^{c}$ 表示电费舒适。$\Delta {p}_{i,t} =$ $\Delta {p}_{i,t}^{l} + \Delta {p}_{i,t}^{s} \cdot \Delta {p}_{i,t}^{l}$ 表示可降低的负载功率。$\Delta {p}_{i,t}^{s}$ 表示可转移负载。消费者通过需求响应激励措施共同参与 DR。

基于 Stackelberg 博弈的假设，消费者策略的可行性域与 UC 策略 ${\mathbf{x}}_{u}$ 相关。因此，消费者策略 ${\mathbf{X}}_{i}^{C}$ 的可行性域定义为：

$$
{\mathbf{X}}_{i}^{C}\left( {\mathbf{x}}_{u}\right)  = \left\{  \left( {{p}_{i,t}^{r},{p}_{i,t}^{u},\Delta {p}_{i,t}^{l},\Delta {p}_{i,t}^{s}}\right) \right.
$$

$$
{p}_{t}^{g} = {p}_{t}^{e} + \mathop{\sum }\limits_{i}^{\left| \mathcal{I}\right| }{p}_{i,t}^{r},{p}_{t}^{s} = {p}_{t}^{d} + \mathop{\sum }\limits_{i}^{\left| \mathcal{I}\right| }{p}_{i,t}^{u} \tag{26}
$$

$$
{p}_{i,t}^{ * } = {p}_{i,t}^{u} + {p}_{i,t}^{r} + \Delta {p}_{i,t} \tag{27}
$$

$$
- \Delta {p}_{i,t}^{ * } \leq  \Delta {p}_{i,t}^{l} + \Delta {p}_{i,t}^{s} \leq  \Delta {p}_{i,t}^{ * }, \tag{28}
$$

$$
0 \leq  \Delta {p}_{i,t}^{l} \leq  \Delta {p}_{i,t}^{ * }, \tag{29}
$$

$$
\mathop{\sum }\limits_{{t = 0}}^{T}\Delta {p}_{i,t}^{s} = 0 \tag{30}
$$

$$
\left. {\mathop{\sum }\limits_{i}^{\left| \mathcal{I}\right| }\Delta {p}_{i,t} \leq  {p}_{t}^{m}}\right\}  . \tag{31}
$$

其中 （26） 代表从 DER 购买的电力和 DER 的电力平衡。（28） 表示消费者积极参与需求响应。（29） 表示可调节的负载约束，（30） 表示可移动的负载约束。（31） 表示所有消费者的 DR 功率不会超过 UC 预期的 DR 功率。

综上所述，拟议的 DR 框架被表述为 Stackelberg-Nash 博弈。UC 通过调节 ESS、放置 DR 容量、TUTT 和 DR 激励措施来参与 UC 消费者 Stackelberg 游戏，而消费者则通过负载计划、负载转移以及决定 DER 和 UC 的功率比来参与 UC 消费者 Stackelberg 游戏。除了 Stackelberg 游戏之外，消费者之间还存在一种不合作的 Nash 游戏，他们争夺 DER 消费和 DR 激励。我们有以下假设来确保优化问题 （1） 和 （2） 有一个可行的解。

假设 3.Stackelberg-Nash 博弈 （1） 和 （2） 至少存在一个可行的解决方案。

所提出的 DR 框架的潮流图如图 2 所示。

## 三.异步 STACKELBERG-NASH 多智能体强化学习算法

UC作突出了动态的 Stackelberg-Nash 博弈过程，影响需求响应的可行域。

![\<img src="attachments/773LK3MF.jpg" alt="" width="686" height="500" data-attachment-key="773LK3MF" ztype="zimage"> | 686](attachments/773LK3MF.jpg)

图 2.拟议的 DR 框架的功率流图。

统一通信优化依赖于消费者的最优响应，这可能会给传统优化算法带来非凸性挑战 \[17]，\[33]。这种复杂性阻碍了收敛并限制了可扩展性。相反，数据驱动的强化学习算法提供了一种更通用和通用的解决方案。在本节中，第二节中提出的非合作博弈论 DR 框架表示为部分可观察马尔可夫博弈 （POMG） \[34]、\[35]。然后，提出了一种异步 MARL 算法来解决所提出的双层 DR 框架中的 Stackelberg-Nash equilbrium （SNE） 策略，即 SN-MAPPO。SN-MAPPO 包含两个异步步骤：Stackelberg 异步强化学习和 Nash MARL。这两个步骤通过消费者策略网络的 shared parameter 部分相互连接。

## A. 双层部分可观察马尔可夫博弈

为了通过 SN-MAPPO 解决 SNE，第二节中提出的 DR 框架被转化为马尔可夫决策过程 （MDP） 的多代理扩展，这是部分可观察的马尔可夫博弈 \[36]。基于 POMG，UC 和每个消费者都被视为 POMG 代理。一般和异步移动马尔可夫博弈的情节版本由元组 $\left( {\mathcal{S},{\mathcal{O}}_{u},{\mathcal{O}}_{c} = {\left\{ {\mathcal{O}}_{i}\right\} }_{i \in \mathcal{I}},{\mathcal{A}}_{u},{\mathcal{A}}_{c} = }\right.$ $\left. {{\left\{ {\mathcal{A}}_{i}\right\} }_{i \in \mathcal{I}},T,{R}^{u},{R}_{i}^{c},\mathcal{P}}\right)$ 定义，其中 $\mathcal{S}$ 是状态空间，${\mathcal{A}}_{u}$ 和 ${\mathcal{A}}_{c}$ 分别是领导者和追随者的动作集，$T$ 是每个情节中的步骤数，${r}_{u}$ 和 ${r}_{c,i}$ 是领导者和追随者的奖励函数， 分别，$\mathcal{P} = {\left\{ {P}_{t} : S \times {\mathcal{A}}_{u} \times {\mathcal{A}}_{c} \rightarrow \left\lbrack 0,1\right\rbrack \right\} }_{t = 1}^{T}$ 是过渡内核的集合。这里 ${\mathcal{A}}_{u} \times {\mathcal{A}}_{c} = {\mathcal{A}}_{u} \times {\mathcal{A}}_{c,1} \times \cdots \times {\mathcal{A}}_{c,\left| \mathcal{I}\right| }$ 。基于所提出的框架，POMG 建模如下：

1） 状态空间：状态空间 $\mathcal{S}$ 是 ${s}_{t} =$ $\left\{ {{p}_{t}^{g},{S}_{t},{\mathcal{T}}_{t}^{a},{\mathcal{T}}_{t}^{b},{\mathcal{T}}^{d}\left( {E}_{t}^{g}\right) ,{\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right) ,{\mathcal{T}}^{s},{\mathcal{T}}^{r}}\right\} .$ 的集合

2） 观测空间：UC 的观测集 ${\mathcal{O}}_{u}$ 是 ${o}_{t}^{u} = \left\{ {{p}_{t}^{g},{p}_{i,t}^{u},{p}_{i,t}^{r},\Delta {p}_{i,t},{S}_{t},{\mathcal{T}}_{t}^{g},{\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right) }\right\}$ 的集合。消费者 ${\left\{ {\mathcal{O}}_{i}\right\} }_{i \in \mathcal{I}}$ 的观测集是 ${o}_{i,t}^{c} =$ $\left\{ {{p}_{t}^{b},{p}_{t}^{s},{p}_{t}^{a},{p}_{t}^{c},{p}_{t}^{d},{p}_{t}^{e},{p}_{t}^{o},{p}_{t}^{m},{S}_{t},{\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right) ,{\mathcal{T}}^{r},{\mathcal{T}}^{d}\left( {p}_{t}^{g}\right) }\right\} .$ 的集合

3） 动作空间：UC ${\mathcal{A}}_{u}$ 的动作集是 ${a}_{t}^{u}$ 的集合，其中 ${a}_{t}^{u} = \left\{ {{p}_{t}^{b},{p}_{t}^{s},{p}_{t}^{a},{p}_{t}^{c},{p}_{t}^{d},{p}_{t}^{e},{p}_{t}^{o},{p}_{t}^{m}}\right\}$ 。联合消费者 ${\mathcal{A}}_{c}$ 的动作集是 ${a}_{t}^{c} = \left\{ {\mathop{\sum }\limits_{{i \in \mathcal{I}}}{p}_{i,t}^{r},\mathop{\sum }\limits_{{i \in \mathcal{I}}}{p}_{i,t}^{u},\mathop{\sum }\limits_{{i \in \mathcal{I}}}\Delta {p}_{i,t}^{l},\mathop{\sum }\limits_{{i \in \mathcal{I}}}\Delta {p}_{i,t}^{s}}\right\}$ 的融合。$i$ -th 使用者 ${\left\{ {\mathcal{A}}_{i}\right\} }_{i \in \mathcal{I}}$ 的作集是 ${a}_{i,t}^{c}$ 的集合，其中 ${a}_{i,t} = \left\{ {{p}_{i,t}^{r},{p}_{i,t}^{u},\Delta {p}_{i,t}^{l},\Delta {p}_{i,t}^{s}}\right\}$ 。

4） 奖励函数：UC 代理的奖励函数定义为 ${R}_{t}^{u} = {C}_{t}^{s} + {C}_{t}^{m} + {C}_{t}^{g} + {C}_{t}^{r}$ ，由当前状态 ${s}_{t}$ 计算得出。消费者代理的奖励函数定义为 ${R}_{i,t}^{c} = {C}_{i,t}^{p} + {C}_{i,t}^{d} + {C}_{i,t}^{r}$ 。

5） 价值函数和优势函数：在此设置中，考虑了决策中的两个层次结构层次：一个领导者 （UC） 和 $\left| \mathcal{I}\right|$ 追随者（消费者）。领导者的随机策略 ${\pi }_{u}\left( {{a}_{u,t} \mid {o}_{u,t};{\theta }_{u}}\right)$ 是给定观察值的行动的一组概率分布。同时，追随者的随机联合策略定义为 ${\pi }_{i}\left( {{a}_{i,t}^{c} \mid {o}_{i,t}^{c},{a}_{t}^{u};{\theta }_{c},{\theta }_{i}}\right)$ 。给定策略 $\left( {{\pi }_{u},{\left\{ {\pi }_{i}\right\} }_{i \in \mathcal{I}}}\right)$ ，领导者 （UC） 和追随者 （消费者） 的状态值函数由

$$
{V}_{u}\left( {o}_{t}^{u}\right)  = \mathbb{E}\left\lbrack  {\mathop{\sum }\limits_{{k = 0}}^{{T - t}}{\gamma }^{k}{R}_{t + k}^{u} \mid  {o}_{t}}\right\rbrack   \tag{32}
$$

$$
{V}_{i}\left( {o}_{i,t}^{c}\right)  = \mathbb{E}\left\lbrack  {\mathop{\sum }\limits_{{k = 0}}^{{T - t}}{\gamma }^{k}{R}_{i,t + k}^{c} \mid  {o}_{i,t},{a}_{t}^{u}}\right\rbrack   \tag{33}
$$

领导者 （UC） 和跟随者 （消费者） 的优势函数由下式定义

$$
{A}_{u}\left( {{s}_{t},{a}_{t}^{u}}\right)  = \mathop{\sum }\limits_{{k = 0}}^{{T - t}}{\gamma }^{k}{R}_{t + k}^{u} + \gamma {V}_{u}\left( {o}_{t + 1}^{u}\right)  - {V}_{u}\left( {o}_{t}^{u}\right)
$$

$$
{A}_{c}\left( {{s}_{t},\mathop{\sum }\limits_{i}{a}_{i,t}^{c}}\right)  = \mathop{\sum }\limits_{{k = 0}}^{{T - t}}\mathop{\sum }\limits_{i}^{\left| \mathcal{I}\right| }{\gamma }^{k}{R}_{i,t + k}^{c} + \gamma {V}_{c}\left( {o}_{t + 1}^{c}\right)  - {V}_{c}\left( {o}_{t}^{c}\right)  \tag{34}
$$

$$
{A}_{c,i}\left( {{s}_{t},{a}_{i,t}^{c}}\right)  = \mathop{\sum }\limits_{{k = 0}}^{{T - t}}{\gamma }^{k}{R}_{i,t + k}^{c} + \gamma {V}_{i}\left( {o}_{i,t + 1}^{c}\right)  - {V}_{i}\left( {o}_{i,t}^{c}\right)
$$

优势函数的目的是评估在给定状态下选择特定行动相对于平均策略行为的相对优势。在这个框架中，advantage 函数直观地表示一个动作在给定状态下的表现是好于还是差于平均值。

6） Stackelberg-Nash 均衡：对于 UC 政策 ${\pi }_{u}$ ，消费者的纳什均衡是一个联合政策 ${\pi }_{c} =$ ${\left\{ {\pi }_{i}^{ * }\right\} }_{i \in \mathcal{I}}$ 。对于每个 UC 策略 ${\pi }_{u}$ ，消费者的最佳响应策略定义为 arg max，其中消费者最佳响应策略 ${\pi }_{c}$ 是给定 UC 策略 ${\pi }_{u}$ 的消费者的纳什均衡。

$$
\mathrm{{NE}} = \left\{  {{\left\{  {\pi }_{i}^{ * }\right\}  }_{i \in  \mathcal{I}} \mid  {V}_{i}^{ * }\left( {o}_{i,t}^{c}\right)  \geq  {V}_{i}\left( {o}_{i,t}^{c}\right) }\right\}   \tag{35}
$$

UC 的 SNE 是“对最佳响应的最佳响应”。换句话说，在消费者总是采用 ${\pi }_{i}^{ * }$ 的情况下，UC 政策 ${\pi }_{u}$ 使价值函数最大化，即

$$
\mathrm{{SNE}} = \left\{  {{\pi }_{u}^{ * },{\left\{  {\pi }_{i}^{ * }\right\}  }_{i \in  \mathcal{I}} \mid  {V}_{u}^{ * }\left( {o}_{t}^{u}\right)  \geq  {V}_{u}\left( {o}_{t}^{u}\right) ,{V}_{i}^{ * }\left( {o}_{i,t}^{c}\right)  \geq  {V}_{i}\left( {o}_{i,t}^{c}\right) }\right\}
$$

(36)

一般和博弈的 SNE 是一个策略对 $\left( {{\pi }_{u}^{ * },{\left\{ {\pi }_{i}^{ * }\right\} }_{i \in \mathcal{I}}}\right)$ 。我们的目标是找到 SNE，即在假设消费者总是做出最佳反应的情况下，UC 的最佳策略。等效地，我们需要通过更新策略 ${\pi }_{i},{\pi }_{u}$ 来最大化 UC 和消费者的状态价值函数 ${V}_{i},{V}_{u}$ 。

## B. Stackelberg 多智能体强化学习

在本节中，应用了一种改进的多代理方法，其中参与者网络的输入是所有消费者观察 ${o}_{t}^{c} = \mathop{\sum }\limits_{i}^{\left| \mathcal{I}\right| }{o}_{i,t}^{c}$ 的总和，批评者网络的输出是中间值 ${a}_{t}^{c}$ 。该算法基于集中式训练和分散执行的范式进行训练 \[37]。本节的主要挑战是 UC 如何估计消费者策略的最佳响应策略。受 \[38] 和 \[39] 的启发，Stackelberg 强化学习算法中政策网络的学习动力学源自基于一阶梯度的充分条件，由 （37） 和 （38） 给出：

$$
{\theta }_{u} \leftarrow  {\theta }_{u}^{\prime } + {\alpha }_{u}\nabla {\mathcal{L}}_{u}\left( {{\theta }_{u},{\theta }_{c}}\right)  \tag{37}
$$

$$
{\theta }_{c} \leftarrow  {\theta }_{c}^{\prime } - {\alpha }_{c}{\nabla }_{{\theta }_{c}}{\mathcal{L}}_{c}\left( {{\theta }_{u},{\theta }_{c}}\right)  \tag{38}
$$

其中 ${\alpha }_{u}$ 和 ${\alpha }_{c}$ 表示参数更新的学习率。$\nabla {\mathcal{L}}_{u}$ 表示 UC 策略损失函数的全导数。${\nabla }_{{\theta }_{c}}{\mathcal{L}}_{c}$ 表示消费者保单损失函数的偏导数。$\nabla {\mathcal{L}}_{u}\left( {{\theta }_{u},{\theta }_{c}}\right)$ 根据 \[40] 定义为 （39）

$$
\nabla {\mathcal{L}}_{u}\left( {{\theta }_{u},{\theta }_{c}}\right)  = {\nabla }_{{\theta }_{u}}{\mathcal{L}}_{u} - {\nabla }_{{\theta }_{u}{\theta }_{c}}{\mathcal{L}}_{u}{\left( {\nabla }_{{\theta }_{c}}^{2}{\mathcal{L}}_{c}\right) }^{-1}{\nabla }_{{\theta }_{c}}{\mathcal{L}}_{u} \tag{39}
$$

UC 策略损失函数的全导数表示基于消费者最佳响应的 UC 优化策略。基于 Stackelberg 多智能体算法，将策略网络的损失函数定义为：

$$
{\mathcal{L}}_{u}^{p}\left( {{\theta }_{u},{\theta }_{c}}\right)  = \mathbb{E}\left\lbrack  {\min \left( {{r}_{u,t}{A}_{u},\operatorname{clip}\left( {{r}_{u,t},1 \pm  \epsilon }\right) {A}_{u}}\right) }\right\rbrack   \tag{40}
$$

$$
{\mathcal{L}}_{c}^{p}\left( {{\theta }_{u},{\theta }_{c}}\right)  = \mathbb{E}\left\lbrack  {\min \left( {{r}_{c,t}{A}_{c},\operatorname{clip}\left( {{r}_{c,t},1 \pm  \epsilon }\right) {A}_{c}}\right) }\right\rbrack
$$

其中 $\mathbb{E}$ 表示预期值。$\epsilon$ 表示剪辑间隔。clip 是一个截断函数，当重要性采样超过指定的上限或下限时，它会返回相应的上限或下限。${r}_{u,t}$ 和 ${r}_{c,t}$ 表示重要性抽样，其中：

$$
{r}_{u,t} = \frac{{\pi }_{u}\left( {{a}_{t}^{u} \mid  {o}_{t}^{u};{\theta }_{u}}\right) }{{\pi }_{u}\left( {{a}_{t}^{u} \mid  {o}_{t}^{u};{\theta }_{u}^{\prime }}\right) },{r}_{c,t} = \frac{{\pi }_{c}\left( {{a}_{t}^{c} \mid  {o}_{t}^{c};{\theta }_{c}}\right) }{{\pi }_{c}\left( {{a}_{t}^{c} \mid  {o}_{t}^{c};{\theta }_{c}^{\prime }}\right) } \tag{41}
$$

为了鼓励探索，使用以下公式对损失函数应用熵正则化项：

$$
{\mathcal{L}}_{u}\left( {{\theta }_{u},{\theta }_{c}}\right)  = {\mathcal{L}}_{u}^{p}\left( {{\theta }_{u},{\theta }_{c}}\right)  + {K}_{u}{D}_{KL}^{\max }\left( {{\pi }_{u},{\pi }_{u}^{\prime }}\right)  \tag{42}
$$

$$
{\mathcal{L}}_{c}\left( {{\theta }_{u},{\theta }_{c}}\right)  = {\mathcal{L}}_{c}^{p}\left( {{\theta }_{u},{\theta }_{c}}\right)  + {K}_{c}{D}_{KL}^{\max }\left( {{\pi }_{c},{\pi }_{c}^{\prime }}\right)  \tag{43}
$$

其中 ${D}_{KL}^{\max }\left( {{\pi }_{c},{\pi }_{c}^{\prime }}\right)$ 和 ${D}_{KL}^{\max }\left( {{\pi }_{u},{\pi }_{u}^{\prime }}\right)$ 表示新政策和旧政策的 KL 背离。UC 和消费者批评家网络的损失函数定义为

$$
\begin{aligned} {\mathcal{L}}_{u}^{k}\left( {\phi }_{u}\right) &  = \mathbb{E}\left\lbrack  {\min \left\lbrack  {{\left( {V}_{u} - {R}_{t}^{u}\right) }^{2},{\left( \operatorname{clip}\left( {V}_{u},{V}_{u}^{\prime } \pm  \epsilon \right)  - {R}_{t}^{u}\right) }^{2}}\right\rbrack  }\right\rbrack  \\  {\mathcal{L}}_{c}^{k}\left( {\phi }_{c}\right) &  = \mathbb{E}\left\lbrack  {\min \left\lbrack  {{\left( {V}_{c} - {R}_{t}^{c}\right) }^{2},{\left( \operatorname{clip}\left( {V}_{c},{V}_{c}^{\prime } \pm  \epsilon \right)  - {R}_{t}^{c}\right) }^{2}}\right\rbrack  }\right\rbrack   \end{aligned} \tag{44}
$$

其中 ${V}_{c}$ 和 ${V}_{u}$ 表示 UC critic network 和 consumer critic network 的输出。${V}_{c}^{\prime }$ 和 ${V}_{u}^{\prime }$ 表示 old UC critic network 和 old consumer critic network 的输出。批评家网络的学习动态定义为

$$
{\phi }_{u} \leftarrow  {\phi }_{u}^{\prime } - {\beta }_{u}{\nabla }_{{\phi }_{u}}{\mathcal{L}}_{u}^{k}\left( {\phi }_{u}\right)  \tag{45}
$$

$$
{\phi }_{c} \leftarrow  {\phi }_{c}^{\prime } - {\beta }_{c}{\nabla }_{{\phi }_{c}}{\mathcal{L}}_{c}^{k}\left( {\phi }_{c}\right)  \tag{46}
$$

其中 ${\beta }_{u}$ 和 ${\beta }_{c}$ 表示 Critic 网络参数更新的学习率。MAPPO 通常使用以下公式将熵正则化项添加到损失函数中

在第 III-B 节中，UC 代理通过总导数 $\nabla {\mathcal{L}}_{u}\left( {{\theta }_{u},{\theta }_{c}}\right)$ 更新策略参数 ${\theta }_{u}$ ，而消费者代理 ${\theta }_{c}$ 通过偏导数 ${\nabla }_{{\theta }_{c}}{\mathcal{L}}_{c}\left( {{\theta }_{u},{\theta }_{c}}\right)$ 更新策略参数。评论家网络通过 （44） 进行了更新。UC agent 在 Stackelberg 游戏环境中的作将影响 Nash 游戏环境。将保存 consumer agent 的更新参数 ${\theta }_{c}$ 以供下一步使用，其中所有具有策略 ${\pi }_{i}\left( {{a}_{i,t}^{c} \mid {o}_{i,t}^{c},{a}_{t}^{u};{\theta }_{c},{\theta }_{i}}\right)$ 的 consumer agent 在 Nash 游戏环境中相互竞争。

## C. Nash 并行多智能体强化学习

由于消费者是独立和自私的个体，因此消费者倾向于最大化效用 ${\mathcal{J}}_{c,i}$ 。应用 MAPPO 算法解决消费者之间的纳什均衡博弈策略。policy network 和 critic network 的损失函数表示为：

$$
{\mathcal{L}}_{i}^{p}\left( {\theta }_{i}\right)  = \mathbb{E}\left\lbrack  {\min \left\lbrack  {{r}_{i,t}{A}_{c,i},\operatorname{clip}\left( {{r}_{i,t},1 \pm  \epsilon }\right) {A}_{c,i}}\right\rbrack  }\right\rbrack   \tag{47}
$$

$$
{\mathcal{L}}_{i}^{k}\left( {\phi }_{i}\right)  = \mathbb{E}\left\lbrack  {\max \left\lbrack  {\left( {V}_{i}\left( {o}_{i,t}^{c};{\phi }_{i}\right)  - {R}_{i,t}^{c}\right) }^{2}\right. }\right.
$$

$$
\left. {\left( \operatorname{clip}\left( {V}_{i}\left( {o}_{i,t}^{c};{\phi }_{i}\right) ,{V}_{i}\left( {o}_{i,t}^{c};{\phi }_{i}^{\prime }\right)  \pm  \epsilon \right)  - {R}_{i,t}^{c}\right) }^{2}\right\rbrack
$$

(48)

参数更新如下：

$$
{\theta }_{i} \leftarrow  {\theta }_{i}^{\prime } - \alpha {\nabla }_{{\theta }_{i}}{\mathcal{L}}_{i}^{p}\left( {\theta }_{i}\right)  \tag{49}
$$

$$
{\phi }_{i} \leftarrow  {\phi }_{i}^{\prime } - \beta {\nabla }_{{\phi }_{i}}{\mathcal{L}}_{i}^{k}\left( {\phi }_{i}\right)  \tag{50}
$$

其中 ${\theta }_{i}^{\prime }$ 表示旧策略网络解码器参数。${\phi }_{i}^{\prime }$ 表示旧的 Critic 网络参数。消费者策略网络解码器和批评者网络基于 MAPPO 进行训练。在执行阶段，消费者的策略网络 ${\pi }_{i}\left( {{a}_{i,t}^{c} \mid {o}_{i,t}^{c},{a}_{t}^{u};{\theta }_{c},{\theta }_{i}}\right)$ 被表述为编码器和解码器的连接。

## D. 建议的 SN-MAPPO 方法

图 3 说明了所提出的 SN-MAPPO 方法的集中式训练和分散执行架构，而算法 1 总结了其训练阶段。

为了更好地解释 SN-MAPPO 算法的训练过程，以下小节是对算法 1 的补充：

1） 神经网络：消费者的策略网络架构如图所示。4. 在提出的异步 MARL 中，策略 ${\pi }_{i}$ 的编码器的参数 ${\theta }_{c}$ 是一个共享参数，由所有消费者代理共享，而解码器参数 ${\theta }_{i}$ 是特征参数，在 Nash 游戏环境下由消费者更新。在所提出的方法中，包括两个环境：Stackelberg 游戏环境和 Nash 游戏环境。在 Stackel-berg 博弈环境中，UC 策略网络 ${\pi }_{u}$ 和消费者策略网络 ${\pi }_{c}$ 的编码器基于 （37） 和 （38） 进行更新。在 Nash 游戏环境中，消费者通过共享编码器 ${\theta }_{c}$ 的参数来基于 MAPPO 算法更新解码器参数。由于 UC 的行动会影响 Nash 游戏的结果，因此 UC 的策略和消费者的策略以异步形式更新，最终的策略收敛到 SNE。

算法 1：SN-MAPPO

***

数据：关税策略、DER 数据、加载数据、环境、超参数 结果：UC 策略 ${\pi }_{u}\left( {{o}_{t}^{u};{\theta }_{u}}\right)$ ，消费者策略 ${\pi }_{i}\left( {{o}_{t}^{i};{\theta }_{c},{\theta }_{i}}\right)$ Stackelberg 游戏环境和策略初始化;当到达 SNE 时，执行时间间隔 $t = 1 : T$ do Sample action ${a}_{t}^{u} \sim {\pi }_{u}$ 和 ${a}_{t}^{c} \sim {\pi }_{c}$ ;在 Stackelberg 游戏环境中执行动作 ${a}_{t}^{u}$ 和 ${a}_{t}^{c}$ 并检索轨迹 $\left( {{o}_{t}^{c},{a}_{t}^{c},{R}_{t}^{c},{o}_{t}^{u},{a}_{t}^{u},{R}_{t}^{u}}\right)$ 以存储在重播缓冲区 ${\mathcal{D}}_{S}$ 中;计算 ${\mathcal{D}}_{S}$ 中每个体验的优势估计值 ${A}_{i,t}$ 和损失函数 ${\mathcal{L}}_{i}^{p}$ 和 ${\mathcal{L}}_{i}^{k}$ ;通过将损失 ${\mathcal{L}}_{u}^{k}$ 和 ${\mathcal{L}}_{c}^{k}$ 最小化为 （37） 来更新 UC 和消费者的批评网络;使用最小化 ${\mathcal{L}}_{u}^{p}$ 计算的总梯度更新 UC 策略网络;使用最小化 ${\mathcal{L}}_{c}^{p}$ 计算的策略梯度更新消费者策略网络编码器，如 （38） ;更新 Nash 游戏环境 ;对于代理 $i = 1 : \left| \mathcal{I}\right|$ do Synchronize 代理策略共享参数 ${\theta }_{c}$ ;示例作 ${a}_{i,t}^{c} \sim {\pi }_{i}$ ;执行作并检索观测值 ${o}_{i,t}^{c}$ 以存储在 ${\mathcal{D}}_{N,i}$ 中;计算 ${\mathcal{D}}_{N,i}$ 中每个体验的优势估计值 ${A}_{i,t}$ 和损失函数 ${\mathcal{L}}_{i}^{p}$ 和 ${\mathcal{L}}_{i}^{k}$ ;通过将 ${\mathcal{L}}_{i}^{p}$ 和 ${\mathcal{L}}_{i}^{k}$ 最小化为 （49） 和 （50） 来更新批评者网络和特征策略网络;

***

![\<img src="attachments/8X9YZK73.jpg" alt="" width="901" height="500" data-attachment-key="8X9YZK73" ztype="zimage"> | 901](attachments/8X9YZK73.jpg)

图 3.拟议的 SN-MAPPO 的框架和更新程序。

![\<img src="attachments/XHHRB3PG.jpg" alt="" width="924" height="500" data-attachment-key="XHHRB3PG" ztype="zimage"> | 924](attachments/XHHRB3PG.jpg)

图 4.消费者策略网络 ${\pi }_{i}$ 的编码器和解码器架构。

UC 策略网络由 5 层组成：3 个并行 MLP、1 个嵌入层、2 个门循环单元 （GRU） 层和 2 个密集层。UC 策略网络的输入和输出为 ${\mathcal{O}}_{u}$ 和 ${\mathcal{A}}_{u}$ 。使用者策略网络由编码器和解码器组成。消费者策略网络的编码器由 2 个并行的 MLP、1 个嵌入层、1 个 GRU 层、2 个密集层组成。消费者策略网络的解码器由 3 个并行 MLP、1 个嵌入层、1 个 GRU 层、2 个密集层组成。使用者策略网络的编码器输入观察状态 $\mathop{\sum }\limits_{i}{\mathcal{O}}_{i}$ 的总和，并输出使用者作 $\mathop{\sum }\limits_{i}{\mathcal{A}}_{i}$ 的总和 。消费者策略网络的解码器输入观察状态 ${\mathcal{O}}_{i}$ 和 $\mathop{\sum }\limits_{i}{\mathcal{A}}_{i}$，并输出消费者作 ${\mathcal{A}}_{i}$ 的总和。UC 和消费者评论家网络由 5 层组成：1 个安全掩码层、2 个 GRU 层、2 个密集层。UC 评论家网络的输入是时间步 $t$ 的 Stackelberg 游戏环境状态 $\mathcal{S}$ 的状态。UC 和消费者评论家网络根据损失函数 （44） 进行更新。

在 SN-MAPPO 中，消费者策略网络的编码器由所有消费者共享。consumer 策略网络和 UC 策略网络的编码器基于 （40） 进行更新。消费者策略网络的解码器基于 （48） 进行更新。消费者策略网络的编码器旨在获得 UC 和消费者之间的 Stackelberg 均衡，而策略网络的解码器旨在获得消费者之间的纳什均衡。

MLP 旨在捕捉功率和关税之间的范围差异。MLP 的输出向量被拼接，然后输入到嵌入层中，以对 UC 和消费者策略网络中的观察结果进行矢量化。

SN-MAPPO 在策略网络的每个激活层之前使用批量归一化层来提高收敛性。

2） 约束：为了过滤掉无效的动作，应用了安全掩码动作 \[41]。要使不同代理的策略网络具有相同维度的输入。

$$
{A}_{u}\left( {{s}_{t},{a}_{t}^{u}}\right)  = {\mathrm{M}}_{d}\left\lbrack  {{A}_{u}\left( {{s}_{t},{a}_{t}^{u}}\right) }\right\rbrack   \tag{51}
$$

$$
{A}_{c,i}\left( {{s}_{t},{a}_{t}^{u}}\right)  = {\mathrm{M}}_{d}\left\lbrack  {{A}_{c,i}\left( {{s}_{t},{a}_{t}^{u}}\right) }\right\rbrack   \tag{52}
$$

其中 SM 运算符 ${\mathrm{M}}_{d}$ 表示将违反约束的优势函数替换为 $- \infty$ 。安全掩码层可以在训练期间中断无效作的更新过程。在训练过程中，安全掩码作可以防止消费者行为超出可行范围。

此外，还可以解决不同消费者具有不同作空间的问题。由于使用者可能具有不同的作空间，为了确保策略网络在参数共享期间的可扩展性属性，将应用 action-mask，以便在训练期间不会更新使用者不包含的作空间。在损失函数中，action 被计算为 ${\mathcal{A}}_{c,i} \cdot {M}_{i}$ 来阻止使用者不包含的策略更新。${M}_{i}$ 表示消费者行动空间的指标。

3） 数据：在所提出的框架中，消费者的原始负载功率满足以下关系：

$$
{p}_{i,t}^{ * } = {p}_{i,t}^{u} + {p}_{i,t}^{r} + \Delta {p}_{i,t} \tag{53}
$$

其中 ${p}_{i,t}^{ * }$ 表示来自 Load 数据集的源 Load 数据。消费者的需求响应能力满足以下关系：

$$
- \Delta {p}_{i,t}^{ * } \leq  \Delta {p}_{i,t}^{l} + \Delta {p}_{i,t}^{s} \leq  \Delta {p}_{i,t}^{ * } \tag{54}
$$

其中 $\Delta {p}_{i,t}^{ * }$ 是第 $i$ 个使用者的最大需求响应能力。负载数据 ${p}_{i,t}^{ * }$ 、最大 DR 功率 $\Delta {p}_{i,t}^{ * }$ 、DER 发电量 ${p}_{t}^{g}$ 和实时市场购电电价 ${\mathcal{T}}_{t}^{b}$ 作为环境状态应用。SN-MAPPO 每月使用负载数据进行训练，因为电费每月都会清算。

4） 优先体验重放：此外，为了进一步加速所提出的方法的学习性能，我们用优先体验重放（PER）方法扩展了它的功能\[42]。PER 根据优势修正状态值估计。对特定政策轨迹进行采样的概率表示：

$$
{\sigma }_{u} = \frac{\sigma {\left( {\tau }_{u}\right) }^{{\kappa }_{u}}}{\mathop{\sum }\limits_{j}\sigma {\left( {\tau }_{u}\right) }^{{\kappa }_{u}}} \tag{55}
$$

$$
{\sigma }_{i} = \frac{\sigma {\left( {\tau }_{i}\right) }^{{\kappa }_{i}}}{\mathop{\sum }\limits_{j}\sigma {\left( {\tau }_{i}\right) }^{{\kappa }_{i}}} \tag{56}
$$

其中 ${\mathcal{D}}_{S}$ 和 ${\mathcal{D}}_{N,i}$ 表示 UC 策略轨迹 ${\tau }_{u} = \left( {{o}_{t}^{c},{a}_{t}^{c},{R}_{t}^{c},{o}_{t}^{u},{a}_{t}^{u},{R}_{t}^{u}}\right)$ 的重放缓冲区，消费者策略轨迹 ${\tau }_{i} = \left( {{o}_{t}^{i},{a}_{t}^{i},{R}_{t}^{i},{o}_{t + 1}^{i}}\right) .{\kappa }_{u}$ 和 ${\kappa }_{i}$ 表示优先级，$\sigma \left( {\tau }_{u}\right) = 1/\operatorname{rank}\left( {\tau }_{u}\right)$ 和 $\sigma \left( {\tau }_{i}\right) = 1/\operatorname{rank}\left( {\tau }_{u}\right)$ 是与策略轨迹 ${\tau }_{u}$ 和 ${\tau }_{i}$ 相关的优先级。rank 表示根据 action 的优势对 ${\mathcal{D}}_{S}$ 和 ${\mathcal{D}}_{N,i}$ 进行排序时的轨迹排名。

5） 收敛条件：SN-MAPPO 算法维护一个策略池，用于在更新之前存储原始策略。SN-MAPPO 算法分别计算原始策略和更新策略的 KL 背离，并确定是否达到均衡。当所有更新的策略都满足 （57） 和 （58） 与策略池中的旧策略相比 50 集时，将达到收敛条件。

$$
{D}_{KL}^{\max }\left( {{\pi }_{i},{\pi }_{i}^{\prime }}\right)  < {\varepsilon }_{i},i \in  \mathcal{I} \tag{57}
$$

$$
{D}_{KL}^{\max }\left( {{\pi }_{u},{\pi }_{u}^{\prime }}\right)  < {\varepsilon }_{u},{D}_{KL}^{\max }\left( {{\pi }_{c},{\pi }_{c}^{\prime }}\right)  < {\varepsilon }_{c} \tag{58}
$$

6） 环境：根据 POMG 的说法，构建了 2 个多智能体游戏环境：Stackelberg 游戏环境和 Nash 游戏环境。在 Nash 游戏环境中，消费者在 UC 的行动条件下保持不合作关系，这种关系是动态的，受 UC 行动的影响。在 Stackelberg 游戏环境中，UC 会更新其策略以获取针对使用者的最佳响应策略。环境的具体状态转换过程显示在 第三节 中描述的框架中。

## E. SN-MAPPO 收敛分析

为了证明 SN-MAPPO 的收敛性，考虑了以下步骤：首先，消费者的政策通过 MAPPO 收敛到 Nash 均衡;收敛到纳什均衡可以通过证明政策更新导致一个固定点来证明，在这个点上，没有代理可以单方面提高其预期回报。从形式上讲，如果对于所有代理来说，联合策略 ${\pi }_{i}^{ * }\left( {i = 1,\ldots ,\left| \mathcal{L}\right| }\right)$ 都满足方程 （41），则联合策略 **0** 是纳什均衡。我们通过以下定理来证明这一点，该定理指出 MAPPO 具有单调改进特性。MAPPO 到 Nash 均衡的收敛证明见 \[43]。以下定理描述了 MAPPO 对 Nash 均衡的渐近收敛行为。

定理 1.假设在 MAPPO 中，任何代理体排列都具有开始更新的固定非零概率。然后，在马尔可夫博弈中，由算法生成的联合策略序列 ${\left( {\pi }_{i}\right) }_{k = 0}^{\infty }$ 具有一组非空的限制点，每个限制点都是一个纳什均衡。

MAPPO 更新的联合策略序列 ${\left( {\pi }_{i}\right) }_{k = 0}^{\infty }$ 对所有 $k$ 具有单调改进属性。这种单调的改进特性是通过多代理优势和顺序更新方案实现的，保证了回报的收敛。此外，更新顺序的随机化可确保在收敛时，没有任何代理程序被激励进行进一步的更新。通过排除算法在非平衡点收敛的可能性，最终确定证明。

其次，UC 策略和消费者联合策略通过 SN-MAPPO 收敛到 Stackelberg 均衡。\[44] 考虑了 Stackelberg 游戏中不完整信息的无悔动态，它收敛到贝叶斯粗略相关均衡。对于 SN-MAPPO，考虑博弈 $\left( {{\theta }_{u}^{ * },{\omega }_{c}^{ * }}\right)$ 的 Stackelberg 均衡，该均衡对于连续时间动态系统在局部渐近稳定，其中 UC 在 Stackelberg 梯度中的总导数由方程 （44） 给出，消费者策略的单个梯度为 ${\nabla }_{{\theta }_{c}}{\mathcal{L}}_{c}\left( {{\theta }_{u},{\theta }_{c}}\right)$ \[40]。

演员和评论家采用算法 1 中给出的离散时间更新，其中 UC 是领导者，消费者是追随者。由于 UC 策略和消费者策略对其梯度进行了无偏估计，因此在对噪声过程和步长序列的假设下，我们将算法 1 中的更新视为随机近似过程 $\left( {{\theta }_{c,k},{\theta }_{u,k}}\right)$ ，其中 $k$ 表示更新生成。然后，我们将渐近轨迹定义为迭代 $\left( {{\theta }_{c,k},{\theta }_{u,k}}\right)$ 和 $\left( {{\theta }_{c,k + 1},{\theta }_{u,k + 1}}\right)$ 之间的线性插值。鉴于 $\left( {{\theta }_{c}^{ * },{\theta }_{u}^{ * }}\right)$ 在局部渐近稳定，因此存在 $\left( {{\theta }_{c}^{ * },{\theta }_{u}^{ * }}\right)$ 的邻域，并且该邻域中有一个局部 Lyapunov 函数。这个 Lyapunov 函数可用于表明，对于从 $\left( {{\theta }_{c},{\theta }_{u}}\right)$ 开始的任何迭代序列，从迭代 $\left( {{\theta }_{c},{\theta }_{u}}\right)$ 开始的连续时间流和渐近伪轨迹彼此渐近地相互收缩。因此，迭代 $\left( {{\theta }_{c},{\theta }_{u}}\right)$ 反过来逐渐收敛到 $\left( {{\theta }_{c}^{ * },{\theta }_{u}^{ * }}\right)$ \[40]。

## F. 可扩展性和适应性

SN-MAPPO 算法的可扩展性和适应性表现在几个方面：

首先，通过在消费者策略网络中使用参数共享，算法的可扩展性得到了显著提高。由于所有消费者策略的 Encoder 部分是共享的，因此只需在每个时间步更新一次，而消费者策略网络的 Decoder 部分则保留了消费者的用电偏好。

其次，作掩码的应用允许具有不同作空间的使用者使用相同的输入和输出维度与策略网络交互。安全掩码还可以帮助策略网络遵守作约束。

第三，随着消费者数量的进一步增长，即 $\left| \mathcal{I}\right| \rightarrow \infty$ ，该算法的可扩展性被考虑在内。考虑多个代理类似地面临的维度灾难问题，因为算法随着代理的数量趋于无穷大。我们认为，更合理的解决方案是均值场博弈论 MARL 来解决 SN-MAPPO 的维度灾难问题，这是我们当前研究的主要关注点之一。在拟议的 DR 框架中，均值场 MARL 仅关注特定的消费者，而其他消费者则根据平均场博弈论统一为一个代理。在这种情况下，基于均值场的 SN-MAPPO 算法将转化为包含 UC 代理、$i$ -th 消费者代理和其他消费者 $- i$ 代理的三代理 MARL 算法，这可能会解决 SN-MAPPO 在 $\left| \mathcal{I}\right| \rightarrow \infty$ 时的维度爆炸问题。

## 四.案例研究

在本节中，我们展示了仿真结果并评估了所提出的框架和算法的性能。

## A. 案例描述

1） 数据来源：为了验证我们在实际场景中的贡献，我们使用了中国国家电网湖北省电力有限公司提供的日期集，其中包括 2021 年 3 栋写字楼、1 家购物中心、1 所大学和 3 个住宅区的负荷数据。每日负载数据包括 96 个采样点，总计一年的负载数据。应用 2022 年 2 个月的负载数据来验证拟议框架的有效性。此外，UC 的 DER 生成如图 5b 所示。电力市场电价 ${\mathcal{T}}_{t}^{b}$ 和相应的不确定性 $\varepsilon$ 如图 5c 所示。

2） 参数设置：根据中国的 TUTT 政策和 DR 政策 \[45]，划定了高峰期、高点和平坦期：高峰期（20：00-22：00）、高负荷期（09：00-15：00）和持平期（其他时间）。${E}_{i,t}^{u}$ 分为三个关税阶段：每月用电量 ${E}_{i,T}^{u}$ 小于 ${2.88} \times {10}^{7}\mathrm{{kWh}}$ ，每月用电量 ${E}_{i,T}^{u}$ 从 ${2.88} \times {10}^{7}\mathrm{{kWh}}$ 到 ${4.8} \times {10}^{7}\mathrm{{kWh}}$ ，每月用电量 ${E}_{i,T}^{u}$ 大于 ${4.8} \times {10}^{7}\mathrm{{kWh}}$ 。DR 灵活性购买价格 ${\mathcal{T}}^{b}$ 是 $ {0.05}\$ /\mathrm{{kWh}} $. DR benefit tariff$ {\mathcal{T}}^{s} $is$ {0.01}\<span class="math">$/\mathrm{{kW}}$ 。电费 舒适关税 ${\mathcal{T}}_{i}^{c}$ 是 $ {0.01}\$ /\mathrm{{kW}} $for all consumers. Furthermore,time interval for optimization$ T \$ 设置为一个月，其中时间分辨率为 15 分钟。武汉市的分层使用时间如表 II 所示。

表 II

UC 资费优化结果

| <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ----------------------------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------- |
| Period Tiered Pricing $\left( {\times {10}^{7}\mathrm{\;{kWh}}}\right)$ | High Consumption 0,2.88) (\\$/kWh)$ | Media Consumption \left\lbrack {{2.88},{4.8}}\right) $(\$/kWh) | Low Consumption $\left\lbrack {{4.8},\infty }\right\rbrack$ (\\$/kWh) |
| Peak                                                                    | 0.078                               | 0.067                                                          | 0.060                                                                 |
| High                                                                    | 0.068                               | 0.059                                                          | 0.053                                                                 |
| Flat                                                                    | 0.048                               | 0.041                                                          | 0.037                                                                 |


所提出的框架和方法的部分超参数设置为表 III。

表 III

超参数设置

| <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ----------------------- | ----------- | ------------------------------- | ------- |
| Hyperparameter in Model |             | Hyperparameter in Algorithm     |         |
| Parameters              | Values      | Parameters                      | Values  |
| $\alpha$                | 0.01(decay) | ${\eta }_{c}/{\eta }_{o}$       | 0.8/0.9 |
| $\tau$                  | 0.05        | ${\eta }_{SOC}$                 | 0.95    |
| $\epsilon$              | 0.15        | ${\lambda }_{p}/{\lambda }_{s}$ | 0.2/0.8 |


3） 案例设置：设置了 4 个案例，并用于验证所提方法的有效性：

（I） UC 和消费者在一个完全合作的同步博弈理论框架中参与 DR \[46];

（II） UC 和消费者在一个完全合作的异步博弈理论框架中参与 DR \[47];

（III） UC 和消费者在非合作同步博弈论框架中参与 DR;

（IV） UC 和消费者在非合作异步博弈论框架（拟议的 DR 框架）中参与 DR。

同步和异步情况比较旨在证明所提出的框架对两者的好处

UC 和消费者。为了与所提出的基于异步Stackelberg-Nash的框架相比，基于Cournot game \[48]的同步框架被制定出来，其中消费者和UC同步采取行动。至于完全合作和非合作的案例研究，这项工作旨在证明在完全合作的框架中，一些消费者的利益会受到影响。在完全合作的框架中，消费者强制满足 UC 设定的需求响应激励目标 ${p}_{t}^{m}$。此外，UC 根据需求响应利润最大化的目标分配 DER 能源。

4） 算法比较：设置四种最先进的基于 MARL 的 SNE 算法与建议的算法进行比较：

（I） 斯塔克尔伯格软演员评论家 \[40]

（II） 乐观值迭代 \[49]

（III） Stackelberg 决策转换器 \[50]

（IV） Stackelberg 多智能体深度确定性策略梯度 \[51]

算法 I 表示单代理 SNE 算法，因为消费者之间没有协调。在算法 I 的设置下，消费者被视为一个代理。算法 II 表示 SNE 的在线 MARL，而算法 III 表示 SNE 的离线 MARL。算法 IV 表示 SNE 的策略外 MARL。应该说明的是，上述所有算法都显示出收敛到 SNE 和开源。

5） 训练设置：SN-MAPPO 的训练是通过 Ray RLlib \[52] 和 PyTorch \[53] 在 Python 中进行的。灵敏度分析和超参数通过 Ray Tune 进行 \[54]。游戏环境是通过 PettingZoo \[55] 执行的。训练是在一台具有双 64 核 ${2.4}\mathrm{{GHz}}\mathrm{{AMD}}\left( \mathrm{R}\right) \mathrm{{EPYC}}$ 7763CPU、512GB RAM 和 2 个 NVIDIA A6000 GPU 的计算机上进行的。

## B. UC 需求响应的结果分析

1） 能源消耗结果：图 5a 说明了在不同情况下 2022-6-1 从 PG ${p}_{t}^{b}$ 购买的 UC 电力金额。图 5b 说明了 2022-6-1 实验群落的 DER 生成。图 5c 显示了 2022-6-1 的 UC 实时市场费率。图 5 的目的是以可视化方式展示拟议的 DR 框架中限峰和负载转移的结果。选择了 2022 年 6 月 1 日的 UC 购电金额结果。

如图 5a 所示，所提出的框架可以有效降低峰值负载。一些被削减的峰值负荷转移到低谷期，而其他一些则由消费者主动削减。桌子。IV 通过 3 个指标说明了不同情况下从 PG ${p}_{t}^{b}$ 购买 UC 电力的负荷指数：

*   PAR （峰均比）：

$$
\mathrm{{PAR}} = \frac{T \cdot  \max {p}_{t}^{b}}{\mathop{\sum }\limits_{t}^{T}{p}_{t}^{b}} \tag{59}
$$

*   斜坡（数据斜率之和，kW）：

$$
\operatorname{Ramp} = {\int }_{0}^{T}\left( {{p}_{t}^{b} - {p}_{t - 1}^{b}}\right) \mathrm{d}t \tag{60}
$$

![\<img src="attachments/TGSFDBMC.jpg" alt="" width="669" height="500" data-attachment-key="TGSFDBMC" ztype="zimage"> | 669](attachments/TGSFDBMC.jpg)

（a） 在不同情况下，UC 在 2022-6-1 向 PG ${p}_{t}^{b}$ 购买电力的金额。

![\<img src="attachments/ZAG7XB4B.jpg" alt="" width="1088.235294117647" height="500" data-attachment-key="ZAG7XB4B" ztype="zimage"> | 1088.235294117647](attachments/ZAG7XB4B.jpg)

图 5.2022 年 6-1 武汉不同情况下 UC 的用电量结果。

*   DAP（日平均峰值，kW）：

$$
\mathrm{{DAP}} = \operatorname{mean}\left( {\max {p}_{t}^{b}}\right)  \tag{61}
$$

表 IV

2022 年 6 月纯电力消耗负荷指数

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ---------- | ------ | ------- | ------------------- | ------------------- |
| Load Index | Case I | Case II | Case III            | Case IV             |
| PAR        | 3.7    | 3.58    | 3.28                | 3.03                |
| Ramp       | 3.15e8 | 2.96e8  | ${2.86}\mathrm{e}8$ | ${2.91}\mathrm{e}8$ |
| DAP        | 4,965  | 4,766   | 4,615               | 4,430               |


如表 IV 所示，在框架中的异步作和非合作关系下，与其他情况相比，PAR 和 DAP 下降。PAR 和 DAP 表示 DR 对峰值负荷降低 ${p}_{t}^{b}$ 的影响。然而，Ramp 指数增加，这意味着在所提出的非合作博弈论框架下，消费者以更频繁的方式调整负载。频繁的负载调整会导致舒适度降低。上述结果说明了所提出的 DR 框架的有效性。

图 6a 说明了不同情况下的 DER 削减结果。如图 6a 所示，在情况 IV 中，94.1% 的 DER 生成被减少，3.9% 的 DER 生成被放弃。在情况 I 中，88.7% 的 DER 生成被减少，6.9% 的 DER 生成被放弃。在案例 II 和案例 III 中，90.1% 和 84.6% 的 DER 产生被减少。图 6b 显示了不同情况下峰值负载期、高负载期和平坦负载期的 UC 能耗结果。如图 6b 所示，在情况 IV 中消耗了 ${9.77} \times {10}^{8}\mathrm{\;{kWh}}$ 的功率。与案例 II 和案例 III 相比，案例 IV 的功耗分别降低了 6.7% 和 5.7%。与案例 III 相比，案例 IV 的功耗增加了 1.8%。但是，峰值负载期间的功耗减少了 8.7%。图 6 表明，使用 SN-MAPPO 时，所提出的 DR 框架对 DER 弃风和降低峰值负荷都有更好的效果。

![\<img src="attachments/WCBSKXFX.jpg" alt="" width="395.1890034364261" height="500" data-attachment-key="WCBSKXFX" ztype="zimage"> | 395.1890034364261](attachments/WCBSKXFX.jpg)

（b） 不同情况下 2021 年不同时期的 UC 用电量结果。

图 6.不同情况下不同时期的 DER 弃风结果和 UC 功耗结果（情况 I：完全协作同步框架，情况 II：完全协作异步框架，情况 III：非合作同步框架，情况 IV：建议的框架，非合作异步框架）。

2） UC 收入计算结果：为了说明所提出的框架对 UC 效用的表现，2022 年每个月在不同情况下 UC 的收入和成本计算如图 7 所示。

如图 7 所示，与其他情况相比，拟议的 DR 框架在市场购买力上的支出更少，因为 96% 的 DER 发电被 ESS 和消费者吸收。使用所提出的方法，ESS 将峰值负载转变为平坦负载期。此外，UC 可以通过将历史数据训练成更合理的 ESS 和消费者策略调度来实现更高的效用。

## C. 消费者需求响应结果分析

1） 消费者节省分析：图 8 说明了不同情况下消费者节省的电费，其中 $\Delta {\mathcal{J}}_{c,i} = {\mathcal{J}}_{c,i} - {\mathcal{J}}_{c,i}^{ * }.{\mathcal{J}}_{c,i}^{ * }$ 表示与原始负荷数据相比节省的公用事业。由于强化学习的作用和关税的随机性，为了说明 SN-MAPPO 算法的鲁棒性，针对每种情况进行了 10 次实验。

![\<img src="attachments/FV8LNBJN.jpg" alt="" width="661.4035087719299" height="500" data-attachment-key="FV8LNBJN" ztype="zimage"> | 661.4035087719299](attachments/FV8LNBJN.jpg)

图 7.2021 年平均每月 UC 收入，在不同情况下有误差线（案例 I：完全协作同步框架，案例 II：完全协作异步框架，案例 III：非合作同步框架，案例 IV：拟议框架，非合作异步框架）。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_11.jpg?x=909\&y=1121\&w=746\&h=563\&r=0)

图 8.消费者在不同情况下节省的电费与置信区间。消费者在每种情况下节省的电费置信区间由浅色区间绘制（每个案例 10 个实验）。

如图 8 所示，在提出的 SN-MAPPO 算法下，所有情况都以大约 600 个步骤收敛。然而，在不同情况下，所提出的框架显示出更好的收敛结果。在情况 IV 中，与完全合作的情况 I 和情况 II 相比，消费者达到更大的均衡策略并节省 $ {3740}\$ $and$ {3272}\<span class="math">$$ 1 年 1 年。与同步 DR 框架中的案例 III 相比，所提出的 DR 框架能够达到 1304\$ 的低电费。上述结果表明，所提出的 DR 框架可以进一步降低消费者的电费并确保消费者的利益。此外，忽视时间关系将导致消费者的电力支出增加。在拟议的框架中，消费者通过 TUTT 和参与 DR 灵活性服务来最小化效用。相比之下，与完全合作的 DR 框架相比，消费者必须满足 UC 预期的 DR 能力，消费者会遭受效用损失。

请参阅 https\://www\.ieee.org/publications/rights/index.html

2） 能源成本分析：图 9 说明了 8 个消费者一个月的支出。消费者的支出包括 DR 成本以及 UC 和 DER 的电力成本。如图 9 所示，所提出的 DR 框架在促进需求响应奖励方面具有更好的表现。尽管在拟议的 DR 框架中，消费者购买 UC 电力和 DER 发电的成本较高，但消费者拥有更具竞争力的 DR 策略并获得更高的 DR 奖励。如图 5a 和图 6 所示，消费者能够将需求转移到平坦负载阶段，并在拟议的框架中达到更高的负载减少百分比。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_12.jpg?x=135\&y=543\&w=743\&h=600\&r=0)

图 9.在不同情况下，每个消费者的每月成本 ${\mathcal{J}}_{c,i}$。虚线表示在比较案例（案例 I、II 和 III）下消费者的效用，条形图表示在拟议框架下不同消费者的效用（案例 IV）。

## D. 所提算法的性能

1） 收敛和计算性能分析：图 10 描述了所有基于 RL 的 SNE 算法的 UC 效用 ${\mathcal{J}}_{u}$ 收敛的比较结果。

如图 10 所示，所有基于强化学习的 SNE 算法都可以实现收敛。其中，单智能体 ST-SAC 算法表现出最佳的收敛速度和结果，因为 ST-SAC 不具备消费者智能体部分观察的条件。与其他 MARL 相比，所提出的 SN-MAPPO 在 UC 效用上表现出更好的收敛结果。然而，与 SNE-OVI 相比，SN-MAPPO 的收敛速度较慢。

桌子。V 显示了比较 SNE 算法的计算性能分析。

作为表。V 表明，由于消费者的动作空间巨大，ST-SAC 算法需要更长的时间才能达到 SNE。然而，与其他多代理算法相比，ST-SAC 收敛的发作次数更少。所提出的 SN-MAPPO 算法虽然需要更多的集数才能收敛，但受益于参数共享技术，使其能够在最短的时间内收敛到 SNE。此外，与其他基于 MARL 的 SNE 算法相比，SN-MAPPO 可帮助消费者改进实用程序。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_12.jpg?x=910\&y=154\&w=746\&h=502\&r=0)

图 10.不同算法下 UC 效用 ${\mathcal{J}}_{u}$ 的收敛结果。每种算法的 UC 效用的置信区间由浅色区间绘制（每种算法 10 个实验）。

表 V

不同算法下的计算性能

| <!-- --> | <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ------------- | --------------------- | ----------- | ----------------- | ---------------------- |
| SNE Algorithm | Convergence (Episode) | Time (Hour) | UC Utility (\\$)$ | Consumers Utility (\\) |
| ST-SAC        | 830                   | 4.77        | 8.17e5            | 1.688e6                |
| SNE-OVI       | 751                   | 4.19        | 7.07e5            | 1.558e6                |
| ST-DT         | 941                   | 3.71        | 7.29e5            | 1.569e6                |
| ST-MADDPG     | 986                   | 5.02        | 6.76e5            | 1.547e6                |
| SN-MAPPO      | 897                   | 4.21        | 7.17e5            | 1.621e6                |


2） 需求响应结果分析：图 11 显示了在基于 RL 的 SNE 算法下从 PG 购买的 UC 的电力。

桌子。VI 显示 ${p}_{t}^{b}$ 的负荷指数，以证明 DR 结果的改善。

表 VI

纯电力消耗负荷指数

| <!-- --> | <!-- --> | <!-- --> | <!-- --> |
| ----------------------------- | ---------------- | ------------------- | ----- |
| $\mathbf{{Algorithm}}$        | $\mathbf{{PAR}}$ | Ramp                | DAP   |
| Algorithm I: ST-SAC \[40]     | 2.97             | 3.11e8              | 4,291 |
| Algorithm II: SNE-OVI \[49]   | 3.19             | 3.22e8              | 4,671 |
| Algorithm III: ST-DT \[50]    | 3.15             | 3.17e8              | 4,599 |
| Algorithm IV: ST-MADDPG \[51] | 3.22             | 3.11e8              | 4,522 |
| Proposed SN-MAPPO             | 3.03             | ${2.91}\mathrm{e}8$ | 4,430 |


如图 11 所示，ST-SAC 仍然具有最佳的 DR 负载结果，因为 ST-SAC 忽略了部分可观察的假设，即所有消费者共享信息和策略。除 ST-SAC 外，SN-MAPPO 在降低峰值负载方面表现出更好的性能。作为表。VI 表明，SN-MAPPO 的 PAR 低于基于 MARL 的 SNE 算法。SN-MAPPO 的低 PAR 指数说明所提算法具有将负载从峰值负载期转移到平坦负载期的能力。此外，由于较低的 Ramp 和 DAP 指数，SN-MAPPO 被证明比基于 MARL 的 SNE 算法更鲁棒，在不同日期内效果更好。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_13.jpg?x=134\&y=156\&w=743\&h=495\&r=0)

图 11.2022 年 6 月 1 日根据基于 RL 的 SNE 算法从 PG ${p}_{t}^{b}$ 购买的 UC 电力。负载数据每 15 分钟有一个点，全天总共有 96 个数据点。

## E. 敏感性分析

为了比较不同类型电价对 DR 结果的影响，应用了电价峰谷比、DER 电价峰谷比和 DR 激励价格的敏感性分析，分别证明了所提算法的稳健性。

1） 电费敏感性分析：图 12 说明了电费对消费者 ${\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right)$ 的敏感性分析。电价的峰谷比定义为 $\frac{\mathop{\max }\limits_{t}{\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right) }{\mathop{\min }\limits_{t}{\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right) }$，峰高电价比定义为 $\frac{\mathop{\max }\limits_{t}{\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right) }{\mathop{\operatorname{mid}}\limits_{t}{\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right) }$ ，这反映了框架中随时间推移的电价波动。

如图 12 所示，随着峰谷电价比和峰高电价比的增加，峰值负荷期的用电量降低。当峰谷电价比大于 1.7 时，高峰期负荷逐渐向平坦负荷期转移。当峰高比大于 1.25 时，峰值负荷期的负荷逐渐转移到平坦负荷期。

2） 需求响应奖励的敏感性分析：图 13 说明了 DR 结果对 DR 灵活性购买价格 ${\mathcal{T}}^{r}$ 的敏感性分析。DR 灵活性幂结果，包括 DR 灵活性幂 $\mathop{\sum }\limits_{i}\Delta {E}_{i,t}$ 和 PAR 因子，通过遍历 $ {0.01}\$ /\mathrm{{kWh}} $to$ {0.1}\<span class="math">$/\mathrm{{kWh}}$ 的 DR 灵活性购买价格进行实验，以证明 DR 灵活性购买价格对消费者 DR 策略的影响。

如图 13 所示，随着 DR 灵活性购买价格的上升，消费者的 DR 灵活性能力逐渐上升，PAR 指数下降，直到 DR 灵活性购买价格 ${\mathcal{T}}^{r}$ 达到 $ {0.05}\$ /\mathrm{{kWh}} $. The consumer demand response effect reaches its limit at the DR flexibility power price of$ {0.05}\<span class="math">$/\mathrm{{kWh}}$ 。随着 DR 灵活性价格的增加，UC 的效用会降低，从而导致 UC 的策略给出较低的 ${p}_{t}^{m}$。DR 灵活性价格限制了 UC 的预期需求响应能力，这反映了消费者与 UC 之间的博弈关系。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_13.jpg?x=917\&y=156\&w=739\&h=482\&r=0)

图 12.负荷消耗的敏感性分析导致不同时期的电费 ${\mathcal{T}}^{u}\left( {t,{E}_{i,t}^{u}}\right)$ 。峰谷电价比从 1.4 到 2.6，步长为 0.1，而峰高比从 1.1 到 1.4，步长为 0.05。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_13.jpg?x=908\&y=800\&w=751\&h=458\&r=0)

图 13.DR 灵活性幂和 PAR 因子对 DR 灵活性购买价格 ${\mathcal{T}}^{r}$ 的敏感性分析。DR 灵活性购买价格从 $ {0.01}\$ /\mathrm{{kWh}} $to$ {0.1}\<span class="math">$/\mathrm{{kWh}}$ 以 $ {0.01}\$ /\mathrm{{kWh}} \$ 为步长遍历。

3） DER 关税的敏感性分析：图 14 说明了 DER 关税因子 ${\mathcal{T}}_{2}^{d}$ 的 DER 削减结果的敏感性分析。

如图 14 所示，随着 ${\mathcal{T}}_{2}^{d}$ 的降低，DER 放弃率逐渐降低，而 DER 充电功率逐渐降低。随着消费者消耗更多的 DER 发电，ESS 用于负载需求转移和降低电力市场的功率。

## V. 结论

该文提出了一种新的双级动态非合作 UC 消费者博弈理论 DR 框架和一种异步 MARL 算法，命名为 SN-MAPPO。拟议的 DR 框架考虑了 UC 和消费者之间的 Stackelberg 博弈以及消费者之间的 Nash 博弈，其中配电网的 DER 弃风率增加，负荷波动减少。结果验证了所提出的 DR 框架在减少 DER 弃风和减少负荷波动方面的有效性。在中国国网湖北电力有限公司提供的真实数据案例研究中，涉及 1 个 UC 和 8 个电力用户，对所提方法进行了训练、实施，并与 3 个案例和 4 个最先进的算法进行了比较。结果验证了拟议的 DR 框架在负载波动、DER 削减和节省电力成本方面的表现。算法比较表明，SN-MAPPO 具有更大的收敛结果和负载结果以及可扩展性和较低的计算复杂度。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_14.jpg?x=134\&y=158\&w=750\&h=444\&r=0)

图 14.DER 放弃功率 ${E}_{t}^{a}$ 和 ESS ${E}_{t}^{e}$ 的 DER 功率对 DER 关税因子 ${\mathcal{T}}^{r}$ 的敏感性分析。DR 灵活性购买价格从 $ - 1 \times {10}^{-7}\$ /\mathrm{{kWh}}/\mathrm{{kW}} $to$ - 1 \times {10}^{-6}\<span class="math">$/\mathrm{{kWh}}/\mathrm{{kW}}$ 以 $ - 1 \times {10}^{-7}\$ /\mathrm{{kWh}} \$ 为步长遍历。

在未来的工作中，可以考虑在非合作 Stackelberg-Nash 框架下考虑公用事业竞争性招标和点对点能源交易机制的情况。此外，考虑到 DR 消费者数量的增长，值得考虑在 MARL 框架下针对类似 DR 消费者的数据驱动用户分析和聚合等效算法。引用

\[1] P. Palensky 和 D. Dietrich，“需求侧管理：需求响应、智能能源系统和智能负载”，IEEE Trans. Ind. Informat.，第 7 卷，第 3 期，第 381-388 页，2011 年。

\[2] Karapetyan 等人，“孤岛微电网中在线需求响应的竞争性调度算法”，IEEE Trans. Power Syst.，第 36 卷，第 4 期，第 3430-3440 页，2021 年 7 月。

\[3] Q. Shi， F. Li， G. Liu， D. Shi， Z. Yi， and Z. Wang， “考虑每日需求曲线和渐进式恢复的系统频率调节的恒温负载控制”，IEEE Trans. Smart Grid，第 10 卷，第 6 期，第 6259-6270 页，2019 年 11 月。

\[4] Z. Zhu， S. Lambotharan， W. H. Chin， and Z. Fan， “结合本地能源的家庭需求管理的博弈论优化框架”，IEEE Trans. Ind. Informat.，第11卷，第2期，第353-362页，2015年4月。

\[5] M. Yu 和 S. H. Hong，“考虑分层电力市场的基于激励的需求响应：一种斯塔克尔伯格博弈方法”，《应用能源》，第 203 卷，第 267-279 页，2017 年 10 月。

\[6] P. Jacquot、O. Beaude、S. Gaubert 和 N. Oudjane，“需求响应管理的小时计费机制的分析和实施”，IEEE Trans. Smart Grid，第 10 卷，第 4 期，第 4265-4278 页，2019 年 7 月。

\[7] W. Chen， J. Qiu， J. Zhao， Q. Chai， and Z. Y. Dong， “使用分层博弈和强化学习方法为虚拟电厂定制返利定价机制”，IEEE Trans. Smart Grid，第 14 卷，第 1 期，第 424-439 页，2023 年 1 月。

\[8] Z. Luo， S.-H.Hong 和 J.-B.Kim，“智能电网中离散制造的基于价格的需求响应方案”，《能源》，第 9 卷，第 8 期，第 650 页，2016 年 8 月。

\[9] R. Lu 和 S. H. Hong，“具有强化学习和深度神经网络的智能电网基于激励的需求响应”，Appl. Energy，第 236 卷，第 937-949 页，2019 年 2 月。

\[10] Y. Chen， W. Wei， H. Wang， Q. 周， and J. P. S. Catalão， “实现与集中调度相同灵活性的能源共享机制”， IEEE Trans. Smart Grid， vol. 12， no. 4， pp. 3379-3389， Jul. 2021.

\[11] P. Srikantha 和 D. Kundur，“通过人口博弈进行弹性分布式实时需求响应”，IEEE Trans. Smart Grid，第 8 卷，第 6 期，第 2532-2543 页，2017 年 11 月。

\[12] D. Xie， M. Liu， L. Xu， and W. Lu， “配电公司参与的电力市场多人纳什-斯塔克尔伯格博弈分析”，IEEE Syst. J.，第 17 卷，第 3 期，第 3658-3669 页，2023 年 9 月。

\[13] Y. Wang， W. Saad， Z. Han， H. V. Poor， and T. Başar， “智能电网中能源交易的博弈论方法”，IEEE Trans. Smart Grid，第5卷，第3期，第1439-1450页，2014年5月。

\[14] L. Yu， P. Wang， Z. Chen， D. Li， N. Li， and R. Cherkaoui， “在不完美电力市场中基于强化学习寻找竞价策略和分布式 iso 算法的纳什均衡”，Appl. Energy，第 350 卷，第 121704 页，2023 年 11 月。

\[15] H. Le Cadre、P. Jacquot、C. Wan 和 C. Alasseur，“点对点电力市场分析：从变分到广义纳什均衡”，Eur. J. Oper。Res.，第 282 卷，第 2 期，第 753-771 页，2020 年 4 月。

\[16] V.-H.Bui、A. Hussain 和 H.-M.Kim，“考虑可调功率和需求响应的多微电网基于多智能体的分层能源管理策略”，IEEE Trans. Smart Grid，第 9 卷，第 2 期，第 1323-1333 页，2018 年 3 月。

\[17] Y. Ding， D. Xie， H. Hui， Y. Xu， and P. Siano， “用于平滑微电网连接线功率的恒温控制负载的博弈论需求侧管理”，IEEE Trans. Power Syst.，第 36 卷，第 5 期，第 4089-4101 页，2021 年 9 月。

\[18] B. Wang、Y. Li、W. Ming 和 S. Wang，“用于可中断负载需求响应管理的深度强化学习方法”，IEEE Trans. Smart Grid，第 11 卷，第 4 期，第 3146-3155 页，2020 年 7 月。

\[19] D. Qiu， J. Wang， Z. Dong， Y. Wang， and G. Strbac， “Mean-field multi-agent reinforcement learning for peer-to-peer multi-energy trading”，IEEE Trans. Power Syst.，第 38 卷，第 5 期，第 4853-4866 页，2023 年 9 月。

\[20] X. Xu， Y. Jia， Y. Xu， Z. Xu， S. Chai， and C. S. Lai， “一种基于多智能体强化学习的家庭能源管理数据驱动方法”，IEEE Trans. Smart Grid，第 11 卷，第 4 期，第 3201-3211 页，2020 年 7 月。

\[21] M. Franceschelli、A. Pilloni 和 A. Gasparri，“用于电力需求侧管理的智能电源插座对恒温控制负载进行多代理协调”，IEEE Trans. Control Syst. Technol.，第 29 卷，第 2 期，第 731-743 页，2021 年 3 月。

\[22] Y. Meng， S. Fan， Y. Shen， J. Xiao， G.He， and Z. Li， “Transmission and distribution network-constrained large-scale demand response based on locational customer directrix load for accommodating renewable energy”，《应用能源》，第 350 卷，第 121681 页，2023 年 11 月。

\[23] K. 任、J. Liu、X. Liu 和 Y. Nie，“电力和天然气综合市场中基于强化学习的燃气机组二级战略投标模型防止市场纵”，应用能源，第 336 卷，第 120813 页，2023 年 4 月。

\[24] K. 任、J. Liu、Z. Wu、X. Liu、Y. Nie 和 H. Xu，“考虑不确定家庭参数的数据驱动的基于 DRL 的家庭能源管理系统优化框架”，Appl. Energy，第 355 卷，第 122258 页，2024 年 2 月。

\[25] A.-H.Mohsenian-Rad、V. W. S. Wong、J. Jatskevich、R. Schober 和 A. Leon-Garcia，“基于博弈论能耗调度的未来智能电网的自主需求侧管理”，IEEE Trans. Smart Grid，第 1 卷，第 3 期，第 320-331 页，2010 年 12 月。

\[26] E. R. Stephens、D. B. Smith 和 A. Mahanti，“分布式能源需求侧管理的博弈论模型预测控制”，IEEE Trans. Smart Grid，第 6 卷，第 3 期，第 1394-1402 页，2015 年 5 月。

\[27] X. Liu， B. Gao， C. Wu， and Y. Tang， “家用插电式电动汽车的需求侧管理：一种贝叶斯博弈论方法”，IEEE Syst. J.，第 12 卷，第 3 期，第 2894-2904 页，2018 年 9 月。

\[28] CP Mediwaththe、ER斯蒂芬斯、D. B. 史密斯和 A. Mahanti，“邻里网络需求方管理的竞争性能源交易框架”，IEEE Trans. 智能电网，第 9 卷，第 5 期，第 4313-4322 页，2018 年 9 月。

\[29] W. Chen， J. Qiu， J. Zhao， Q. Chai， and Z. Y. Dong， “使用分层博弈和强化学习方法为虚拟电厂定制返利定价机制”，IEEE Trans. Smart Grid，第 14 卷，第 1 期，第 424-439 页，2023 年 1 月。

\[30] E. Nekouei、T. Alpcan 和 D. Chattopadhyay，“电力市场需求响应的博弈论框架”，IEEE Trans. Smart Grid，第 6 卷，第 2 期，第 748-758 页，2015 年 3 月。

\[31] K. Alshehri、J. Liu、X. Chen 和 T. Başar，“智能电网中多周期-多公司需求响应管理的博弈论框架”，IEEE Trans. Control Syst. Technol.，第 29 卷，第 3 期，第 1019-1034 页，2021 年 5 月。

\[32] N. Qi、L. Cheng、H. Xu、Z. Wang 和 X. 周，“聚合客户的空调负载的实际需求响应潜力评估”，能源代表，第 6 卷，第 71-81 页，2020 年 12 月。

\[33] M. Yu、S. H. Hong、Y. Ding 和 X. Ye，“考虑复合 DR 资源的基于激励的需求响应 （dr） 模型”，IEEE Trans. Ind. Electron.，第 66 卷，第 2 期，第 1488-1498 页，2019 年 2 月。

\[34] Y. Yang 和 J. Wang，“博弈论视角下的多智能体强化学习概述”，arXiv：2011.00583 \[cs.MA]，2021 年 3 月。\[在线]。可用： <https://doi.org/10.48550/arXiv.2011.00583>

\[35] Muller 等人，“多智能体学习的通用训练方法”，arXiv：1909.12823 \[cs.MA]，2020 年 2 月。\[在线]。可用： <https://doi.org/10.48550/arXiv.1909.12823>

\[36] R. Lowe， Y. Wu， A. Tamar， J. Harb， P. Abbeel， and I. Mordatch， “Multi-agent actor-critic for mixed cooperative-competitive environments，” arXiv：1706.02275 \[cs.LG]，2020 年 3 月。\[在线]。可用地址： https：//doi.org/10.48550/arXiv.1706.02275

\[37] 周 et al.， “具有分散执行框架的集中式训练是否足以集中于马尔？”arXiv：2305.17352 \[cs.AI]，2023 年 5 月。\[在线]。可用： <https://doi.org/10.48550/arXiv.2305.17352>

\[38] Brero 等人，“Stackelberg pomdp：一种用于经济设计的强化学习方法”，arXiv：2210.03852 \[cs.GT]，2023 年 2 月。\[在线]。可用： <https://doi.org/10.48550/arXiv.2210.03852>

\[39] S. Bandyopadhyay、C. Zhu、P. Daniel、J. Morrison、E. Shay 和 J. Dickerson，“强化学习中的目标以解决 stackelberg 安全游戏”，arXiv：2211.17132 \[cs.LG]，2022 年 11 月。\[在线]。可用： <https://doi.org/10.48550/arXiv.2211.17132>

\[40] L. Zheng， T. Fiez， Z. Alumbaugh， B. Chasnov， and L. J. Ratliff， “Stack-elberg actor-critic： 博弈论强化学习算法”，arXiv：2109.12286.

\[41] J. L. Yu Zhao， Y. N. Xiaoming Liu， and X. Liu， “Dnopernet： A generic network architecture for scalable and safe deep reinforcement learning，” in 2024 IEEE Power & Energy Society General Meeting （PESGM）， 2024， 即将发布。

\[42] T. Schaul， J. Quan， I. Antonoglou， and D. Silver， “优先体验重播”， arXiv：1511.05952 \[cs.LG]，2016 年 2 月。\[在线]。可用： <https://doi.org/10.48550/arXiv.1511.05952>

\[43] J. G. Kuba， R. Chen， M. 温， Y. 温， F. Sun， J. Wang， 和 Y. Yang， “多智能体强化学习中的信任区域策略优化”，arXiv：2109.11251,2022 年 4 月。

\[44] G. Brero， A. Eden， D. Chakrabarti， M. Gerstgrasser， V. Li， and D. C. Parkes， “Stackelberg pomdp： A reinforcement learning approach for economic design，” arXiv：2210.03852， 2023年2月.

\[45] “湖北省政府信息公开网站”，2023 年。\[在线]。可用： <https://www.hubei.gov.cn/>

\[46] J. R. Vazquez-Canteli、S. Dey、G. Henze 和 Z. Nagy，“城市学习：标准化需求响应和城市能源管理的多智能体强化学习研究”，arXiv：2012.10504 \[cs.LG]，2020 年 12 月。\[在线]。可用： <https://doi.org/10.48550/arXiv.2012.10504>

\[47] Zhang 等人，“基于住宅多载流子能源系统的多智能体深度强化学习方法的双时间尺度自主能源管理策略”，应用能源，第 351 卷，第 121777 页，2023 年 12 月。

\[48] Y. Shi 和 B. Zhang，“库尔诺游戏中的多智能体强化学习”，arXiv：2009.06224,2020 年 9 月。

\[49] H. Zhong， Z. Yang， Z. Wang， and M. I. Jordan， “强化学习能否在近视追随者的广义和马尔可夫博弈中找到 stackelberg-nash 均衡？arXiv：2112.13521,2021 年 12 月。

\[50] 张斌、毛浩、李丽、徐志、李丹、赵和范国，“多智能体系统中异步动作协调的 Stackelberg 决策转换器”，2023 年 5 月。

\[51] B. Yang， L. Zheng， L. J. Ratliff， B. Boots， and J. R. Smith， “Stackelberg games for learning emergent behaviors during competitive autocurric-ula，” arXiv：2305.03735， 2023 年 5 月.

\[52] Liang et al.， “Rllib： Abstractions for distributed reinforcement learning，” arXiv：1712.09381 \[cs.AI]，2018 年 6 月。\[在线]。可用： <https://doi.org/10.48550/arXiv.1712.09381>

\[53] Paszke 等人，“Pytorch：一种命令式、高性能深度学习库”，arXiv：1912.01703 \[cs.LG]，2019 年 12 月。\[在线]。可用： <https://doi.org/10.48550/arXiv.1912.01703>

\[54] R. Liaw， E. Liang， R. Nishihara， P. Moritz， J. E. Gonzalez， and I. Stoica， “Tune： A research platform for distributed model selection and training，” arXiv：1807.05118， 2018.

\[55] J. Terry， B. Black， N. Grammel， M. Jayakumar， A. Hari， R. Sullivan， L. S. Santos， C. Dieffendahl， C. Horsch， R. Perez-Vicente et al.， “Pettingzoo： Gym for multi-agent reinforcement learning，” Proc. Adv. Neural Inf. Process.系统，第 34 卷，第 15032-15043 页，2021 年。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_15.jpg?x=917\&y=771\&w=210\&h=268\&r=0)

聂永欣（IEEE 学生会员）于 2022 年获得中国习安习安交通大学电气工程学士学位。他目前正在中国习安的习安交通大学攻读电气工程硕士学位。他的研究兴趣主要包括智能电网、多智能体强化学习、博弈论和LLM智能体在电力系统中的应用。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_15.jpg?x=916\&y=1099\&w=207\&h=268\&r=0)

刘军（IEEE高级会员）分别于2004年和2012年在中国习习安交通大学获得电气工程学士和博士学位。现任习安交通大学电气工程学院教授;兼任习安交通大学陕西省智能电网智能电网重点实验室、电气绝缘与电力设备国家重点实验室研究员。2008 年 10 月至 2010 年 8 月，他在美国德克萨斯州 College Station 的德克萨斯 A\&M 大学电气与计算机工程系担任访问学者。他的兴趣是电力系统稳定性、可再生能源和电动汽车集成、电力系统运行和控制、智能电网、综合能源系统、机器学习以及电力和能源系统中的人工智能应用。

刘晓明（IEEE学生会员）于2019年获得中国习安习安交通大学电气工程学士学位。他目前正在中国习安的习安交通大学攻读电气工程博士学位。他的研究兴趣主要包括电力系统风险评估、电力系统运行和控制、智能电网。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_15.jpg?x=927\&y=1566\&w=190\&h=225\&r=0) ![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_15.jpg?x=917\&y=1877\&w=213\&h=270\&r=0)

赵宇（IEEE 学生会员）于 2021 年获得中国习安习安交通大学电气工程学士学位。他目前正在中国习安的习安交通大学攻读电气工程博士学位。他的研究兴趣主要包括电力系统运行和控制、信息物理系统以及人工智能在能源/电力系统中的应用。本文已被 IEEE Transactions on Smart Grid 接受发表。这是作者的版本，尚未完全编辑，内容可能会在最终发布之前发生变化。引用信息：DOI 10.1109/TSG.2024.3417535

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_16.jpg?x=141\&y=173\&w=217\&h=275\&r=0)

任克政（IEEE学生会员）于2024年获得习安交通大学电气工程硕士学位。他的研究兴趣包括需求响应管理、家庭和社区能源管理系统、电力系统运行和控制。

![](https://cdn.noedgeai.com/01978cb1-96d7-7c75-a918-697a34a47a88_16.jpg?x=149\&y=616\&w=199\&h=275\&r=0)

陈晨（IEEE高级会员）分别于2006年和2009年获得中国习安习安交通大学（XJTU）的学士和硕士学位，并于2013年获得美国宾夕法尼亚州伯利恒的利哈伊大学电气工程博士学位。他目前是西安交通大学电气工程学院的教授。在加入西安交通大学之前，他在美国伊利诺伊州莱蒙特的阿贡国家实验室工作了六年多，最后被任命为能源系统部的能源系统科学家。他的研究兴趣包括电力系统弹性、配电系统和微电网、需求侧管理、智能电网的通信和信号处理。他还是 2017 年 IEEE PES 芝加哥分会杰出工程师奖的获得者。他是 IEEE TRANSACTIONS ON SMART GRID 和 IEEE POWER ENGINEERING LETTERS 的编辑。
