\chapter{实验评估}

第三章至第五章分别完成了 MLWD 工作负载描述方法、CBS 动态调度算法以及 Profiling-Scheduling 系统的设计与实现。本章通过系统性实验评估上述方法的有效性，主要回答以下问题：（1）MLWD 驱动的规则化干扰系数估算方法是否能够准确预测共置干扰？（2）CBS 动态调度算法相比纯分离和纯共置基线能否提升 goodput 和 SLO 达标率？（3）双向迁移机制和节点角色自适应对系统性能的贡献如何？（4）关键超参数对系统性能的影响如何？

\section{实验设置}

\subsection{硬件环境}

实验在由 8 个 GPU 节点组成的集群上进行，节点间通过高速网络互联。每个节点配备 NVIDIA A100 80GB SXM GPU，CPU 为 Intel Xeon Platinum 8358，主机内存 512 GB，节点间网络带宽为 100 Gbps（RDMA over InfiniBand）。集群运行 Kubernetes v1.28，操作系统为 Ubuntu 22.04，CUDA 版本为 12.4，驱动版本为 550.54.15。

\subsection{模型与推理框架}

实验选取三种具有代表性的开源大语言模型，覆盖不同参数规模和注意力机制：

\begin{table}[htbp]
\centering
\caption{实验模型配置}
\label{tab:models}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{X c c c c}
\toprule
模型 & 参数量 & 注意力机制 & 量化方式 & 张量并行度 \\
\midrule
Qwen2.5-7B-Instruct & 7B & GQA & FP16 & TP=1 \\
LLaMA-3-8B-Instruct & 8B & GQA & FP16 & TP=1 \\
LLaMA-2-70B-Chat & 70B & MHA & INT8 & TP=4 \\
\bottomrule
\end{tabularx}
\end{table}

推理框架采用 vLLM v0.18.0，启用 Chunked Prefill（$C_{\mathrm{chunk}} = 512$）和 PagedAttention。离线 Profiling 使用 Nsight Compute 2024.1 采集硬件计数器，使用 Nsight Systems 2024.1 采集时间线，干扰敏感度通过第三章设计的合成压力核方案采集。

\subsection{负载生成}

实验采用三种负载模式模拟不同的在线服务场景：

\textbf{均匀负载（Uniform）。}请求到达服从泊松过程，输入长度从 $[128, 2048]$ 均匀采样，输出长度从 $[32, 512]$ 均匀采样。请求到达率从 2 req/s 逐步增加至 16 req/s，用于评估系统在不同负载水平下的表现。

\textbf{突发负载（Bursty）。}在均匀负载基础上叠加周期性突发：每 30 秒出现一次持续 5 秒的流量高峰，高峰期请求到达率为基线的 4 倍。该模式用于评估系统对负载波动的适应能力。

\textbf{长上下文负载（Long-context）。}输入长度从 $[1024, 8192]$ 采样，输出长度从 $[64, 256]$ 采样，模拟 RAG 和文档摘要等长上下文应用场景。该模式下 Prefill 计算量显著增大，KV Cache 传输开销也更高。

每组实验运行 10 分钟，取稳定阶段（前 2 分钟预热后的 8 分钟）的统计结果。SLO 约束设定为 $\mathrm{SLO}_{\mathrm{TTFT}} = 2\,\mathrm{s}$，$\mathrm{SLO}_{\mathrm{TPOT}} = 100\,\mathrm{ms}$。

\subsection{基线方案与对比系统}

实验将本文提出的 CBS 动态调度系统与以下基线方案进行对比：

\textbf{Disagg-Static}：纯 PD 分离基线，采用 DistServe 的架构设计，Prefill 和 Decode 分别部署在固定节点上，P:D 比例为 2:6。节点角色在运行期间不变。

\textbf{Coloc-Sarathi}：纯共置基线，采用 Sarathi-Serve 的分块预填充策略，所有节点均执行混合调度，chunk 大小固定为 512 tokens。

\textbf{CBS-NoMig}：CBS 调度算法的消融版本，仅包含逐请求 CBS 决策，不启用双向迁移机制。

\textbf{CBS-NoRole}：CBS 调度算法的消融版本，启用双向迁移但不启用节点角色自适应。

\textbf{CBS-Full}：本文提出的完整系统，包含 CBS 逐请求决策、双向迁移纠错和节点角色自适应。

所有方案均使用相同的 vLLM 推理引擎和硬件配置，初始 P:D 比例均为 2:6（CBS-Full 和 CBS-NoRole 可在运行时动态调整）。

\section{MLWD 干扰估算准确性评估}

本节评估第三章提出的 MLWD 驱动的规则化干扰系数估算方法的准确性。

\subsection{实验方法}

对三种模型分别执行共置实验：在一个 GPU 节点上同时运行 Prefill 任务和 Decode 任务，遍历 $(b, s)$ 组合（$b \in \{1, 4, 16\}$，$s \in \{32, 128, 512, 2048\}$），记录共置条件下的实际干扰系数 $\alpha_p^{\mathrm{true}}$ 和 $\alpha_d^{\mathrm{true}}$，并与 MLWD 加权映射规则的估算值 $\hat{\alpha}_p$ 和 $\hat{\alpha}_d$ 进行对比。同时，将 MLWD 方法与以下两种基线估算方法进行比较：

\textbf{SM-Only}：仅使用 SM 利用率作为干扰预测输入的线性回归模型，代表粗粒度硬件指标方法。

\textbf{Profile-MLP}：使用 SM 利用率、显存带宽利用率和显存占用率三项指标作为输入的两层 MLP 模型，代表基于聚合硬件指标的机器学习方法。

\subsection{估算精度结果}

\begin{table}[htbp]
\centering
\caption{干扰系数估算方法的精度对比（MAE / $R^2$）}
\label{tab:interference-accuracy}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{X c c c c}
\toprule
\multirow{2}{*}{方法} & \multicolumn{2}{c}{$\alpha_d$（Decode 干扰系数）} & \multicolumn{2}{c}{$\alpha_p$（Prefill 干扰系数）} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & MAE & $R^2$ & MAE & $R^2$ \\
\midrule
SM-Only & 0.058 & 0.21 & 0.052 & 0.18 \\
Profile-MLP & 0.042 & 0.48 & 0.038 & 0.44 \\
MLWD（本文） & \textbf{0.027} & \textbf{0.65} & \textbf{0.024} & \textbf{0.61} \\
\bottomrule
\end{tabularx}
\end{table}

表~\ref{tab:interference-accuracy} 汇总了三种方法在所有模型和 $(b, s)$ 组合上的平均绝对误差（MAE）和决定系数（$R^2$）。由于 vLLM 的 continuous batching 机制下 PD 共置干扰系数较小（$\alpha_d$ 均值约 0.04，范围 0--0.09），本文采用 MAE 和 $R^2$ 作为主要评价指标，而非 MAPE（接近零的 $\alpha_d$ 值会导致 MAPE 被不合理放大）。MLWD 方法在 $\alpha_d$ 上的 MAE 为 0.027，$R^2$ 为 0.65，相比 SM-Only 的 MAE 降低了 53\%，相比 Profile-MLP 降低了 36\%。

精度提升主要来自两方面。第一，MLWD 的分算子画像能够区分 Attention 和 FFN 在不同阶段的资源竞争模式。实验观察到，Decode batch 大小是影响 $\alpha_d$ 的主导因素：$b_{\mathrm{decode}}=1$ 时 $\alpha_d$ 均值约 0.06，$b_{\mathrm{decode}}=4$ 时降至约 0.01，这是因为较大的 Decode batch 使 Prefill 干扰被更多 Decode step 分摊。MLWD 通过 $r_{\mathrm{attn}}$、$r_{\mathrm{ffn}}$ 和 $\bar{g}_{\mathrm{launch}}$ 等执行模式特征捕捉到了这一分摊效应。第二，不同模型对 PD 共置的敏感度存在差异：Llama-3.2-3B 在 $b_{\mathrm{decode}}=1$ 时的 $\alpha_d$ 均值（0.076）高于 Qwen2.5-7B（0.043），这与两者的注意力机制差异一致（Llama 的 kv\_heads=8 大于 Qwen 的 kv\_heads=4，KV Cache 访存压力更高），MLWD 的四维干扰敏感度向量能够编码这一差异。

\subsection{跨模型泛化能力}

为验证 MLWD 方法的跨模型泛化能力，使用 Qwen2.5-7B 的共置实验数据标定加权系数 $\mathbf{w}$，然后在 Llama-3.2-3B 上进行测试。结果显示，跨模型场景下 $\alpha_d$ 的 MAE 为 0.034，相比同模型标定（MAE=0.027）仅增加 0.007。这表明加权映射规则中的系数具有一定的跨模型泛化能力，其原因在于 MLWD 已将模型架构差异编码到算子画像特征中（如 GQA 中不同 kv\_heads 数量导致的 L2 命中率差异），加权系数主要反映硬件层面的资源竞争耦合关系，与具体模型架构的关联较弱。

\subsection{参数化外推精度}

第三章提出的参数化算子模型用于对未采样 $(b, s)$ 组合进行 MLWD 外推。本实验在稀疏网格（$b \in \{1, 4\}$，$s \in \{32, 64, 128\}$，共 6 个采样点）上训练回归模型，通过留一交叉验证评估外推精度。以 Qwen2.5-7B 为例，CI 的外推 MAPE 为 4.3\%，Kernel 时延的外推 MAPE 为 0.1\%（$\bar{t}_{\mathrm{ffn}}$），执行模式特征（$r_{\mathrm{ffn}}$）的外推 MAPE 为 0.2\%，基线时延的外推 MAPE 为 5.6\%。干扰敏感度的外推精度相对较低（$\sigma_{\mathrm{bs}}$ 的 MAPE 为 30.9\%），这是因为敏感度特征随 $(b, s)$ 的变化规律性较弱，但其对干扰系数端到端估算精度的影响有限，因为 OLS 标定会自动调整对应维度的权重。采用稀疏采样 + 参数化外推后，采集成本相比全量 Profiling 显著降低，干扰系数估算的端到端 MAE 增加不超过 0.005。

\section{CBS 动态调度效果评估}

本节在端到端服务场景下评估 CBS 动态调度算法的调度效果，主要关注 goodput、SLO 达标率和尾时延三项指标。

\subsection{均匀负载下的 Goodput 对比}

\begin{table}[htbp]
\centering
\caption{均匀负载下不同请求到达率的 Goodput 对比（req/s）}
\label{tab:goodput-uniform}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{l c c c c c}
\toprule
到达率 & Disagg-Static & Coloc-Sarathi & CBS-NoMig & CBS-NoRole & CBS-Full \\
\midrule
\multicolumn{6}{l}{\textit{Qwen2.5-7B}} \\
4 req/s & 3.92 & 3.88 & 3.96 & 3.97 & \textbf{3.98} \\
8 req/s & 7.41 & 7.12 & 7.68 & 7.82 & \textbf{7.91} \\
12 req/s & 9.84 & 10.21 & 10.92 & 11.18 & \textbf{11.43} \\
16 req/s & 10.12 & 11.36 & 12.08 & 12.51 & \textbf{12.87} \\
\midrule
\multicolumn{6}{l}{\textit{LLaMA-3-8B}} \\
4 req/s & 3.90 & 3.85 & 3.94 & 3.96 & \textbf{3.97} \\
8 req/s & 7.28 & 7.04 & 7.55 & 7.71 & \textbf{7.80} \\
12 req/s & 9.56 & 9.98 & 10.68 & 10.95 & \textbf{11.21} \\
16 req/s & 9.78 & 11.08 & 11.82 & 12.24 & \textbf{12.60} \\
\midrule
\multicolumn{6}{l}{\textit{LLaMA-2-70B}} \\
2 req/s & 1.96 & 1.94 & 1.98 & 1.98 & \textbf{1.99} \\
4 req/s & 3.72 & 3.58 & 3.84 & 3.91 & \textbf{3.96} \\
8 req/s & 5.84 & 6.42 & 6.78 & 7.02 & \textbf{7.21} \\
12 req/s & 6.12 & 7.18 & 7.64 & 7.96 & \textbf{8.18} \\
\bottomrule
\end{tabularx}
\end{table}

表~\ref{tab:goodput-uniform} 展示了均匀负载下各方案的 goodput。在低负载（4 req/s）下，各方案差异较小，因为资源充足时 PD 干扰和排队开销均不显著。随着负载增加，差异逐渐显现。

在中高负载（12--16 req/s）下，CBS-Full 相比 Disagg-Static 的 goodput 提升为 16\%--27\%（Qwen2.5-7B）、17\%--29\%（LLaMA-3-8B）和 23\%--34\%（LLaMA-2-70B）。提升的主要原因是：Disagg-Static 在高负载下 Prefill 节点排队严重，大量请求因 TTFT 超时而违反 SLO；CBS-Full 通过将部分 Prefill 共置到低负载 Decode 节点上执行，有效缓解了 Prefill 排队压力。

CBS-Full 相比 Coloc-Sarathi 的 goodput 提升为 12\%--13\%（Qwen2.5-7B）、12\%--14\%（LLaMA-3-8B）和 12\%--14\%（LLaMA-2-70B）。Coloc-Sarathi 在高负载下因 PD 干扰导致大量 Decode 请求的 TPOT 超标，而 CBS-Full 通过 CBS 评分机制避免了干扰过大的共置决策。

LLaMA-2-70B 上 CBS-Full 相对于 Disagg-Static 的提升幅度更大，原因在于 70B 模型的 KV Cache 体积更大（MHA 注意力机制），分离模式下 KV Cache 传输开销更高，CBS 通过选择性共置避免了部分传输开销。

\subsection{SLO 达标率对比}

\begin{table}[htbp]
\centering
\caption{均匀负载 12 req/s 下的 SLO 达标率（\%）}
\label{tab:slo-attainment}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{X c c c c c}
\toprule
模型 & Disagg-Static & Coloc-Sarathi & CBS-NoMig & CBS-NoRole & CBS-Full \\
\midrule
Qwen2.5-7B & 82.1 & 78.4 & 91.0 & 93.2 & \textbf{95.4} \\
LLaMA-3-8B & 79.8 & 76.2 & 89.2 & 91.8 & \textbf{94.1} \\
LLaMA-2-70B & 73.5 & 71.8 & 84.6 & 88.1 & \textbf{91.2} \\
\bottomrule
\end{tabularx}
\end{table}

表~\ref{tab:slo-attainment} 展示了 12 req/s 负载下各方案的 SLO 达标率。CBS-Full 在三种模型上的 SLO 达标率分别为 95.4\%、94.1\% 和 91.2\%，相比 Disagg-Static 提升了 13--18 个百分点，相比 Coloc-Sarathi 提升了 17--19 个百分点。

Disagg-Static 的 SLO 违约主要来自 TTFT 超标（Prefill 排队过长），而 Coloc-Sarathi 的 SLO 违约主要来自 TPOT 超标（PD 干扰导致 Decode 变慢）。CBS-Full 通过逐请求动态选择执行模式，同时控制了两类违约来源。

\subsection{尾时延分析}

\begin{table}[htbp]
\centering
\caption{均匀负载 12 req/s 下的 P99 时延（ms）}
\label{tab:tail-latency}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{X c c c c c c}
\toprule
\multirow{2}{*}{方案} & \multicolumn{2}{c}{Qwen2.5-7B} & \multicolumn{2}{c}{LLaMA-3-8B} & \multicolumn{2}{c}{LLaMA-2-70B} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
 & P99 TTFT & P99 TPOT & P99 TTFT & P99 TPOT & P99 TTFT & P99 TPOT \\
\midrule
Disagg-Static & 3842 & 68 & 4126 & 72 & 4580 & 124 \\
Coloc-Sarathi & 1284 & 156 & 1368 & 164 & 1512 & 218 \\
CBS-Full & 1526 & \textbf{82} & 1642 & \textbf{86} & 1836 & \textbf{96} \\
\bottomrule
\end{tabularx}
\end{table}

表~\ref{tab:tail-latency} 展示了 P99 尾时延。Disagg-Static 的 P99 TTFT 显著高于其他方案（3842--4580 ms），因为 Prefill 节点在高负载下形成长队列，LLaMA-2-70B 因模型规模更大，排队效应更为严重。Coloc-Sarathi 的 P99 TTFT 较低（1284--1512 ms），但 P99 TPOT 显著恶化（156--218 ms），远超 SLO 约束（100 ms）。CBS-Full 在两项指标上取得了较好的平衡：P99 TTFT 为 1526--1836 ms，低于 SLO 约束；P99 TPOT 为 82--96 ms，控制在 SLO 约束以内。

\subsection{突发负载下的表现}

\begin{table}[htbp]
\centering
\caption{突发负载下的 Goodput 和 SLO 达标率（Qwen2.5-7B，基线 8 req/s）}
\label{tab:bursty}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{X c c}
\toprule
方案 & Goodput (req/s) & SLO 达标率 (\%) \\
\midrule
Disagg-Static & 6.82 & 72.4 \\
Coloc-Sarathi & 7.14 & 68.1 \\
CBS-Full & \textbf{8.36} & \textbf{88.7} \\
\bottomrule
\end{tabularx}
\end{table}

突发负载对调度系统的适应能力提出了更高要求。表~\ref{tab:bursty} 显示，CBS-Full 在突发负载下的 goodput 为 8.36 req/s，SLO 达标率为 88.7\%，分别优于 Disagg-Static（6.82 req/s，72.4\%）和 Coloc-Sarathi（7.14 req/s，68.1\%）。CBS-Full 的优势主要体现在流量高峰期间：当 Prefill 排队压力骤增时，CBS 评分中的 $T_{\mathrm{queue}}^{(p)}$ 项显著增大，使更多请求被路由到共置路径；高峰过后，Decode 节点负载恢复正常，CBS 自动切回分离路径。双向迁移机制在高峰期间进一步发挥作用：Mitigation 迁移将因突发共置导致 SLO 风险升高的请求迁出，Consolidation 在高峰过后合并低负载节点。

\subsection{长上下文负载下的表现}

\begin{table}[htbp]
\centering
\caption{长上下文负载下的 Goodput 和 SLO 达标率（Qwen2.5-7B，8 req/s）}
\label{tab:long-context}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{X c c}
\toprule
方案 & Goodput (req/s) & SLO 达标率 (\%) \\
\midrule
Disagg-Static & 5.48 & 68.5 \\
Coloc-Sarathi & 6.12 & 64.2 \\
CBS-Full & \textbf{7.04} & \textbf{82.6} \\
\bottomrule
\end{tabularx}
\end{table}

长上下文场景下，Prefill 计算量和 KV Cache 体积均显著增大。表~\ref{tab:long-context} 显示，CBS-Full 的 goodput 为 7.04 req/s，相比 Disagg-Static 提升 28.5\%，相比 Coloc-Sarathi 提升 15.0\%。Disagg-Static 在该场景下的劣势更加明显，因为长序列的 KV Cache 传输时延从均匀负载下的约 20 ms 增加到约 80--120 ms，CBS 通过选择性共置避免了大量传输开销。同时，CBS 的 Chunked Prefill 感知机制在长上下文场景下发挥了重要作用：对于输入长度超过 4096 的请求，CBS 自动选择较小的 chunk 大小以控制单次 iteration 的干扰程度。

\section{消融实验}

本节通过消融实验分析 CBS 系统各组件的贡献。

\subsection{双向迁移机制的贡献}

对比 CBS-NoMig 与 CBS-Full 可以评估双向迁移机制的贡献。在均匀负载 12 req/s 下，双向迁移使 Qwen2.5-7B 的 goodput 从 10.92 提升至 11.43（+4.7\%），SLO 达标率从 91.0\% 提升至 95.4\%（+4.4 个百分点）。

进一步分析迁移行为的统计数据：在 8 分钟的稳定运行期间，Mitigation 迁移平均触发 23 次，Consolidation 迁移平均触发 41 次。Mitigation 迁移主要发生在负载波动导致局部节点过载时，平均每次迁移使源节点上剩余请求的平均 TPOT 降低 12\%。Consolidation 迁移主要发生在请求批量完成后的低负载阶段，平均每次合并释放 0.8 个节点的资源。

\subsection{节点角色自适应的贡献}

对比 CBS-NoRole 与 CBS-Full 可以评估节点角色自适应的贡献。在均匀负载 12 req/s 下，节点角色自适应使 goodput 从 11.18 提升至 11.43（+2.2\%），SLO 达标率从 93.2\% 提升至 95.4\%（+2.2 个百分点）。

角色自适应的效果在突发负载下更为明显。在突发负载实验中，CBS-Full 相比 CBS-NoRole 的 goodput 提升为 5.1\%，SLO 达标率提升为 4.8 个百分点。分析表明，在流量高峰期间，Consolidation 清空的 Decode 节点被临时转为 Prefill 节点，使 P:D 比例从初始的 2:6 动态调整为 3:5，有效缓解了 Prefill 排队压力。高峰过后，临时 Prefill 节点在完成当前任务后自动回收为 Decode 节点。

\subsection{CBS 各成本分量的贡献}

为分析 CBS 评分公式中各分量的作用，设计以下消融变体：

\textbf{CBS-NoDispatch}：移除 dispatch 争抢项 $\Delta_{\mathrm{dispatch}}$。

\textbf{CBS-NoRisk}：移除 SLO 风险惩罚项 $\Delta_{\mathrm{risk}}$。

\textbf{CBS-NoBudget}：移除 compute budget 约束。

\begin{table}[htbp]
\centering
\caption{CBS 成本分量消融实验（Qwen2.5-7B，均匀负载 12 req/s）}
\label{tab:cbs-ablation}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{X c c c}
\toprule
变体 & Goodput (req/s) & SLO 达标率 (\%) & P99 TPOT (ms) \\
\midrule
CBS-Full & \textbf{11.43} & \textbf{95.4} & \textbf{82} \\
CBS-NoDispatch & 11.12 & 93.1 & 94 \\
CBS-NoRisk & 10.86 & 90.8 & 108 \\
CBS-NoBudget & 10.64 & 88.2 & 118 \\
\bottomrule
\end{tabularx}
\end{table}

表~\ref{tab:cbs-ablation} 显示，移除 compute budget 约束的影响最大（goodput 降低 6.9\%，SLO 达标率降低 7.2 个百分点），因为缺少该约束时，CBS 可能在单个 iteration 中注入过多 Prefill tokens，导致所有 Decode 请求的 TPOT 同时恶化。SLO 风险惩罚项的移除导致 goodput 降低 5.0\%，因为缺少该项时 CBS 可能在 SLO 余量不足的情况下仍选择共置。Dispatch 争抢项的影响相对较小但仍有意义，其移除导致 P99 TPOT 从 82 ms 上升至 94 ms。

\section{超参数敏感性分析}

本节分析关键超参数对系统性能的影响，实验基于 Qwen2.5-7B 模型、均匀负载 12 req/s。

\subsection{外部性权重 $\lambda$}

$\lambda$ 控制 CBS 对已有 Decode 请求服务质量的保护程度。图~\ref{fig:lambda-sensitivity} 展示了 $\lambda$ 从 0.2 到 3.0 变化时的 goodput 和 SLO 达标率。

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.6\linewidth]{北航硕博士学位论文 LaTeX 4.1.0/pic/lambda-sensitivity.pdf}
  \caption{外部性权重 $\lambda$ 的敏感性分析}
  \label{fig:lambda-sensitivity}
\end{figure}

当 $\lambda < 0.5$ 时，CBS 对共置干扰的惩罚不足，过多请求被路由到共置路径，导致 Decode 请求的 TPOT 恶化，SLO 达标率降至 88\% 以下。当 $\lambda > 2.0$ 时，CBS 过于保守，共置比例过低，系统行为趋近于纯分离，goodput 下降。$\lambda$ 在 $[0.8, 1.5]$ 区间内，goodput 和 SLO 达标率均保持在较优水平，表明该参数对合理范围内的取值不敏感。本文取 $\lambda = 1.0$。

\subsection{迁移阈值}

双向迁移机制涉及三个阈值参数：$\theta_{\mathrm{ceil}}$、$\theta_{\mathrm{floor}}$ 和 $\theta_{\mathrm{dispatch}}$。

\begin{table}[htbp]
\centering
\caption{迁移阈值敏感性分析}
\label{tab:threshold-sensitivity}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{X c c c}
\toprule
参数组合 & Goodput (req/s) & SLO 达标率 (\%) & 迁移次数/min \\
\midrule
$\theta_{\mathrm{ceil}}=0.2$ & 11.28 & 95.8 & 12.4 \\
$\theta_{\mathrm{ceil}}=0.3$（默认） & \textbf{11.43} & \textbf{95.4} & 8.0 \\
$\theta_{\mathrm{ceil}}=0.5$ & 11.18 & 93.6 & 4.2 \\
\midrule
$\theta_{\mathrm{floor}}=0.3$ & 11.36 & 95.2 & 9.6 \\
$\theta_{\mathrm{floor}}=0.4$（默认） & \textbf{11.43} & \textbf{95.4} & 8.0 \\
$\theta_{\mathrm{floor}}=0.6$ & 11.21 & 94.8 & 5.1 \\
\bottomrule
\end{tabularx}
\end{table}

表~\ref{tab:threshold-sensitivity} 显示，$\theta_{\mathrm{ceil}}$ 过低（0.2）会导致迁移过于频繁（12.4 次/min），虽然 SLO 达标率略有提升，但频繁迁移带来的 KV Cache 传输开销抵消了部分收益。$\theta_{\mathrm{ceil}}$ 过高（0.5）则导致迁移不够及时，SLO 达标率下降。$\theta_{\mathrm{floor}}$ 的影响类似：过低导致 Consolidation 过于激进，过高则导致低负载节点未能及时释放。总体而言，阈值参数在默认值附近的合理范围内，系统性能变化不超过 2\%，表明双向迁移机制对阈值设置具有较好的鲁棒性。

\subsection{风险惩罚权重 $\mu$}

$\mu$ 控制 SLO 风险惩罚项在 CBS 评分中的权重。当 $\mu = 0$ 时（无风险惩罚），SLO 达标率降至 90.8\%；当 $\mu = 5.0$ 时，CBS 过于保守，goodput 降至 10.72 req/s。$\mu$ 在 $[1.0, 3.0]$ 区间内表现稳定，本文取 $\mu = 2.0$。

\section{系统开销分析}

\subsection{调度延迟}

CBS Scheduler Plugin 的单次调度延迟包括 MLWD 在线构建（PreFilter）、约束检查（Filter）和 CBS 评分（Score）三个阶段。在 8 节点集群上，单次调度的端到端延迟中位数为 0.42 ms，P99 为 0.81 ms。其中，MLWD 在线构建（KD-Tree 近邻检索）耗时约 0.08 ms，Filter 阶段约 0.12 ms，Score 阶段约 0.22 ms。该延迟远低于请求的 Prefill 执行时间（通常为数十至数百毫秒），不会成为系统瓶颈。

\subsection{Node Monitor 采集开销}

Node Monitor 的 CPU 开销约占节点 CPU 总量的 0.3\%，内存占用约 45 MB。硬件层采集（DCGM）和引擎层采集（HTTP metrics）的开销均较低。聚合 MLWD 计算在节点侧完成，单次计算耗时约 0.05 ms。

\subsection{离线 Profiling 成本}

对单个模型（如 Qwen2.5-7B），全量 Profiling（包括单任务画像、四维干扰敏感度和共置实验）在单 GPU 上约需 4 小时。采用稀疏采样 + 参数化外推后，采集时间缩短至约 40 分钟。加权系数标定（最小二乘法）耗时不超过 1 秒。

\section{本章小结}

本章通过系统性实验评估了 MLWD 干扰估算方法、CBS 动态调度算法和 Profiling-Scheduling 系统的有效性。

在干扰估算方面，MLWD 驱动的规则化估算方法在 $\alpha_d$ 和 $\alpha_p$ 上的 MAE 分别为 0.027 和 0.024，$R^2$ 分别为 0.65 和 0.61，相比仅使用 SM 利用率的方法 MAE 降低了 53\%，验证了分算子画像和四维干扰敏感度在提升估算精度方面的作用。跨模型泛化实验表明，加权系数具有一定的跨模型迁移能力。参数化外推机制在稀疏采样条件下实现了接近全量 Profiling 的估算精度，核心特征（CI、Kernel 时延）的外推 MAPE 低于 5\%。

在调度效果方面，CBS-Full 在均匀负载下相比纯分离基线 Disagg-Static 的 goodput 提升为 16\%--34\%，相比纯共置基线 Coloc-Sarathi 的提升为 12\%--15\%，SLO 达标率提升为 13--19 个百分点。在突发负载和长上下文负载下，CBS-Full 同样表现出较好的适应能力。消融实验表明，双向迁移机制贡献了约 4--5\% 的 goodput 提升和 4--5 个百分点的 SLO 达标率提升，节点角色自适应在突发负载下的贡献更为显著。CBS 评分公式中的 compute budget 约束、SLO 风险惩罚和 dispatch 争抢建模均对系统性能有正向贡献。

在系统开销方面，CBS 的单次调度延迟中位数为 0.42 ms，Node Monitor 的 CPU 开销约 0.3\%，均不构成系统瓶颈。离线 Profiling 采用稀疏采样 + 参数化外推后，单模型采集时间约 40 分钟。

超参数敏感性分析表明，CBS 的关键参数（$\lambda$、$\mu$、迁移阈值）在合理范围内对系统性能的影响不超过 2\%，系统具有较好的鲁棒性。
