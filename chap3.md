\chapter{多层次工作负载描述符设计}

在 LLM 在线推理系统中，推理请求通常分为 Prefill 和 Decode 两个阶段。二者在资源需求和性能目标上存在明显差异：Prefill 阶段需要对输入上下文进行并行编码，通常具有较高的计算密度；Decode 阶段以逐 token 的自回归生成为主，对显存访问和 KV Cache 读写更为敏感。由于两个阶段的行为具有显著异构性，仅仅依赖于节点级的 SM 利用率、显存带宽利用率和显存占用率等粗粒度指标，难以准确描述不同请求之间的实际干扰关系。

同时，LLM 推理任务的性能退化不仅受节点整体资源使用情况影响，还与模型架构、推理框架、量化方式、并行策略以及关键算子的执行行为密切相关。即使两个任务表现出相近的硬件利用率，其底层计算模式也可能存在较大差异，从而在共置时产生不同程度的资源竞争。因此，如果要实现精细化的 PD 动态调度，就需要构建一种能够同时反映算子资源行为和请求实时状态的统一任务描述方法。

基于此，本章提出多层次工作负载描述符（Multi-Level Workload Descriptor, MLWD）。MLWD 从算子运行时画像和请求动态状态两个层次建立了统一描述，为后续的干扰预测与调度决策提供数据基础。通过显式刻画 Prefill 与 Decode 的阶段异构性，提高了任务描述的细粒度与判别能力，同时在保证在线查询效率的前提下提升描述符的表达能力与泛化能力。

\section{问题分析}

现有的 GPU 共置干扰建模工作主要将 SM 利用率、显存带宽利用率等硬件级指标作为干扰预测输入。这一建模方式隐含了一个前提，即相同的硬件负载水平对应相近的干扰行为。然而，在 LLM 推理场景下，不同模型架构和推理框架的资源占用模式并不一致，上述前提并不总是成立。本节通过两组情形分析 LLM 推理场景下的共置干扰现象。

\subsection{同 SM 利用率下的干扰差异分析}

研究指出，GPU 内部共享资源可以进一步分解为四个竞争维度：线程块调度器、计算单元、L1/L2 缓存和显存带宽\upcite{elvinger2025gpuutildeeper}。对于不同的内核，即使 SM 利用率接近，它们在上述四个维度上的资源占用分布也可能明显不同。一类内核可能主要占用计算单元而缓存压力较小；另一类内核对于计算的需求不高，但会频繁地竞争 L2 缓存和显存带宽。当这两类任务分别与同一任务共置时，产生的干扰情况并不相同。

\begin{figure}[htbp]
    \centering

    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth]{北航硕博士学位论文 LaTeX 4.1.0/pic/L1 缓存容量溢出引发的共置干扰激增.png}
        \caption{L1 缓存容量压力}
        \label{fig:l1-cache-overflow}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth]{北航硕博士学位论文 LaTeX 4.1.0/pic/L2 缓存容量压力下的共置干扰加剧.png}
        \caption{L2 缓存容量压力}
        \label{fig:l2-cache}
    \end{subfigure}
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth]{北航硕博士学位论文 LaTeX 4.1.0/pic/显存带宽竞争下的共置 Kernel 性能退化.png}
        \caption{显存带宽竞争}
        \label{fig:memory-bandwidth}
    \end{subfigure}

    \caption{不同硬件资源瓶颈下的共置 Kernel 干扰现象}
    \label{fig:colocation-interference}
\end{figure}

Weng进一步通过实验表明\upcite{zhao2025mlpredictable}，如果使用整体 SM 利用率等粗粒度硬件指标进行干扰估算，容易产生较大的偏差；相比之下，基于内核级细粒度特征的方法具有更高的精度。因此，要建立准确的干扰估算方法，需要对内核级的资源占用画像进行分析，而不能局限在系统层面的聚合指标。

\subsection{模型架构差异对 Kernel 资源占用模式的影响}

LLM 中注意力机制的实现方式直接影响了 Decode 阶段的 GPU 资源占用模式。目前当前常见的多头注意力（Multi-Head Attention, MHA）、分组查询注意力（Grouped-Query Attention, GQA）和多查询注意力（Multi-Query Attention, MQA）在 KV Cache 大小和显存访问模式上存在明显差异。

以 GQA 和 MHA 的对比为例，GQA 通过多个查询头共享同一组 KV 头，压缩了 KV Cache 大小。在 Decode 阶段，GQA 模型的注意力 Kernel 需要从显存加载的 KV 数据量少于 MHA，因此显存带宽压力较低，L2 缓存命中率相对更高；而 MHA 模型的注意力 Kernel 需要为每个注意力头独立加载完整的 KV 对，显存带宽的占用更高，L2 缓存刷新概率也更大。GQA 通过让多个 query heads 共享一组 key/value heads，减少了 Decode 阶段需要加载的 KV 数据量，从而降低了显存带宽压力。

上述分析表明，SM 利用率和显存带宽利用率等硬件级聚合指标难以完整反映工作负载的计算模式差异。相同 SM 利用率的任务，在计算单元、缓存和显存带宽等子资源上的占用分布可能并不相同；不同的注意力机制又会进一步影响 Decode 阶段的显存访问频率和缓存友好性。

\subsection{干扰估算的两个必要维度：资源竞争强度与干扰敏感度}

综合上述分析，准确的共置干扰建模需要同时刻画两个相互独立的维度。第一个维度是“资源竞争强度”（aggressor strength），即任务在各资源竞争维度上的实际消耗情况，也就是该任务作为干扰源时对共置任务施加的压力。这一维度可由算子级计算强度（CI）和 L2 缓存命中率等指标描述。例如，CI 较高的任务通常会在计算单元维度上施加更强压力，而 L2 命中率较低的任务则可能在显存带宽维度上施加更强压力。第二个维度是“干扰敏感度”（victim sensitivity），即任务在各资源竞争维度上对外部干扰的敏感程度，可以使用四维干扰敏感度向量 $(\sigma_{\text{bs}}, \sigma_{\mathrm{cu}}, \sigma_{\text{l2}}, \sigma_{\text{bw}})$ 表示。

如果只是考虑资源竞争强度而忽略了干扰敏感度，估算方法将无法区分"两项高 CI 任务共置"和"一项高 CI 任务与一项对计算单元竞争不敏感的任务共置"这两种情形。尽管前者和后者的资源竞争强度可能相近，但由于受到干扰的任务负载干扰敏感性不同，实际干扰程度并不相同。反之，如果只考虑干扰敏感度而忽略了资源竞争强度，也无法判断干扰源实际造成的资源压力。因此，干扰系数的估算应该同时基于这两个维度的特征。

此外，干扰估算还需要考虑请求的实时动态状态，包括当前阶段、批大小、序列长度以及对未来行为的预测，因为算子画像的实际取值会随着这些动态参数变化。MLWD 的两层结构正是基于这一需求设计：第一层编码资源竞争强度和干扰敏感度等算子级特征，第二层编码请求的动态状态。

\section{MLWD 向量的两层结构设计}

MLWD 是本文提出的多层次工作负载描述符，其目标是将影响 GPU 共置干扰的关键因素编码为结构化向量，供 CBS 算法中的干扰系数估算模块使用。根据 3.1 节的分析，MLWD 向量包括了两类必要特征，即算子运行时画像和请求动态状态，前者用于刻画资源竞争强度与干扰敏感度，后者则用于描述算子画像在具体请求条件下的实际取值。

完整的 MLWD 向量可形式化表示为：
\begin{equation}
\mathbf{W}=[\mathbf{f}^{(1)}_{\text{operator}},\ \mathbf{f}^{(2)}_{\text{request}}]
\end{equation}

其中，两层向量分别对应算子运行时特征向量和请求级动态特征向量。以下分别说明各层特征的设计动机、字段定义和编码方式。

\subsection{算子运行时画像层}

算子运行时画像层 $\mathbf{f}^{(1)}_{\text{operator}}$ 用于编码通过离线 Profiling 获得的 Kernel 级硬件性能指标。需要在实际运行推理任务的过程中，借助 GPU 性能分析工具采集相关特征，是 MLWD 向量的核心部分。

这一层描述的是推理任务在 GPU 上执行时的资源占用画像。在相同部署配置下，其取值会随 $(b, s)$ 平滑变化，但无法仅由静态配置参数解析得到。这是因为推理框架实现细节、CUDA Kernel 优化策略和硬件微架构特性等因素会引入难以由静态参数直接表达的非线性影响。

具体字段定义如下：
\begin{table}[htbp]
\centering
\caption{算子运行时画像层字段定义}
\label{tab:operator_profile_fields}
\small
\renewcommand{\arraystretch}{1.22}
\setlength{\tabcolsep}{6pt}
\begin{tabularx}{0.98\linewidth}{l c c X}
\toprule
字段名 & 符号 & 采集工具 & 说明 \\
\midrule
Attention 计算强度 & $\mathrm{CI}_{\mathrm{attn}}$ & Nsight Compute & FLOP/Byte，反映计算与访存的相对强度。 \\
FFN 计算强度 & $\mathrm{CI}_{\mathrm{ffn}}$ & Nsight Compute & FLOP/Byte。 \\
Attention L2 命中率 & $\mathrm{L2}_{\mathrm{attn}}$ & Nsight Compute & L2 Cache Hit Rate。 \\
FFN L2 命中率 & $\mathrm{L2}_{\mathrm{ffn}}$ & Nsight Compute & L2 Cache Hit Rate。 \\
Attention 平均时延 & $\bar{t}_{\mathrm{attn}}$ & Neutrino / Nsight & 单次 Kernel 执行的平均耗时（$\mu$s）。 \\
FFN 平均时延 & $\bar{t}_{\mathrm{ffn}}$ & Neutrino / Nsight & 单次 Kernel 执行的平均耗时（$\mu$s）。 \\
Launch 间隔 & $\bar{g}_{\mathrm{launch}}$ & Nsight Systems & 相邻 Kernel Launch 的平均间隔（$\mu$s）。 \\
干扰敏感度--BS & $\sigma_{\mathrm{bs}}$ & 合成压力核 & Block Scheduler 维度的干扰敏感度。 \\
干扰敏感度--CU & $\sigma_{\mathrm{cu}}$ & 合成压力核 & 计算单元维度的干扰敏感度。 \\
干扰敏感度--L2 & $\sigma_{\mathrm{l2}}$ & 合成压力核 & L2 缓存维度的干扰敏感度。 \\
干扰敏感度--BW & $\sigma_{\mathrm{bw}}$ & 合成压力核 & 显存带宽维度的干扰敏感度。 \\
Attention 时间占比 & $r_{\mathrm{attn}}$ & Nsight Systems & $r_{\mathrm{attn}}=\bar{t}_{\mathrm{attn}}\cdot L / T_{\mathrm{total}}$。 \\
FFN 时间占比 & $r_{\mathrm{ffn}}$ & Nsight Systems & $r_{\mathrm{ffn}}=\bar{t}_{\mathrm{ffn}}\cdot L / T_{\mathrm{total}}$。 \\
平均 IPC & $\overline{\mathrm{IPC}}$ & Nsight Compute & 每周期指令数，用于反映 pipeline 饱和度。 \\
\bottomrule
\end{tabularx}
\end{table}

本层的 14 个特征可以分为三组，分别用于描述资源竞争强度、干扰敏感度和执行模式。计算强度（Compute Intensity, CI）定义为单位数据传输量对应的浮点运算次数，即 $\operatorname{CI} = \text{FLOP} / \text{Byte}$。在共置干扰预测中，CI 的作用在于区分资源竞争类型：两个计算密集型 Kernel 共置时主要竞争 SM 计算单元，而计算密集型 Kernel 与访存密集型 Kernel 共置时则主要竞争不同资源维度，其干扰模式明显不同。L2 命中率则从缓存角度反映资源竞争强度。例如，GQA 模型的 Decode Attention Kernel 由于 KV Cache 体积较小，L2 命中率通常高于 MHA 模型。当两个高 L2 命中率的 Kernel 共置时，L2 Cache 容量竞争可能加剧，进而引发 Cache Thrashing，造成非线性的性能退化。因此，分算子的 CI 和 L2 命中率共同构成了对资源竞争强度的细粒度刻画。

干扰敏感度特征组由四维干扰敏感度向量 $(\sigma_{\text{bs}}, \sigma_{\mathrm{cu}}, \sigma_{\text{l2}}, \sigma_{\text{bw}})$ 构成，分别量化工作负载对线程块调度器（Block Scheduler）、计算单元（Compute Units）、L2 缓存和显存带宽四个竞争维度的敏感程度。其采集方式为：在离线 Profiling 过程中，对每个 $(\text{model}, \text{framework}, \text{config}, b, s)$ 组合分别注入四类仅占用单一资源维度的合成干扰 Kernel，并测量目标任务在对应压力下的性能退化比例，作为敏感度取值。通过这一向量，MLWD 不仅描述任务的资源消耗情况，也刻画了任务对资源竞争的脆弱程度，从而为干扰预测提供重要的判别信息。
\begin{figure}[htpb]
    \centering
    \includegraphics[width=1\linewidth]{北航硕博士学位论文 LaTeX 4.1.0/pic/compare_heatmap.png}
    \caption{不同模型的GPU资源干扰敏感度对比}
    \label{fig:heatmap}
\end{figure}

执行模式特征组用于进一步描述任务在时间维度上的执行行为。这是因为共置干扰不仅与单个 Kernel 的资源消耗有关，也与多个 Kernel 的执行顺序及时间重叠模式密切相关。执行模式包含以下特征：Kernel Launch 间隔 $\bar{g}_{\text{launch}}$ 用于反映框架调度粒度和调度效率；Attention 时间占比 $r_{\mathrm{attn}}$ 和 FFN 时间占比 $r_{\mathrm{ffn}}$ 描述两类主要算子的时间分布；平均 IPC $\overline{\text{IPC}}$ 则反映 pipeline 饱和度和 warp scheduler 的竞争程度。即使一个 compute-bound Kernel 与一个 memory-bound Kernel 共置，若前者 IPC 已接近硬件上限，后者仍可能因 warp scheduler 竞争而出现明显性能下降。

本层所有特征均为连续型变量，统一采用 Min-Max 归一化编码：
\[
\begin{aligned}
\mathbf{f}^{(1)}_{\mathrm{operator}}
= \big[
&\hat{\mathrm{CI}}_{\mathrm{attn}},\,
\hat{\mathrm{CI}}_{\mathrm{ffn}},\,
\hat{\mathrm{L2}}_{\mathrm{attn}},\,
\hat{\mathrm{L2}}_{\mathrm{ffn}},\,
\bar{\hat{t}}_{\mathrm{attn}},\,
\bar{\hat{t}}_{\mathrm{ffn}},\,
\bar{\hat{g}}_{\mathrm{launch}},\,
\hat{\sigma}_{\mathrm{bs}}, \\
&\hat{\sigma}_{\mathrm{cu}},\,
\hat{\sigma}_{\mathrm{l2}},\,
\hat{\sigma}_{\mathrm{bw}},\,
\hat{r}_{\mathrm{attn}},\,
\hat{r}_{\mathrm{ffn}},\,
\overline{\mathrm{IPC}}
\big]
\in \mathbb{R}^{14}.
\end{aligned}
\]
\subsection{请求级动态特征层}

请求级动态特征层 $\mathbf{f}^{(2)}_{\text{request}}$ 用于编码推理请求的实时状态。与第一层可在离线阶段预先确定不同，本层特征需在请求到达时实时获取，是 CBS 算法进行动态调度决策的重要输入。

具体字段定义如下：

\begin{table}[htbp]
\centering
\caption{请求级动态特征定义}
\label{tab:req_dynamic_features}
\small
\renewcommand{\arraystretch}{1.22}
\begin{tabularx}{\linewidth}{l c c X}
\toprule
字段名 & 符号 & 采集方式 & 说明 \\
\midrule
当前阶段 & $\phi$ & 框架 API & Prefill 为 0，Decode 为 1。 \\
当前批处理大小 & $b$ & 框架 API & 当前迭代的实际 batch size。 \\
当前序列长度 & $s$ & 框架 API & Prefill 为输入长度，Decode 为已生成长度。 \\
预估剩余输出长度 & $\hat{o}_{\mathrm{remain}}$ & 历史分布 & 基于历史输出长度分布估计剩余输出长度。 \\
\bottomrule
\end{tabularx}
\end{table}

当前阶段 $\phi$ 是最基本的动态特征。Prefill 和 Decode 在资源消耗模式上存在明显差异：Prefill 更偏向计算密集型，对 SM 计算单元需求更高；Decode 更偏向访存密集型，对显存带宽更为敏感。CBS 算法的核心决策之一，即是否将新请求的 Prefill 与已有请求的 Decode 共置，正是建立在这一阶段差异之上。

当前批处理大小 $b$ 和序列长度 $s$ 共同决定第一层算子画像特征的实际取值。离线 Profiling 阶段采集的算子画像是在特定 $(b, s)$ 组合下得到的，而在线推理中的实际 batch size 和序列长度不一定与离线采样点完全一致。因此，本层记录实际的 $b$ 和 $s$，供后续系统通过插值或最近邻检索，从离线 MLWD 库中获取最接近的第一层特征。

预估剩余输出长度 $\hat{o}_{\mathrm{remain}}$ 是本层中的关键特征。请求的未来行为会直接影响调度决策：一个即将结束的 Decode 请求与一个刚开始执行的 Decode 请求，对共置带来的额外干扰具有不同的容忍度。$\hat{o}_{\mathrm{remain}}$ 可根据历史输出长度分布进行估计，并直接影响 CBS 中外部性代价 $\Delta_{\mathrm{ext}}$ 的计算精度。

预估剩余执行时间
\begin{equation}
\hat{T}_{\mathrm{remain}}=\hat{o}_{\mathrm{remain}}\cdot T_{\mathrm{decode\_step}}^{(0)}
\end{equation}
可以由 $\hat{o}_{\mathrm{remain}}$ 与离线基线时延查找表计算得到，在 CBS 的外部性代价计算中实时快速推导。

在编码方式上，$\phi$ 为二值特征，$b$、$s$ 和 $\hat{o}_{\mathrm{remain}}$ 采用归一化处理：
\begin{equation}
\mathbf{f}^{(2)}_{\text{request}} = [\phi,\ \hat{b},\ \hat{s},\ \hat{o}_{\mathrm{remain}}] \in \mathbb{R}^{4}
\end{equation}

\subsection{MLWD 完整向量与分层构建策略}

综合两层特征，可得完整的 MLWD 向量：
\begin{equation}
\mathbf{W} = [\mathbf{f}^{(1)}_{\text{operator}},\ \mathbf{f}^{(2)}_{\text{request}}] \in \mathbb{R}^{18}
\end{equation}

其中，$|\mathbf{f}^{(1)}| = 14$，$|\mathbf{f}^{(2)}| = 4$。

MLWD 的两层结构如图\ref{fig:mlwd_struct}所示，按时间稳定性由强到弱排列：第一层属于半静态特征，依赖离线 Profiling 结果，可在部署配置确定后通过查表或参数化模型推理获取；第二层为动态特征，需要在请求到达时实时采集。
\begin{figure}[htp]
    \centering
    \includegraphics[width=1\linewidth]{北航硕博士学位论文 LaTeX 4.1.0/pic/mlwd-struct.jpg}
    \caption{MLWD 两层向量结构图}
    \label{fig:mlwd_struct}
\end{figure}

这种分层设计将在线阶段的 MLWD 构建开销控制在较低水平。在线时仅需获取第二层的 4 个动态特征，即 $\phi$、$b$、$s$ 和 $\hat{o}_{\mathrm{remain}}$，再根据 $(b, s)$ 从预先构建的 MLWD 库中检索，或通过参数化模型推理得到第一层特征即可。

\section{MLWD 离线采集流程}

MLWD 向量中的算子运行时画像层需要通过离线 Profiling 获取。本节说明离线采集的技术路线、实验矩阵设计和数据存储方式。离线采集的目标是：针对目标部署环境中的各类可行模型、框架和配置组合，在不同 $(b, s)$ 条件下测量 Kernel 级硬件性能指标，从而建立离线 MLWD 库。

\subsection{基于可编程探针的 Attention/FFN Kernel 插桩方案}

可编程探针技术是一种基于 GPU 汇编级插桩的非侵入式性能探测方法，其基本思想是在 GPU Kernel 的 SASS（Streaming ASSembler）指令中插入轻量级探针，直接在硬件执行层采集性能数据，而无需修改应用源代码或重新编译。与传统基于 CUPTI 的 Profiling 工具相比，可编程探针能够以较低代价对特定 Kernel 进行选择性插桩，并尽量减小对被测 Kernel 执行行为的扰动。

在本文的 MLWD 采集流程中，可编程探针主要用于采集 Attention Kernel 和 FFN Kernel 的执行时延 $\bar{t}_{\mathrm{attn}}$ 与 $\bar{t}_{\mathrm{ffn}}$，具体插桩方案如下。

步骤 1：Kernel 识别。首先使用 Nsight Systems 对目标推理框架执行完整时间线追踪，以识别 Attention 和 FFN 相关 Kernel 的函数签名。具体做法是，通过 Nsight Systems 的 \texttt{nsys profile} 命令对推理基准测试程序进行 CUDA 和 NVTX 级别的追踪，生成完整的 Kernel Timeline。随后，使用 \texttt{nsys stats} 对追踪结果进行统计分析，按 Kernel 名称过滤出与 Attention（如 Flash Attention、FMHA）和 FFN（如 GEMM、Linear、MLP）相关的 Kernel 函数签名。

步骤 2：SASS 级插桩。根据已识别的 Kernel 函数签名配置探针工具，采集单次 Kernel 时延和 Warp Stall 分布。探针配置中需指定目标 Kernel 的名称匹配模式（分别标记为 Attention 类和 FFN 类），并设置采集指标为 Kernel 级时延和 Warp 级停顿原因分布，采样率设为 1.0 以确保完整覆盖。

步骤 3：执行插桩采集。运行插桩后的推理任务，在稳定负载下重复多轮实验（本文取 100 轮迭代），统计 Attention 和 FFN Kernel 的平均时延及停顿原因分布。实验结束后，从结果中提取 $\bar{t}_{\mathrm{attn}}$ 和 $\bar{t}_{\mathrm{ffn}}$。

\subsection{基于 Nsight Compute 的硬件 Counter 采集方案}

与可编程探针主要用于获取细粒度时延和定位执行瓶颈不同，Nsight Compute 主要用于采集 Kernel 级硬件 Counter，以获取算子运行时画像层中的计算强度、L2 命中率等关键指标。

对于每个 $(\text{model}, \text{framework}, \text{config}, b, s, \phi)$ 组合，本文分别对 Prefill 和 Decode 两个阶段进行独立 Profiling。设目标 Kernel 为 $k$，其 Profiling 结果记为：
\begin{equation}
m(k) = [\operatorname{FLOPs}(k),\ \operatorname{Bytes}(k),\ \operatorname{L2Hit}(k)]
\end{equation}

则该 Kernel 的计算强度定义为：
\begin{equation}
\operatorname{CI}(k) = \frac{\operatorname{FLOPs}(k)}{\operatorname{Bytes}(k)}
\end{equation}

对属于 Attention 类和 FFN 类的 Kernel 集合分别求均值，即可得到 $\text{CI}_{\mathrm{attn}}$、$\text{CI}_{\mathrm{ffn}}$、$\text{L2}_{\mathrm{attn}}$ 和 $\text{L2}_{\mathrm{ffn}}$。随后，对采集结果进行后处理，将同类算子的指标聚合为对应配置下的算子画像。

\subsection{四维干扰敏感度的采集方案}

如前所述，资源消耗量与干扰敏感度并不是同一概念。为了采集 MLWD 第一层中的四维干扰敏感度向量 $(\sigma_{\text{bs}}, \sigma_{\mathrm{cu}}, \sigma_{\text{l2}}, \sigma_{\text{bw}})$，本文设计了基于合成压力核（Synthetic Stress Kernel）的采集方案。

对于每个资源竞争维度，构造一个主要占用该维度资源的合成 Kernel，并将其与目标推理任务共置运行，目标任务在该条件下的性能退化比例即定义为对应维度的干扰敏感度。四类合成压力核的设计如下：

\begin{table}[htbp]
\centering
\caption{四维干扰敏感度采集的合成压力核设计}
\label{tab:synthetic_stressors}
\small
\renewcommand{\arraystretch}{1.2}
\begin{tabularx}{0.98\linewidth}{l X}
\toprule
维度 & 合成压力核设计 \\
\midrule
Block Scheduler 干扰（$\sigma_{\mathrm{bs}}$）
& 通过大量轻量级 Kernel 的高频 Launch 占满 Block Scheduler 队列，同时尽量避免额外的计算与显存资源消耗。 \\

计算单元干扰（$\sigma_{\mathrm{cu}}$）
& 采用纯计算型 Kernel 执行密集浮点运算，不访问全局显存，并尽量减小 L2 缓存压力。 \\

L2 缓存干扰（$\sigma_{\mathrm{l2}}$）
& 通过特定访问模式反复读写接近 L2 容量范围内的数据，制造 cache thrashing，同时控制计算量与显存带宽占用。 \\

显存带宽干扰（$\sigma_{\mathrm{bw}}$）
& 采用大块连续显存读写的 Kernel 占满显存带宽，同时保持较低计算量，并降低 L2 缓存命中率。 \\
\bottomrule
\end{tabularx}
\end{table}
对于实验矩阵 $\mathcal{M}$ 中的每个 $(\text{model}, \text{framework}, \text{config}, b, s, \phi)$ 组合，分别与四类压力核共置运行，记录目标任务在各维度压力下的性能退化比例：
\begin{equation}
\sigma_{\text{dim}}(\text{model}, \text{framework}, \text{config}, b, s, \phi) = \frac{T_{\text{stressed}}^{(\text{dim})} - T^{(0)}}{T^{(0)}}
\end{equation}

其中，$\text{dim} \in \{\text{bs}, \text{cu}, \text{l2}, \text{bw}\}$，$T^{(0)}$ 表示无干扰基线时延，$T_{\text{stressed}}^{(\text{dim})}$ 表示在对应维度压力下的实际时延。

\subsection{基于参数化算子模型的 MLWD 外推}

MLWD 第一层依赖于离散 $(b, s)$ 网格上的实际 Profiling数据，在线阶段再通过近邻检索或插值方法获取对应的特征。然而，随着模型、框架、量化方案和并行策略组合数量的增加，离散网格会带来显著的采集成本。此外，对于随 $(b, s)$ 非线性变化的算子性能指标，如 L2 命中率，简单的线性插值可能引入较大误差。

为此，本文对 MLWD 第一层中的关键指标建立轻量级参数化回归模型。具体而言，分别针对 Attention 和 FFN 两类算子，训练以 $(b, s)$ 及部署配置参数为输入、以算子画像指标（CI、L2 命中率、时延等）为输出的分段线性回归模型。

采用该方法可以显著减少离线采样点数量。参数化模型只需在稀疏网格上采集训练数据，即可对整个 $(b, s)$ 连续空间进行预测，从而降低离线采集成本。同时，能够为未采样的 $(b, s)$ 组合提供更准确的预测。与直接线性插值相比，回归模型更容易刻画算子性能指标随 $(b, s)$ 变化的非线性趋势。当部署新模型时，只需在少量 $(b, s)$ 点上补充采样并微调模型参数，即可生成较完整的第一层特征，而不必重新执行全量 Profiling。

\subsection{共置干扰实验矩阵设计}

由于算子的运行时画像与模型结构、推理框架、量化方式、并行策略以及输入规模等因素密切相关，离线采集需要构造系统性的实验矩阵。本文定义离线实验矩阵为：
\begin{equation}
\mathcal{M} = \{(\text{model}, \text{framework}, \text{config}, b, s, \phi)\}
\end{equation}

其中，$\text{model}$ 包括 LLaMA、Qwen 等主流开源模型，$\text{framework}$ 为 vLLM 推理引擎，$\text{config}$ 包括不同量化方案（FP16、INT8、INT4）和并行策略（TP=1,4,8）。输入规模方面，batch size $b$ 取值为 $\{1, 4, 16\}$，序列长度 $s$ 取值为 $\{32, 64, 128, 512, 2048\}$，阶段 $\phi \in \{\text{prefill}, \text{decode}\}$。实验矩阵中的 model 维度意味着需要为每个部署的模型独立建立 MLWD 画像库，因为不同模型架构（如 GQA 与 MHA）的算子资源占用模式存在显著差异。共置干扰实验在每个模型内部独立进行，即在同一 vLLM 实例内测量 Prefill 与 Decode 的相互干扰。

为了控制离线采集成本，本文对高频区域采用密集采样，对低频区域采用稀疏采样。短上下文和中等 batch size 是在线服务中的主要工作区间，因此对 $s \leq 1024, b \leq 16$ 的组合进行更细粒度采样；对于长上下文和极端 batch 组合，仅保留具有代表性的采样点，在线阶段再通过插值进行近似估计。

\subsection{MLWD 库的存储结构与索引方式}

离线采集完成后，需要将 MLWD 算子画像存储到支持在线快速检索的数据库中。本文采用“主键索引 + 多维近邻检索”的方式组织 MLWD 库。部署配置参数，即模型架构和推理框架配置，作为离散主键索引：
\begin{equation}
\operatorname{Key} = (\mathrm{model}, \mathrm{framework}, \mathrm{config})
\end{equation}

对于与请求输入规模相关的连续特征 $b$ 和 $s$，则采用多维近邻检索，以支持在线插值查询。在线请求到达时，调度器首先根据部署配置参数定位到对应子库，再根据实际的 $b$ 和 $s$ 在子库中执行近邻检索，找到最匹配的算子画像特征，供 CBS 算法使用。

静态库中的每条记录包含三部分内容：MLWD 第一层算子画像特征向量、采样时的统计量（均值和标准差）以及元数据，如采样时间、GPU 型号、驱动版本和 CUDA 版本等。当部署环境发生变化时，可据此判断历史画像是否仍然适用。

\section{基于 MLWD 的规则化干扰系数估算方法}

前两节完成了 MLWD 的结构设计和离线采集方案。MLWD 为 CBS 调度算法提供了共置干扰系数 $\alpha_p$ 和 $\alpha_d$ 的估算能力。本文采用规则驱动的估算方式，其核心思想是：共置干扰的程度取决于干扰源在各资源维度上的竞争强度与受干扰方在对应维度上的敏感程度之间的耦合关系。本文进一步利用 MLWD 中已编码的资源竞争强度和干扰敏感度特征，通过加权映射规则直接估算干扰系数。

\subsection{干扰系数的定义}

在 PD 分离调度场景中，Decode 节点上正在运行若干请求的 Decode 阶段，调度器需要决定是否将新到达请求 $r$ 的 Prefill 阶段放置到该节点上执行。由于 vLLM 等推理引擎采用 continuous batching 机制，新请求的 Prefill 与已有请求的 Decode 会在同一个推理迭代中被合并处理，共享 GPU 的计算单元、缓存和显存带宽等资源，从而产生干扰。

需要注意的是，在典型的 PD 分离部署中，每个 GPU 节点运行单个模型的单个 vLLM 实例。因此，共置干扰发生在同一模型的不同请求之间——新请求 $r$ 和已有请求 $u$ 属于同一模型，但处于不同的推理阶段（$r$ 处于 Prefill，$u$ 处于 Decode），且具有不同的 $(b, s)$ 参数。二者的 MLWD 第一层特征虽然来自同一模型的离线画像库，但由于阶段和输入规模不同，其算子画像取值存在显著差异。

定义两个干扰系数：
\[
\alpha_p(r,u)=\frac{T_{\mathrm{prefill}}^{\mathrm{coloc}}(r)}
{T_{\mathrm{prefill}}^{(0)}(r)}-1,
\qquad
\alpha_d(u,r)=\frac{T_{\mathrm{decode\_step}}^{\mathrm{coloc}}(u)}
{T_{\mathrm{decode\_step}}^{(0)}(u)}-1.
\]

其中，$T^{(0)}$ 表示无干扰条件下的基线时延（纯 Decode 或纯 Prefill 运行），$T^{\text{coloc}}$ 表示 Prefill 与 Decode 在同一推理迭代中混合执行时的实际时延。$\alpha_p \geq 0$ 表示 Prefill 因共置产生的时延膨胀比例，$\alpha_d \geq 0$ 表示 Decode 因共置产生的时延膨胀比例。

干扰系数的标定数据通过离线共置实验获得。对于每个模型，在单个 vLLM 实例内，首先测量纯 Decode 负载下的基线时延，然后向同一实例注入不同规模的 Prefill 请求，测量 Decode 时延的退化程度。通过遍历不同的 $(b_{\mathrm{decode}}, b_{\mathrm{prefill}}, s_{\mathrm{prefill}})$ 组合，获得覆盖实验矩阵的标定数据集。

\subsection{基于资源竞争强度与干扰敏感度的加权映射规则}

根据 3.1 节的分析，共置干扰可分解为四个资源竞争维度：线程块调度器（BS）、计算单元（CU）、L2 缓存（L2）和显存带宽（BW）。干扰系数的估算基于以下物理直觉：Prefill 请求 $r$ 对 Decode 请求 $u$ 造成的干扰程度，等于 $r$ 在 Prefill 阶段各资源维度上施加的竞争压力与 $u$ 在 Decode 阶段对应维度上的敏感程度的加权耦合。

首先，从请求 $r$ 在 Prefill 阶段对应的 MLWD 第一层特征中提取四维资源竞争强度向量 $\mathbf{A}(r) = (A_{\text{bs}}, A_{\mathrm{cu}}, A_{\text{l2}}, A_{\text{bw}})$。各分量的映射规则如下：

$A_{\text{bs}}(r)$：由 Kernel Launch 频率 $1/\bar{g}_{\text{launch}}$ 归一化得到，反映任务对线程块调度器的占用压力；

$A_{\mathrm{cu}}(r)$：由分算子计算强度 $\operatorname{CI}_{\mathrm{attn}}$ 和 $\operatorname{CI}_{\mathrm{ffn}}$ 的加权平均得到，权重为对应算子的时间占比 $r_{\mathrm{attn}}$ 和 $r_{\mathrm{ffn}}$；

$A_{\text{l2}}(r)$：由 $(1 - \text{L2}_{\mathrm{attn}})$ 和 $(1 - \text{L2}_{\mathrm{ffn}})$ 的加权平均得到（L2 命中率越低，对 L2 缓存的竞争压力越大）；

$A_{\text{bw}}(r)$：由分算子计算强度的倒数加权得到（CI 越低，任务越偏向访存密集，对显存带宽的竞争压力越大）。

然后，干扰系数通过以下加权映射规则进行估算：
\begin{equation}
\hat{\alpha}_d(u, r) = \sum_{\text{dim} \in \{\text{bs}, \text{cu}, \text{l2}, \text{bw}\}} w_{\text{dim}} \cdot \sigma_{\text{dim}}^{(u)} \cdot A_{\text{dim}}^{(r)} + w_{\text{ipc}} \cdot \overline{\text{IPC}}^{(r)} \cdot \sigma_{\mathrm{cu}}^{(u)} + w_{\text{overlap}} \cdot \Omega(r, u)
\end{equation}
\begin{equation}
\hat{\alpha}_p(r, u) = \sum_{\text{dim} \in \{\text{bs}, \text{cu}, \text{l2}, \text{bw}\}} w_{\text{dim}} \cdot \sigma_{\text{dim}}^{(r)} \cdot A_{\text{dim}}^{(u)} + w_{\text{ipc}} \cdot \overline{\text{IPC}}^{(u)} \cdot \sigma_{\mathrm{cu}}^{(r)} + w_{\text{overlap}} \cdot \Omega(u, r)
\end{equation}

其中：
$w_{\text{dim}}$（$\text{dim} \in \{\text{bs}, \text{cu}, \text{l2}, \text{bw}\}$）为各资源维度的加权系数，通过离线标定确定；

$\sigma_{\text{dim}}^{(u)}$ 为请求 $u$ 在 Decode 阶段的四维干扰敏感度（来自 MLWD 第一层，按 $u$ 当前的 $(b, s)$ 从离线库中检索）；

$A_{\text{dim}}^{(r)}$ 为请求 $r$ 在 Prefill 阶段的四维资源竞争强度（由 $r$ 的 $(b, s)$ 对应的 MLWD 第一层特征映射得到）；

$w_{\text{ipc}} \cdot \overline{\text{IPC}}^{(r)} \cdot \sigma_{\mathrm{cu}}^{(u)}$ 为 IPC 修正项，用于捕捉 pipeline 饱和度对 warp scheduler 竞争的额外影响；

$\Omega(r, u)$ 为算子时间交错因子，定义为 $\Omega(r, u) = \min(r_{\mathrm{attn}}^{(r)} + r_{\mathrm{ffn}}^{(r)},\ r_{\mathrm{attn}}^{(u)} + r_{\mathrm{ffn}}^{(u)})$，用于反映两个任务在时间维度上的 Kernel 重叠程度——当两个任务的活跃 Kernel 时间占比都较高时，时间重叠概率更大，干扰更强。

该规则的物理含义清晰：核心项 $\sigma_{\text{dim}}^{(u)} \cdot A_{\text{dim}}^{(r)}$ 表示 Prefill 请求在维度 dim 上施加的竞争压力与 Decode 请求在该维度上的敏感程度的耦合，四个维度求和后得到总干扰估算；IPC 修正项捕捉 pipeline 饱和度对 warp scheduler 竞争的额外影响；时间交错因子 $\Omega$ 反映两类请求的 Kernel 在时间上的重叠概率。在同模型 PD 共置场景下，Prefill 阶段通常具有较高的 $A_{\mathrm{cu}}$（计算密集）和较低的 $A_{\mathrm{bw}}$，而 Decode 阶段通常对 $\sigma_{\mathrm{bw}}$ 更为敏感，这一互补结构正是 PD 分离调度的理论基础——将计算密集的 Prefill 与访存敏感的 Decode 共置时，干扰可能低于两个同阶段任务的共置。

\subsection{从逐对估算到节点级估算}

上述加权映射规则以两个任务为对象，给出逐对干扰系数 $\hat{\alpha}_d(u, r)$ 和 $\hat{\alpha}_p(r, u)$。在实际调度场景中，一个 Decode 节点上通常同时运行同一模型的多个 Decode 请求，需要将逐对估算扩展到节点级，以支持第四章 CBS 评分和双向迁移机制中的干扰系数计算。

对于 Decode 节点 $d$ 在时刻 $t$ 上的活跃请求集合 $\mathcal{W}_d^{\text{dec}}(t)$，定义聚合干扰敏感度向量为各请求干扰敏感度按预估剩余输出长度加权平均：
\begin{equation}
\bar{\sigma}_{\text{dim}}(d, t) = \sum_{u \in \mathcal{W}_d^{\text{dec}}(t)} \frac{\hat{o}_{\mathrm{remain}}^{(u)}}{\sum_{v \in \mathcal{W}_d^{\text{dec}}(t)} \hat{o}_{\mathrm{remain}}^{(v)}} \cdot \sigma_{\text{dim}}^{(u)}, \quad \text{dim} \in \{\text{bs}, \text{cu}, \text{l2}, \text{bw}\}
\end{equation}

采用剩余输出长度作为权重的依据是：剩余输出越长的请求将在节点上持续更久，其干扰敏感度对节点整体状态的影响也更持久。聚合算子时间占比 $\bar{r}_{\mathrm{attn}}(d, t)$、$\bar{r}_{\mathrm{ffn}}(d, t)$，聚合 IPC $\overline{\text{IPC}}(d, t)$，以及聚合资源竞争强度 $\bar{A}_{\text{dim}}(d, t)$ 均按相同权重计算。聚合时间交错因子定义为 $\bar{\Omega}(r, d, t) = \min\!\big(r_{\mathrm{attn}}^{(r)} + r_{\mathrm{ffn}}^{(r)},\; \bar{r}_{\mathrm{attn}}(d, t) + \bar{r}_{\mathrm{ffn}}(d, t)\big)$。

基于聚合向量，定义节点级干扰系数。当新请求 $r$ 的 Prefill 与节点 $d$ 上的 Decode 负载共置时，Prefill 所受到的干扰可以表示为：
\begin{equation}
\hat{\alpha}_p(r, d, t) = \sum_{\text{dim}} w_{\text{dim}} \cdot \sigma_{\text{dim}}^{(r)} \cdot \bar{A}_{\text{dim}}(d, t) + w_{\text{ipc}} \cdot \overline{\text{IPC}}(d, t) \cdot \sigma_{\mathrm{cu}}^{(r)} + w_{\text{overlap}} \cdot \bar{\Omega}(d, r, t)
\end{equation}

节点上已有的 Decode 负载由于新请求共置而受到的平均干扰为：
\begin{equation}
\hat{\alpha}_d(d, r, t) = \sum_{\text{dim}} w_{\text{dim}} \cdot \bar{\sigma}_{\text{dim}}(d, t) \cdot A_{\text{dim}}^{(r)} + w_{\text{ipc}} \cdot \overline{\text{IPC}}^{(r)} \cdot \bar{\sigma}_{\mathrm{cu}}(d, t) + w_{\text{overlap}} \cdot \bar{\Omega}(r, d, t)
\end{equation}

\subsection{加权系数的离线标定与在线校准}

加权系数 $\mathbf{w} = (w_{\text{bs}}, w_{\mathrm{cu}}, w_{\text{l2}}, w_{\text{bw}}, w_{\text{ipc}}, w_{\text{overlap}})$ 通过离线标定方法确定。标定数据来源于同模型 PD 共置实验：在单个 vLLM 实例内，通过向纯 Decode 负载注入不同规模的 Prefill 请求，测量 Decode 时延的退化程度，获得实测 $\alpha_d$。对于实验矩阵 $\mathcal{M}$ 中的每个 $(\text{model}, b_{\mathrm{decode}}, b_{\mathrm{prefill}}, s_{\mathrm{prefill}})$ 组合，以最小二乘法拟合上述映射规则中的加权系数，使得估算值与实测值之间的均方误差最小：
\begin{equation}
\mathbf{w}^{\star} = \arg\min_{\mathbf{w}} \sum_{(r, u) \in \mathcal{M}} \left[(\hat{\alpha}_d(u, r; \mathbf{w}) - \alpha_d^{\text{true}})^2 + (\hat{\alpha}_p(r, u; \mathbf{w}) - \alpha_p^{\text{true}})^2\right]
\end{equation}

由于公式中每一项均为 $w_i$ 与不含 $\mathbf{w}$ 的特征乘积之和（$\Omega$ 中的 $\min$ 操作不涉及 $\mathbf{w}$），整体关于 $\mathbf{w}$ 是线性的，该优化问题可以通过普通最小二乘法（OLS）直接求解，计算开销极低。

在线校准阶段，系统持续记录每次调度决策对应的实际时延结果。当累计新样本达到了预设阈值（本文取 1000 条）时，利用新样本对加权系数进行增量更新。由于同模型 PD 共置场景下待估参数仅有 6 个，重新拟合的计算开销可忽略不计。通过这一机制，估算公式能够逐步适应在线负载分布的变化。

\section{本章小结}

本章提出了多层次工作负载描述符 MLWD，其由算子运行时画像层和请求级动态特征层构成，总维度为 $\mathbb{R}^{18}$，并作为干扰系数规则化估算的基础。

与现有主要依赖粗粒度硬件指标的方法相比，MLWD 进一步刻画了 Kernel 级资源占用画像，能够区分在相同 SM 利用率下具有不同计算模式的任务，从而为共置干扰估算提供更有判别力的特征。算子运行时画像层同时覆盖干扰估算所需的两个关键维度，即以计算强度和 L2 命中率为代表的资源竞争强度特征，以及以四维干扰敏感度向量为代表的干扰敏感度特征，并进一步引入算子时间交错特征和 IPC 指标，用于描述 Kernel 级执行模式。请求级动态特征层则编码阶段、批大小、序列长度和预估剩余输出长度，用于确定算子画像在当前请求条件下的实际取值，并支持面向未来行为的调度决策。在离线采集方面，本章设计了基于合成压力核的四维干扰敏感度采集方案，并结合参数化算子建模方法提出了 MLWD 外推机制，以降低离线采集成本并提升对未见 $(b, s)$ 组合的泛化能力。

在 MLWD 的基础上，本章提出了基于资源竞争强度与干扰敏感度加权映射的规则化干扰系数估算方法。该方法通过向量内积运算直接从两个共置任务的 MLWD 向量中估算得到 Prefill 和 Decode 的干扰系数 $\alpha_p$ 和 $\alpha_d$，维持了较低的在线计算开销。加权系数通过离线最小二乘法标定，同时进一步通过在线校准机制从而适应负载分布的动态变化。