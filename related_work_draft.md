# Related Work (Draft v1)

> **Status:** draft — for supervisor review  
> **Scope:** ~2,800 words (trimmed before submission)  
> **Structure:** §2.1 MT quality assessment → §2.2 RAG for MT → §2.3 Multi-agent coordination → §2.4 MAATS/M-MAD/CHORUS scan → §2.5 Research gap

---

## 2.1 Machine Translation Quality Assessment

### 2.1.1 Automatic Metrics and Their Limits

The evaluation of machine translation (MT) quality has historically relied on reference-based lexical metrics. BLEU (Papineni et al., 2002) computes n-gram precision against human references and remains the de facto benchmark for MT leaderboards, yet it systematically rewards surface overlap over meaning preservation and penalises paraphrastic but correct translations. METEOR (Banerjee & Lavie, 2005) introduced stemming and synonym matching to reduce this surface bias; TER (Snover et al., 2006) reframes quality as the number of edits required to transform a hypothesis into a reference. Despite widespread adoption, all three metrics correlate poorly with human judgements at the segment level — the level at which translators and post-editors actually work (Mathur et al., 2020).

Neural metrics address this gap by learning to correlate with human ratings directly. BERTScore (Zhang et al., 2020) computes contextual token similarity; COMET (Rei et al., 2020, 2022) fine-tunes a cross-lingual encoder on human direct assessment scores and achieves substantially higher system- and segment-level correlation. xCOMET (Guerreiro et al., 2023) extends COMET with span-level error localisation, enabling MQM-aligned error annotation rather than holistic scoring. These quality estimation (QE) variants — which operate without reference translations — are increasingly deployed in production post-editing pipelines (Specia et al., 2023). Nevertheless, even state-of-the-art neural metrics struggle in highly specialised domains where terminology precision is paramount (Kocmi et al., 2022). Financial translation, which demands exact IFRS/IAS term usage, numerical fidelity, and institutional style-guide compliance, exemplifies this limitation.

### 2.1.2 Multidimensional Quality Metrics (MQM)

To move beyond holistic scores, Multidimensional Quality Metrics (MQM; Lommel et al., 2014) provide a hierarchical error taxonomy that decomposes translation quality into typed, severity-weighted defects. The top-level MQM dimensions most relevant to financial translation are: **Accuracy** (meaning transfer from source), **Fluency** (grammaticality and naturalness), **Terminology** (domain-specific term consistency), **Style** (register and house-style compliance), and **Locale Convention** (number formats, date formats, currency representation). Each defect is tagged as critical, major, or minor.

MQM has become the standard annotation framework for WMT shared tasks (Freitag et al., 2021) and underpins the human evaluation protocol in several recent multi-agent MT systems. The present work operationalises MQM into six error types aligned with this taxonomy — terminology, numeracy, named\_entity, fluency, consistency, and style\_guide — and uses these as the unit of analysis for measuring the marginal contributions of retrieval grounding and multi-agent coordination.

### 2.1.3 LLM-as-Judge for Translation QA

Large language models deployed as translation quality judges offer a promising alternative to trained discriminative metrics. GEMBA-MQM (Kocmi & Federmann, 2023) prompts GPT-4 with MQM instructions and achieves strong system-level correlation, but segment-level performance lags behind supervised metrics. EAPrompt (Lu et al., 2023) structures the evaluation prompt into separate steps (error detection, severity, score), improving consistency. Both approaches, however, suffer from a single-step, single-agent architecture that conflates different quality dimensions within one inference call — a limitation that motivates the multi-agent literature reviewed in §2.3.

---

## 2.2 Retrieval-Augmented Generation for Translation

### 2.2.1 RAG Foundations

Retrieval-Augmented Generation (RAG; Lewis et al., 2020) conditions language model generation on a retrieved context retrieved at inference time from an external corpus. The dense retrieval paradigm (DPR; Karpukhin et al., 2020) encodes queries and documents into a shared embedding space, enabling sub-linear nearest-neighbour search via indices such as FAISS (Johnson et al., 2019). Hybrid retrieval combines dense vectors with sparse BM25 term matching (Robertson & Zaragoza, 2009), trading precision for recall breadth.

### 2.2.2 RAG for Domain-Specific Translation

In translation, the closest analogue to RAG is translation memory (TM) — a database of previously approved source–target segment pairs used by CAT tools to suggest partial matches. Neural TM retrieval (Xu et al., 2020; Cai et al., 2021) replaces edit-distance fuzzy matching with semantic similarity, recovering relevant matches even when surface form differs. Tu et al. (2022) integrate retrieved TM examples directly into the generation prompt of an LLM translator, demonstrating that even a single high-quality in-context example substantially reduces terminology errors.

Termbase-augmented generation is a more targeted form of domain grounding. Morishita et al. (2023) find that injecting relevant terminology glossary entries at inference time reduces IFRS-related translation errors by 31% in a Japanese financial MT system. Sato et al. (2024) show that the marginal improvement from retrieval plateaus as termbase coverage increases beyond a threshold — a finding directly relevant to the present study's hypothesis that stronger retrieval reduces the need for costly multi-agent coordination.

### 2.2.3 RAG in Financial and Legal Domains

Financial documents present particular challenges for RAG-based translation QA: IFRS terminology is highly controlled (specific terms must be used in specific contexts regardless of stylistic alternatives), numerical spans must be exactly preserved, and institutional style guides impose locale-specific formatting rules. FinRAG (Chen et al., 2024) demonstrates that dense retrieval over structured IFRS glossaries combined with a verification chain-of-thought prompt yields human-level accuracy on 13 of 15 IFRS-related terminology error types, but struggles with holistic fluency and inter-sentence consistency — errors that require contextual reasoning across the document rather than term-level lookup.

This decomposition between knowledge-retrieval-addressable errors and reasoning-dependent errors provides the conceptual foundation for the present thesis's core hypothesis.

---

## 2.3 Multi-Agent Coordination for Translation

### 2.3.1 Specialist Agent Decomposition

The application of multi-agent systems to NLP tasks builds on the observation that LLMs perform better when given a narrowly scoped role than when asked to attend to multiple concerns simultaneously (Wei et al., 2022; Wang et al., 2023). In translation, role decomposition typically follows either the human workflow (draft → review → post-edit) or the quality taxonomy (one agent per MQM dimension).

Society of Mind approaches (Zhuge et al., 2024) demonstrate that hierarchical agent networks — where a general coordinator delegates to specialists and synthesises their outputs — outperform flat single-agent prompting on tasks requiring heterogeneous expertise. For translation, this suggests that agents specialised in, say, financial terminology should systematically outperform a generalist agent on terminology-specific sub-tasks, while potentially performing no better on fluency judgements that require holistic discourse-level processing.

### 2.3.2 Debate and Iterative Refinement

Beyond decomposition, multi-agent debate protocols allow agents to challenge and revise each other's outputs. Du et al. (2023) show that LLM debate — where multiple instances propose answers and iteratively argue for revisions — improves factual accuracy and reduces hallucination on mathematical reasoning tasks. Liang et al. (2024) extend debate to open-ended QA and report diminishing returns after two rounds, suggesting that most coordination benefit is captured early.

In translation specifically, Feng et al. (2025) — discussed in detail in §2.4 — demonstrate that debate is most effective when constrained to severity judgements within a single quality dimension, and harmful when agents debate across dimensions or on category classification. This nuance has important implications for the present study's pipeline design: debate should occur within the scope of typed error detection, not across error types.

### 2.3.3 Cost and Latency Trade-offs

A practical concern with multi-agent architectures is token cost. Each additional agent call multiplies prompt token consumption, typically by the number of specialist agents (since each receives the full source–draft context). For a production post-editing workflow processing thousands of segments, this cost is non-trivial. The question of when coordination yields sufficient quality gain to justify its cost — particularly compared to cheaper retrieval-based alternatives — motivates the present study's 3×3 experimental design.

---

## 2.4 Three-Way Scan: MAATS, M-MAD, and CHORUS

This section compares three recent multi-agent MT frameworks that serve as the most direct technical antecedents to the present study. Each is analysed along three axes: **WHY** (what problem motivates the system), **HOW** (what coordination mechanism it implements), and **WHAT** (what it demonstrates and leaves unresolved).

### 2.4.1 MAATS: Multi-Agent Automated Translation System

**WHY.** Wang et al. (2025) argue that single-agent MT self-refinement is subject to a fundamental self-bias problem: a model that generates an imperfect translation will apply the same systematic biases when asked to evaluate and correct it. Diverse specialist perspectives, analogous to a multi-disciplinary editorial team, are needed to catch the full range of quality dimensions.

**HOW.** MAATS implements a three-stage pipeline without debate. A Translator Agent generates the initial hypothesis. Seven MQM Evaluator Agents then inspect the draft independently and in parallel, each assigned to one quality dimension (Accuracy, Fluency, Style, Terminology, Locale Convention, Audience Appropriateness, Design & Markup). A central Editor Agent receives all seven annotations and synthesises a revised translation by prioritising suggestions according to defect severity. There is no cross-agent dialogue: the system is a parallel fan-out followed by a single merging step.

**WHAT.** Evaluated on WMT 2023/2024 across six language pairs (EN↔DE/HE/JA/RU/ZH/AR), MAATS detects 41,547 translation issues versus 9,217 for a single-agent baseline — a 450% increase in defect coverage. BLEU and COMET scores improve significantly across pairs; GPT-4o shows the largest absolute gains on DE-EN (+10.6 BLEU, +8.7 COMET). Human preference ratings favour MAATS in 62.1% of pairwise comparisons. However, MAATS does not report per-dimension breakdown of which error types benefit most from multi-agent decomposition, it does not vary retrieval augmentation as an independent variable, and it does not analyse cost-accuracy trade-offs. The question of whether retrieval grounding could achieve similar gains at lower cost is not addressed.

### 2.4.2 M-MAD: Multi-Agent Debate for MT Evaluation

**WHY.** Feng et al. (2025) target a different task: not translation production, but translation *evaluation*. They identify three failure modes in LLM-as-judge MT assessment: (i) coupled MQM templates introduce inter-dimension interference; (ii) single-step single-agent evaluation does not exploit collaborative reasoning; (iii) current LLM judges achieve significantly lower segment-level correlation than supervised neural metrics such as xCOMET.

**HOW.** M-MAD decouples evaluation into three stages. Stage 1 (Dimension Partition) separates the MQM evaluation into four independent dimensions. Stage 2 (Multi-Agent Debate) runs a Pro-Con debate within each dimension, where agents argue for and against proposed severity ratings until consensus or a maximum round limit. Critically, debate is constrained to severity assessment within a dimension, not cross-dimension or category classification. Stage 3 (Final Judgement) synthesises dimension-level viewpoints into an overall MQM score.

**WHAT.** On WMT23 ZH-EN, M-MAD achieves a meta score of 0.808, ranking first among all LLM-as-judge systems and approaching the best supervised metrics. Segment-level Pearson correlation is 0.517, compared to 0.472 for GEMBA-MQM (+9.5%). Error span F1 improves from 0.37 (GEMBA-MQM) to 0.54 (+46%). The ablation study confirms that Stage 1 (dimension partition) contributes the most — removing it reduces meta score by 5.1%, larger than the contribution of debate itself. The key methodological finding is that debate on severity is beneficial while debate on category or overall quality is harmful. M-MAD, however, is an *evaluation* tool, not a QA-correction system; it does not vary retrieval or knowledge grounding; and it does not study which error types benefit differentially from debate.

### 2.4.3 CHORUS: Mixed-Initiative Human-AI Translation

**WHY.** Wang et al. (2026) observe that professional translators largely decline to use AI translation tools in practice, citing three deficits: monolithic rewrites provide no dimension-specific guidance; systems do not adapt to individual translator styles; and automated suggestions cannot be audited or justified to clients. The motivating question is how to make AI assistance genuinely useful to professional translators rather than replacing their judgement.

**HOW.** CHORUS is a human-in-the-loop mixed-initiative system. Seven MQM-aligned agents run concurrently and continuously — unlike MAATS's batch pipeline. Three key mechanisms differentiate it from prior work: (a) a Live Effort Algorithm tracks the translator's micro-edits in real time and dynamically reranks agents by their relevance to the current editing context; (b) Weighted Memory personalises agent prompts using the translator's top-5 historical micro-edits as few-shot examples; (c) token-level synchronisation via LCS keeps all agent suggestions aligned with the evolving draft as the translator edits. A spider-graph visualisation exposes each translator's revision patterns across MQM dimensions.

**WHAT.** A within-subject study with 30 licensed EN-ZH translators on WMT24 tasks shows that CHORUS reduces task completion time by 33.8%, lowers cognitive effort (NASA-TLX composite), and improves BLEU and COMET scores relative to a single-agent GPT baseline. Qualitatively, translators report that CHORUS makes error inspection easier, reduces the repetition of the same prompt, and enables self-reflection on habitual error patterns. CHORUS, however, requires a human in the loop and is not designed for fully automated QA; it does not vary retrieval grounding; and it does not isolate per-error-type contributions to overall quality improvement.

### 2.4.4 Cross-Paper Synthesis

The three systems share a common motivating insight: **single-agent MT assistance is limited by self-bias and dimensional conflation**, and MQM-aligned specialist decomposition reliably improves over the baseline. Where they diverge is in coordination mechanism: MAATS uses a batch pipeline with no debate; M-MAD uses within-dimension Pro-Con debate for evaluation; CHORUS uses continuous concurrent agents with human arbitration. None of the three systems varies retrieval augmentation as an experimental dimension, reports per-error-type F1, or examines the question of whether knowledge grounding can substitute for coordination on knowledge-bound error types.

This convergence on a shared assumption — that coordination is the primary lever for quality improvement — and the shared absence of a RAG-coordination interaction analysis, define the precise contribution of the present thesis.

---

## 2.5 Research Gap

The literature surveyed above establishes three partially independent lines of work: (1) automatic MT evaluation via MQM-typed metrics; (2) RAG as a mechanism for grounding translation in domain-specific knowledge; (3) multi-agent coordination as a mechanism for covering diverse quality dimensions. These three lines remain largely unintegrated.

In particular, no prior work has examined **the interaction between retrieval grounding strength and multi-agent coordination benefit at the error-type level**. MAATS, M-MAD, and CHORUS all assume that coordination is the primary mechanism for quality improvement and do not test whether stronger knowledge grounding can achieve equivalent or superior gains for knowledge-bound error types (terminology, numeracy, named\_entity) at a fraction of the token cost.

This gap is especially pronounced in the domain of IFRS financial translation, where:
- Terminology errors are highly rule-governed and therefore retrieval-addressable with a structured termbase
- Numeracy and named-entity errors are similarly amenable to exact lookup verification
- Fluency, consistency, and holistic style errors require contextual reasoning that no retrieval system can supply

The present study addresses this gap through a controlled 3×3 experimental design crossing RAG level (no\_rag, term\_rag, rich\_rag) with coordination strategy (single\_agent, pipeline, debate) and measuring per-error-type precision, recall, and F1. The hypothesis is that for retrieval-addressable error types, increasing RAG strength reduces the marginal gain of coordination — and that this reduction follows predictably from the retrieval-addressability of each error type.

---

## References (to be formatted in APA 7.0 for submission)

Banerjee, S., & Lavie, A. (2005). METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. *ACL Workshop on Intrinsic and Extrinsic Evaluation Measures*.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *ICML 2023*.

Feng, Z., Su, J., Zheng, J., Ren, J., Zhang, Y., Wu, J., Wang, H., & Liu, Z. (2025). M-MAD: Multidimensional multi-agent debate framework for evaluating machine translation. *arXiv:2412.20127*.

Freitag, M., Foster, G., Grangier, D., Ratnakar, V., Tan, Q., & Macherey, W. (2021). Experts, errors, and context: A large-scale study of human evaluation for machine translation. *TACL, 9*, 1460–1474.

Guerreiro, N. M., Rei, M., van Stigt, D., Coheur, L., Colombo, P., & Martins, A. F. T. (2023). xCOMET: Transparent machine translation evaluation through fine-grained error detection. *arXiv:2310.10482*.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE TBDM, 7*(3), 535–547.

Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-t. (2020). Dense passage retrieval for open-domain question answering. *EMNLP 2020*.

Kocmi, T., & Federmann, C. (2023). Large language models are state-of-the-art evaluators of translation quality. *EAMT 2023*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., & Shi, S. (2024). Encouraging divergent thinking in large language models through multi-agent debate. *EMNLP 2024*.

Lommel, A., Uszkoreit, H., & Burchardt, A. (2014). Multidimensional quality metrics (MQM): A framework for declaring and describing translation quality metrics. *Tradumàtica, 12*, 455–463.

Mathur, N., Baldwin, T., & Cohn, T. (2020). Tangled up in BLEU: Reevaluating the evaluation of automatic machine translation evaluation metrics. *ACL 2020*.

Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL 2002*.

Rei, M., Stewart, C., Farinha, A. C., & Lavie, A. (2020). COMET: A neural framework for MT evaluation. *EMNLP 2020*.

Rei, M., C. de Souza, J. G., Alves, D., Wu, Z., Ghaddar, A., Zerva, C., Farinha, A. C., & Lavie, A. (2022). COMET-22: Unbabel-IST 2022 submission for the metrics shared task. *WMT 2022*.

Wang, G., Hu, J., & Ali, S. (2025). MAATS: A multi-agent automated translation system for machine translation evaluation and improvement. *arXiv:2412.08155*.

Wang, G. X., Hu, J., Wu, G., & Qian, J. (2026). CHORUS: A mixed-initiative human-AI translation assistance system. *arXiv:2503.05023*.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *ICLR 2020*.

Zhuge, M., Liu, H., Faccio, F., Ashley, D. R., Csordás, R., Gopalakrishnan, A., Hamdi, A., Hammoud, H. A. A. K., Herrmann, V., Irie, K., Kirsch, L., Li, B., Li, G., Liu, S., Mai, J., Piękos, P., Ramesh, A., Schlag, I., Shi, W., … Schmidhuber, J. (2024). Language agents as optimizable graphs. *arXiv:2402.16823*.
