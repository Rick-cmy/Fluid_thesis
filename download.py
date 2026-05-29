#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_papers.py
------------------
下载 Mingyang 论文文献包(v2,顶会顶刊 + 近期必引预印本)的开放获取 PDF,
并打包成 thesis_pdfs.zip。

在【有网、能访问 arxiv.org / aclanthology.org 的机器】上运行:
    python3 download_papers.py

产出:
    ./thesis_pdfs/         每篇一个 PDF(文件名带编号,便于对照清单)
    ./thesis_pdfs.zip      打包结果
    ./download_report.txt  成功/失败清单

说明:
- arXiv 条目用 arXiv API 按【标题】解析 ID 再下载,避免手抄 ID 出错;
  脚本会打印"请求标题 vs 实际命中标题",你扫一眼即可核对是否抓错。
- 少数 shared-task findings / 标准文档不在 arXiv,见文件末尾 MANUAL 列表,需手动下。
- 礼貌限速(arXiv API 要求 ≥3s/次),全程约 4–6 分钟。
"""

import os, re, sys, time, zipfile, urllib.parse, urllib.request

OUT_DIR = "papers"
ZIP_NAME = "papers.zip"
REPORT = "download_report.txt"
UA = {"User-Agent": "Mozilla/5.0 (academic paper downloader; personal use)"}
ARXIV_API = "http://export.arxiv.org/api/query?"

# 每条: (序号_短名, 标题, 类型, 值)
#   类型 "arxiv": 值=arXiv ID(本会话已核实的直接给;None=按标题解析)
#   类型 "url":   值=直接 PDF 链接
PAPERS = [
    # A. RAG 基础
    ("01_RAG_Lewis2020", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "arxiv", "2005.11401"),
    ("02_DPR_Karpukhin2020", "Dense Passage Retrieval for Open-Domain Question Answering", "arxiv", "2004.04906"),
    ("03_REALM_Guu2020", "REALM: Retrieval-Augmented Language Model Pre-Training", "arxiv", "2002.08909"),
    ("04_FiD_Izacard2021", "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", "arxiv", "2007.01282"),
    ("05_RETRO_Borgeaud2022", "Improving language models by retrieving from trillions of tokens", "arxiv", "2112.04426"),
    ("06_Atlas_Izacard2023", "Atlas: Few-shot Learning with Retrieval Augmented Language Models", "arxiv", "2208.03299"),
    ("07_kNNLM_Khandelwal2020", "Generalization through Memorization: Nearest Neighbor Language Models", "arxiv", "1911.00172"),
    ("08_ColBERT_Khattab2020", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", "arxiv", "2004.12832"),
    ("10_BEIR_Thakur2021", "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", "arxiv", "2104.08663"),
    ("11_SelfRAG_Asai2024", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "arxiv", "2310.11511"),
    ("12_AdaptiveRAG_Jeong2024", "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity", "arxiv", "2403.14403"),
    ("13_HyDE_Gao2023", "Precise Zero-Shot Dense Retrieval without Relevance Labels", "arxiv", "2212.10496"),
    ("14_FLARE_Jiang2023", "Active Retrieval Augmented Generation", "arxiv", "2305.06983"),
    ("15_InContextRALM_Ram2023", "In-Context Retrieval-Augmented Language Models", "arxiv", "2302.00083"),
    ("16_REPLUG_Shi2024", "REPLUG: Retrieval-Augmented Black-Box Language Models", "arxiv", "2301.12652"),
    ("17_LongTail_Kandpal2023", "Large Language Models Struggle to Learn Long-Tail Knowledge", "arxiv", "2211.08411"),
    ("18_PopQA_Mallen2023", "When Not to Trust Language Models: Investigating the Effectiveness of Parametric and Non-Parametric Memories", "arxiv", "2212.10511"),
    ("19_KILT_Petroni2021", "KILT: a Benchmark for Knowledge Intensive Language Tasks", "arxiv", "2009.02252"),

    # B. 多智能体框架
    ("20_AutoGen_Wu2024", "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework", "arxiv", "2308.08155"),
    ("21_CAMEL_Li2023", "CAMEL: Communicative Agents for Mind Exploration of Large Language Model Society", "arxiv", "2303.17760"),
    ("22_MetaGPT_Hong2024", "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework", "arxiv", "2308.00352"),
    ("23_ChatDev_Qian2024", "ChatDev: Communicative Agents for Software Development", "arxiv", "2307.07924"),
    ("24_AgentVerse_Chen2024", "AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors in Agents", "arxiv", "2308.10848"),
    ("25_GPTSwarm_Zhuge2024", "GPTSwarm: Language Agents as Optimizable Graphs", "arxiv", "2402.16823"),

    # C. 多智能体辩论
    ("26_Debate_Du2024", "Improving Factuality and Reasoning in Language Models through Multiagent Debate", "arxiv", "2305.14325"),
    ("27_ChatEval_Chan2024", "ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate", "arxiv", "2308.07201"),
    ("28_ReConcile_Chen2024", "ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs", "arxiv", "2309.13007"),
    ("29_PersuasiveDebate_Khan2024", "Debating with More Persuasive LLMs Leads to More Truthful Answers", "arxiv", "2402.06782"),

    # D. agent 原语
    ("30_CoT_Wei2022", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "arxiv", "2201.11903"),
    ("31_SelfConsistency_Wang2023", "Self-Consistency Improves Chain of Thought Reasoning in Language Models", "arxiv", "2203.11171"),
    ("32_ReAct_Yao2023", "ReAct: Synergizing Reasoning and Acting in Language Models", "arxiv", "2210.03629"),
    ("33_Toolformer_Schick2023", "Toolformer: Language Models Can Teach Themselves to Use Tools", "arxiv", "2302.04761"),
    ("34_SelfRefine_Madaan2023", "Self-Refine: Iterative Refinement with Self-Feedback", "arxiv", "2303.17651"),
    ("35_Reflexion_Shinn2023", "Reflexion: Language Agents with Verbal Reinforcement Learning", "arxiv", "2303.11366"),
    ("36_ToT_Yao2023", "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "arxiv", "2305.10601"),

    # E. MT 评估 / QE
    ("37_BLEU_Papineni2002", "BLEU: a Method for Automatic Evaluation of Machine Translation", "url", "https://aclanthology.org/P02-1040.pdf"),
    ("38_BERTScore_Zhang2020", "BERTScore: Evaluating Text Generation with BERT", "arxiv", "1904.09675"),
    ("39_BLEURT_Sellam2020", "BLEURT: Learning Robust Metrics for Text Generation", "arxiv", "2004.04696"),
    ("40_COMET_Rei2020", "COMET: A Neural Framework for MT Evaluation", "arxiv", "2009.09025"),
    ("41_CometKiwi_Rei2022", "CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task", "arxiv", "2209.06243"),
    ("42_MQM_Freitag2021", "Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation", "arxiv", "2104.14478"),
    ("43_xCOMET_Guerreiro2024", "xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection", "arxiv", "2310.10482"),
    ("44_QE4PE_Sarti2025", "QE4PE: Word-level Quality Estimation for Human Post-Editing", "arxiv", "2503.03044"),
    ("45_GEMBA_MQM_Kocmi2023", "GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4", "arxiv", "2310.13988"),
    ("46_MetricX24_Juraska2024", "MetricX-24: The Google Submission to the WMT 2024 Metrics Shared Task", "arxiv", None),
    ("47_PitfallsCOMET_Zouhar2024", "Pitfalls and Outlooks in Using COMET", "arxiv", None),
    ("48_UnsupQE_Fomicheva2020", "Unsupervised Quality Estimation for Neural Machine Translation", "arxiv", "2005.10608"),
    ("49_NMTHalluc_Guerreiro2023", "Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation", "arxiv", "2208.05309"),
    ("50_ClinicalHarm_Mehandru2023", "Physician Detection of Clinical Harm in Machine Translation", "arxiv", None),
    ("51_MQMAPE_Lu2025", "MQM-APE: Toward High-Quality Error Annotation Predictors with Automatic Post-Editing", "arxiv", None),
    ("55_LLMJudge_Zheng2023", "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", "arxiv", "2306.05685"),
    ("56_FActScore_Min2023", "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation", "arxiv", "2305.14251"),

    # F. 术语约束 / 领域 & 文档级翻译
    ("57_TermConstraint_Dinu2019", "Training Neural Machine Translation to Apply Terminology Constraints", "arxiv", "1906.01105"),
    ("58_LexConstrained_Susanto2020", "Lexically Constrained Neural Machine Translation with Levenshtein Transformer", "arxiv", "2004.12681"),
    ("59_LemmaTerm_Bergmanis2021", "Facilitating Terminology Translation with Target Lemma Annotations", "arxiv", "2101.10035"),
    ("60_TermAware_Bogoychev2023", "Terminology-Aware Translation with Constrained Decoding and Large Language Model Prompting", "arxiv", "2310.05824"),
    ("61_AdaptiveMT_Moslem2023", "Adaptive Machine Translation with Large Language Models", "arxiv", "2301.13294"),
    ("62_DocLevelMT_Wang2023", "Document-Level Machine Translation with Large Language Models", "arxiv", "2304.02210"),
    ("63_DocNMTSurvey_Maruf2021", "A Survey on Document-level Neural Machine Translation: Methods and Evaluation", "arxiv", "1912.08494"),

    # G. RAG 评估 / 整体评测
    ("64_RAGAS_Es2024", "RAGAS: Automated Evaluation of Retrieval Augmented Generation", "arxiv", "2309.15217"),
    ("65_HELM_Liang2023", "Holistic Evaluation of Language Models", "arxiv", "2211.09110"),

    # H. 金融 NLP
    ("66_FinQA_Chen2021", "FinQA: A Dataset of Numerical Reasoning over Financial Data", "arxiv", "2109.00122"),
    ("67_TATQA_Zhu2021", "TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance", "arxiv", "2105.07624"),

    # I. 近期必引预印本
    ("68_MAST_Cemri2025", "Why Do Multi-Agent LLM Systems Fail", "arxiv", "2503.13657"),
    ("69_MissMark_LaMalfa2025", "Large Language Models Miss the Multi-Agent Mark", "arxiv", None),
    ("70_MAATS_Wang2025", "MAATS: A Multi-Agent Automated Translation System Based on MQM Evaluation", "arxiv", "2505.14848"),
    ("71_MMAD", "M-MAD: Multidimensional Multi-Agent Debate for Advanced Machine Translation Evaluation", "arxiv", "2412.20127"),
    ("72_TransAgents2024", "Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts", "arxiv", "2405.11804"),
    ("73_CHORUS2026", "CHORUS", "arxiv", "2602.19016"),
    ("74_FairQE2026", "FairQE", "arxiv", "2604.21420"),
    ("75_DelTA2024", "DelTA: An Online Document-Level Translation Agent Based on Multi-Level Memory", "arxiv", "2410.08143"),
    ("76_AgenticRAGSurvey_Singh2025", "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG", "arxiv", "2501.09136"),
    ("77_FinMASBench2026", "Benchmarking Multi-Agent LLM Architectures for Financial Document Processing", "arxiv", "2603.22651"),
    ("78_FinAgentBench2025", "FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering", "arxiv", "2508.14052"),
    ("79_FinRetrieval2026", "FinRetrieval: A Benchmark for Financial Data Retrieval by AI Agents", "arxiv", "2603.04403"),
    ("80_AgentsHypeMT2025", "AI agents may be worth the hype but not the resources yet", "arxiv", "2505.01560"),

    # ===== v3 补充(81-130):强相关、放松 venue、与 v2 不重复 =====
    # J. 进阶 / agentic RAG
    ("81_RAPTOR_Sarthi2024", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", "arxiv", "2401.18059"),
    ("82_RADIT_Lin2024", "RA-DIT: Retrieval-Augmented Dual Instruction Tuning", "arxiv", "2310.01352"),
    ("83_RECOMP_Xu2024", "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation", "arxiv", "2310.04408"),
    ("84_SelfRoute_Xu2024", "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach", "arxiv", "2407.16833"),
    ("85_SpeculativeRAG_Wang2024", "Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting", "arxiv", "2407.08223"),
    ("86_CRAG_Yan2024", "Corrective Retrieval Augmented Generation", "arxiv", "2401.15884"),
    ("87_GraphRAG_Edge2024", "From Local to Global: A Graph RAG Approach to Query-Focused Summarization", "arxiv", "2404.16130"),

    # K. RAG vs 微调
    ("88_RAGvsFT_Ovadia2024", "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs", "arxiv", "2312.05934"),
    ("89_FTvsRAG_Soudani2024", "Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge", "arxiv", "2403.01432"),
    ("90_RAGvsFT_Balaguer2024", "RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture", "arxiv", "2401.08406"),

    # L. 成本 / 路由 / 效率
    ("91_FrugalGPT_Chen2023", "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance", "arxiv", "2305.05176"),
    ("92_RouteLLM_Ong2024", "RouteLLM: Learning to Route LLMs with Preference Data", "arxiv", "2406.18665"),
    ("93_HybridLLM_Ding2024", "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing", "arxiv", None),
    ("94_AutoMix_Aggarwal2024", "AutoMix: Automatically Mixing Language Models", "arxiv", None),
    ("95_RouterBench_Hu2024", "RouterBench: A Benchmark for Multi-LLM Routing System", "arxiv", "2403.12031"),
    ("96_MoA_Wang2024", "Mixture-of-Agents Enhances Large Language Model Capabilities", "arxiv", "2406.04692"),
    ("97_AgentsThatMatter_Kapoor2024", "AI Agents That Matter", "arxiv", "2407.01502"),

    # M. LLM 翻译方法
    ("98_GPTMT_Hendy2023", "How Good Are GPT Models at Machine Translation? A Comprehensive Evaluation", "arxiv", "2302.09210"),
    ("99_PromptPaLM_Vilar2023", "Prompting PaLM for Translation: Assessing Strategies and Performance", "arxiv", "2211.09102"),
    ("100_ALMA_Xu2024", "A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models", "arxiv", "2309.11674"),
    ("101_MAPS_He2024", "Exploring Human-like Translation Strategy with Large Language Models", "arxiv", "2305.04118"),
    ("102_ParroT_Jiao2023", "ParroT: Translating During Chat Using Large Language Models tuned with Human Translation and Feedback", "arxiv", "2304.02426"),
    ("103_Tower_Alves2024", "Tower: An Open Multilingual Large Language Model for Translation-Related Tasks", "arxiv", "2402.17733"),

    # N. QE / 错误标注与后编辑工具
    ("104_AutoMQM_Fernandes2023", "The Devil is in the Errors: Leveraging Large Language Models for Fine-grained Machine Translation Evaluation", "arxiv", "2308.07286"),
    ("105_InstructScore_Xu2023", "INSTRUCTSCORE: Explainable Text Generation Evaluation with Fine-grained Feedback", "arxiv", "2305.14282"),
    ("106_PostEditErrAnno_Ki2024", "Guiding Large Language Models to Post-Edit Machine Translation with Error Annotations", "arxiv", "2404.07851"),
    ("107_EAPrompt_Lu2024", "Error Analysis Prompting Enables Human-Like Translation Evaluation in Large Language Models", "arxiv", None),

    # O. 翻译记忆 / 检索增强 NMT / 重复
    ("108_NeuralFuzzyRepair_Bulte2019", "Neural Fuzzy Repair: Integrating Fuzzy Matches into Neural Machine Translation", "url", "https://aclanthology.org/P19-1175.pdf"),
    ("109_kNNMT_Khandelwal2021", "Nearest Neighbor Machine Translation", "arxiv", "2010.00710"),
    ("110_MonoTM_Cai2021", "Neural Machine Translation with Monolingual Translation Memory", "arxiv", "2105.11269"),
    ("111_SearchEngineNMT_Gu2018", "Search Engine Guided Neural Machine Translation", "arxiv", None),
    ("112_TMaugLLM_Mu2023", "Augmenting Large Language Model Translators via Translation Memories", "arxiv", "2305.17367"),
    ("113_RethinkTM_Hoang2023", "Rethinking Translation Memory Augmented Neural Machine Translation", "arxiv", "2306.06948"),

    # P. 术语约束解码
    ("114_GridBeam_Hokamp2017", "Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search", "arxiv", "1704.07138"),
    ("115_DynBeam_Post2018", "Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation", "arxiv", "1804.06609"),

    # Q. 金融 NLP
    ("117_ConvFinQA_Chen2022", "ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering", "arxiv", "2210.03849"),
    ("118_DocFinQA_Reddy2024", "DocFinQA: A Long-Document Financial Reasoning Dataset", "arxiv", "2401.06915"),
    ("119_PIXIU_Xie2023", "PIXIU: A Comprehensive Benchmark, Instruction Dataset and Large Language Model for Finance", "arxiv", "2306.05443"),
    ("120_FinBen_Xie2024", "FinBen: A Holistic Financial Benchmark for Large Language Models", "arxiv", "2402.12659"),
    ("121_FinGPT_Yang2023", "FinGPT: Open-Source Financial Large Language Models", "arxiv", "2306.06031"),
    ("122_FinBERT_Araci2019", "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", "arxiv", "1908.10063"),
    ("123_BloombergGPT_Wu2023", "BloombergGPT: A Large Language Model for Finance", "arxiv", "2303.17564"),
    ("124_FinanceBench_Islam2023", "FinanceBench: A New Benchmark for Financial Question Answering", "arxiv", "2311.11944"),
    ("125_GoldenTouchstone_Wu2024", "Golden Touchstone: A Comprehensive Bilingual Benchmark for Evaluating Financial Large Language Models", "arxiv", "2411.06272"),

    # R. 综述 & LLM-as-judge 偏差
    ("126_MASSurvey_Guo2024", "Large Language Model based Multi-Agents: A Survey of Progress and Challenges", "arxiv", "2402.01680"),
    ("127_AgentSurvey_Wang2024", "A Survey on Large Language Model based Autonomous Agents", "arxiv", "2308.11432"),
    ("128_LLMJudgeSurvey_Gu2024", "A Survey on LLM-as-a-Judge", "arxiv", "2411.15594"),
    ("129_LLMsJudgesSurvey_Li2024", "LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods", "arxiv", "2412.05579"),
    ("130_ManyMinds_2025", "Judging with Many Minds: Do More Perspectives Mean Less Prejudice", "arxiv", "2505.19477"),
]

# 不在 arXiv,需手动下(脚本会写进 report)
MANUAL = [
    ("09_BM25_Robertson2009", "The Probabilistic Relevance Framework: BM25 and Beyond", "Now Publishers / 期刊,无免费 PDF;按需检索"),
    ("52_WMT24Findings_Kocmi2024", "Findings of the WMT24 General/Metrics Shared Task", "https://aclanthology.org/events/wmt-2024/ 内检索"),
    ("53_WMT23QE_Blain2023", "Findings of the WMT 2023 Quality Estimation Shared Task", "https://aclanthology.org/2023.wmt-1.52/"),
    ("54_MQM_Lommel2014", "Multidimensional Quality Metrics (MQM) Framework", "标准文档,见 themqm.org / Tradumàtica"),
    ("116_WMT23TermTask_Semenov2023", "Findings of the WMT 2023 Terminology Shared Task", "https://aclanthology.org/2023.wmt-1.21/"),
]

def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def arxiv_resolve(title):
    """按标题查 arXiv,返回 (arxiv_id, hit_title) 或 (None, None)。"""
    q = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "start": 0, "max_results": 1})
    xml = http_get(ARXIV_API + q).decode("utf-8", "ignore")
    m_entry = re.search(r"<entry>(.*?)</entry>", xml, re.S)
    if not m_entry:
        return None, None
    entry = m_entry.group(1)
    m_id = re.search(r"<id>\s*https?://arxiv\.org/abs/([^<\s]+)", entry)
    m_ti = re.search(r"<title>(.*?)</title>", entry, re.S)
    if not m_id:
        return None, None
    arx = m_id.group(1).strip()
    arx = re.sub(r"v\d+$", "", arx)
    hit = re.sub(r"\s+", " ", m_ti.group(1)).strip() if m_ti else ""
    return arx, hit

def download_pdf(arxiv_id, dest):
    data = http_get(f"https://arxiv.org/pdf/{arxiv_id}")
    if data[:4] != b"%PDF":
        raise ValueError(f"下载内容不是 PDF({len(data)} bytes)")
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)

def download_url(url, dest):
    data = http_get(url)
    if data[:4] != b"%PDF":
        raise ValueError(f"下载内容不是 PDF({len(data)} bytes)")
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok, fail = [], []
    for i, (key, title, kind, val) in enumerate(PAPERS, 1):
        dest = os.path.join(OUT_DIR, key + ".pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[{i:>3}/{len(PAPERS)}] 已存在,跳过: {key}")
            ok.append((key, "已存在"))
            continue
        try:
            if kind == "url":
                size = download_url(val, dest)
                print(f"[{i:>3}/{len(PAPERS)}] OK  {key}  ({size//1024} KB)  <- {val}")
                ok.append((key, val))
            else:
                arx, hit = (val, "(直接ID)") if val else arxiv_resolve(title)
                if not arx:
                    raise ValueError("arXiv 标题解析失败")
                size = download_pdf(arx, dest)
                note = f"arXiv:{arx}"
                if hit and hit != "(直接ID)":
                    note += f"  | 命中标题: {hit[:70]}"
                print(f"[{i:>3}/{len(PAPERS)}] OK  {key}  ({size//1024} KB)  {note}")
                ok.append((key, note))
        except Exception as e:
            print(f"[{i:>3}/{len(PAPERS)}] 失败 {key}: {e}")
            fail.append((key, title, str(e)))
        time.sleep(3.2)

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(f"成功 {len(ok)} / 共 {len(PAPERS)}\n\n=== 成功 ===\n")
        for k, n in ok:
            f.write(f"  {k}\t{n}\n")
        f.write("\n=== 失败(请手动核对标题/ID 后重下)===\n")
        for k, t, e in fail:
            f.write(f"  {k}\t{t}\t[{e}]\n")
        f.write("\n=== 不在 arXiv,需手动下载 ===\n")
        for k, t, where in MANUAL:
            f.write(f"  {k}\t{t}\t{where}\n")

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(OUT_DIR)):
            if fn.endswith(".pdf"):
                z.write(os.path.join(OUT_DIR, fn), fn)
        if os.path.exists(REPORT):
            z.write(REPORT, REPORT)

    print(f"\n完成:成功 {len(ok)} / {len(PAPERS)},失败 {len(fail)}。")
    print(f"打包 -> {ZIP_NAME}(含 download_report.txt)")
    if fail:
        print("失败项见 download_report.txt;多半是标题里有特殊字符,手动补即可。")

if __name__ == "__main__":
    main()
