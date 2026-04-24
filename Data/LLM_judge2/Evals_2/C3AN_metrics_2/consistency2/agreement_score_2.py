"""
BMEN-499 AlphaFold -- LLM Judge 2: Agreement Score (BioGPT vs Ground Truth)
----------------------------------------------------------------------------
Purpose:
    Computes agreement between BioGPT-generated answers and ground truth
    answers parsed directly from the BioGPT evaluation log file.
    No API calls required -- all metrics are computed locally.

Background:
    The original 3-judge evaluation failed with HTTP 401 errors for all
    100 questions. This script replaces that evaluation using five local
    agreement metrics that together form a composite agreement score.

Input:
    --log   : BioGPT evaluation log (Ground Truth AND BioGPT answers inside)
    --gt    : Optional separate ground_truth_output.txt to override log GT
    --output: Optional output path (default: agreement_score_2.txt)
    --demo  : Run on 5 built-in sample pairs

Agreement Metrics (all local, zero dependencies beyond stdlib):
    ROUGE-1    Unigram F1       basic lexical agreement
    ROUGE-2    Bigram  F1       phrase-level agreement
    Precision  BioGPT tokens found in ground truth
    Recall     Ground truth tokens found in BioGPT answer
    Keyword    Domain keyword coverage (disorder, IDR, pLDDT, etc.)
    COMPOSITE  mean of all 5  [0.0 = no agreement, 1.0 = perfect]

Output:
    agreement_score_2.txt  saved to same folder as this script

Usage:
    python agreement_score_2.py --log BioGPT_eval_log.txt
    python agreement_score_2.py --log log.txt --gt ground_truth_output.txt
    python agreement_score_2.py --demo
"""

import re, os, sys, math, argparse
from pathlib import Path
from collections import Counter

# ── stopwords ────────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
    "from","is","are","was","were","be","been","has","have","had","do","does",
    "did","not","this","that","these","those","it","its","we","our","they",
    "their","as","than","more","also","can","may","would","could","should",
    "which","when","where","how","what","who","will","if","so","such","each",
    "both","very","most","some","any","all","per","into","after","before",
    "between","within","across","about","while","however","therefore","thus",
    "one","two","three",
}

DOMAIN_KEYWORDS = [
    "disorder","disordered","idr","idrs","idp","idps","disprot","alphafold",
    "plddt","confidence","score","scores","prediction","predictions","protein",
    "proteins","residue","residues","sequence","region","regions","calibration",
    "threshold","cutoff","proline","glycine","amino","acid","neural","symbolic",
    "neurosymbolic","rag","retrieval","window","sliding","smoothing","noise",
    "variance","bias","precision","recall","f1","auroc","mcc","brier","mae",
    "isotonic","pfam","domain","terminal","binding","morf","structured",
    "unstructured","folded","mean","fraction",
]

# ── tokeniser ────────────────────────────────────────────────────────────────
def tokenize(text, remove_sw=True):
    text = re.sub(r"[^a-z0-9\s]"," ", text.lower())
    toks = text.split()
    if remove_sw:
        toks = [t for t in toks if t not in STOPWORDS and len(t)>1]
    return toks

def bigrams(toks):
    return [(toks[i],toks[i+1]) for i in range(len(toks)-1)]

# ── metrics ──────────────────────────────────────────────────────────────────
def rouge1(cand,ref):
    ct=Counter(tokenize(cand)); rt=Counter(tokenize(ref))
    if not ct or not rt: return 0.0
    ov=sum((ct&rt).values())
    p=ov/sum(ct.values()); r=ov/sum(rt.values())
    return 2*p*r/(p+r) if p+r else 0.0

def rouge2(cand,ref):
    ct=Counter(bigrams(tokenize(cand,False))); rt=Counter(bigrams(tokenize(ref,False)))
    if not ct or not rt: return 0.0
    ov=sum((ct&rt).values())
    p=ov/sum(ct.values()); r=ov/sum(rt.values())
    return 2*p*r/(p+r) if p+r else 0.0

def prec(cand,ref):
    ct=set(tokenize(cand)); rt=set(tokenize(ref))
    return len(ct&rt)/len(ct) if ct else 0.0

def rec(cand,ref):
    ct=set(tokenize(cand)); rt=set(tokenize(ref))
    return len(ct&rt)/len(rt) if rt else 0.0

def kw_cov(cand,ref):
    rl=ref.lower(); cl=cand.lower()
    ref_kws=[k for k in DOMAIN_KEYWORDS if k in rl]
    if not ref_kws:
        ck=[k for k in DOMAIN_KEYWORDS if k in cl]
        return min(len(ck)/max(len(DOMAIN_KEYWORDS)*0.1,1),1.0)
    return sum(1 for k in ref_kws if k in cl)/len(ref_kws)

def score_pair(biogpt,gt):
    r1=rouge1(biogpt,gt); r2=rouge2(biogpt,gt)
    p=prec(biogpt,gt);    r=rec(biogpt,gt); kw=kw_cov(biogpt,gt)
    return {"rouge1":round(r1,4),"rouge2":round(r2,4),"precision":round(p,4),
            "recall":round(r,4),"keyword":round(kw,4),"composite":round((r1+r2+p+r+kw)/5,4)}

# ── parsers ──────────────────────────────────────────────────────────────────
def _clean(text):
    for bad,good in [("ΓÇô","-"),("ΓÇö","--"),("╬▒","alpha"),("╬▓","beta"),
                     ("\u2019","'"),("\u2013","-"),("\u2014","--")]:
        text=text.replace(bad,good)
    return text

def parse_log(filepath):
    raw=_clean(Path(filepath).read_text(encoding="utf-8",errors="replace"))
    blocks=re.split(r"\[Q(\d+)\]\s*",raw)
    records=[]
    for i in range(1,len(blocks),2):
        qnum=int(blocks[i]); content=blocks[i+1] if i+1<len(blocks) else ""
        lines=[l.strip() for l in content.split("\n") if l.strip()]
        question=lines[0] if lines else f"Q{qnum}"
        gt_m=re.search(r"\[Ground Truth\]\s*\n(.*?)(?=\n\s*\[BioGPT|\Z)",content,re.DOTALL)
        bg_m=re.search(r"\[BioGPT Answer\]\s*\n(.*?)(?=\n\s*\[Judge|\n-{20,}|\Z)",content,re.DOTALL)
        gt=gt_m.group(1).strip() if gt_m else ""
        bg=bg_m.group(1).strip() if bg_m else ""
        if gt or bg:
            records.append({"question_id":qnum,"question":question,
                             "ground_truth":gt,"biogpt_answer":bg})
    return records

def parse_gt_file(filepath):
    raw=_clean(Path(filepath).read_text(encoding="utf-8",errors="replace"))
    if "[Ground Truth]" in raw:
        blocks=re.split(r"\[Q(\d+)\]\s*",raw); d={}
        for i in range(1,len(blocks),2):
            m=re.search(r"\[Ground Truth\]\s*\n(.*?)(?=\n\s*\[|\Z)",
                        blocks[i+1] if i+1<len(blocks) else "",re.DOTALL)
            if m: d[int(blocks[i])]=m.group(1).strip()
        if d: return d
    d={}
    for line in raw.splitlines():
        m=re.match(r"Q?(\d+)[:\.\)]\s*(.+)",line.strip())
        if m: d[int(m.group(1))]=m.group(2).strip()
    return d or {0:raw.strip()}

# ── aggregate ────────────────────────────────────────────────────────────────
def run_scoring(records):
    scored=[]
    for rec in records:
        m=score_pair(rec["biogpt_answer"],rec["ground_truth"])
        scored.append({**rec,**m})

    def stats(key):
        vals=[s[key] for s in scored]
        if not vals: return {"mean":0,"stdev":0,"min":0,"median":0,"max":0}
        mu=sum(vals)/len(vals)
        sd=math.sqrt(sum((x-mu)**2 for x in vals)/max(len(vals)-1,1))
        sv=sorted(vals)
        return {"mean":round(mu,4),"stdev":round(sd,4),"min":round(min(vals),4),
                "median":round(sv[len(sv)//2],4),"max":round(max(vals),4)}

    bk={"STRONG(0.8-1.0)":0,"MODERATE(0.6-0.8)":0,"WEAK(0.4-0.6)":0,
        "POOR(0.2-0.4)":0,"NONE(0.0-0.2)":0}
    for s in scored:
        c=s["composite"]
        if   c>=0.80: bk["STRONG(0.8-1.0)"]+=1
        elif c>=0.60: bk["MODERATE(0.6-0.8)"]+=1
        elif c>=0.40: bk["WEAK(0.4-0.6)"]+=1
        elif c>=0.20: bk["POOR(0.2-0.4)"]+=1
        else:         bk["NONE(0.0-0.2)"]+=1

    by_c=sorted(scored,key=lambda x:x["composite"],reverse=True)
    return {"n":len(scored),"scored":scored,
            "rouge1_stats":stats("rouge1"),"rouge2_stats":stats("rouge2"),
            "precision_stats":stats("precision"),"recall_stats":stats("recall"),
            "keyword_stats":stats("keyword"),"composite_stats":stats("composite"),
            "distribution":bk,"top10":by_c[:10],"bottom10":by_c[-10:]}

# ── report ───────────────────────────────────────────────────────────────────
def bar(v,w=36): n=max(0,min(int(v*w),w)); return "█"*n+"░"*(w-n)

def wrap(text,indent=2,width=72):
    words=text.split(); out=[]; line=" "*indent
    for w in words:
        if len(line)+len(w)+1>width: out.append(line); line=" "*indent+w+" "
        else: line+=w+" "
    if line.strip(): out.append(line)
    return out

def write_report(res,output_path):
    n=res["n"]; L=[]
    L+=["="*72,
        "  BMEN-499 AlphaFold -- LLM Judge 2: Agreement Score",
        "  Metric   : BioGPT Answers vs. Ground Truth (local, no API)",
        f"  Questions: {n}","="*72,""]
    L+=["WHY THIS SCRIPT EXISTS","-"*72,
        "  The original 3-judge evaluation returned HTTP 401 errors for all",
        "  100 questions. This script replaces it with five deterministic",
        "  local metrics computed directly from BioGPT answers vs. ground",
        "  truth extracted from the same log -- no API or model required.","",
        "METRICS","-"*72,
        "  ROUGE-1    Unigram F1   -- basic word-level agreement",
        "  ROUGE-2    Bigram  F1   -- phrase-level agreement",
        "  Precision  fraction of BioGPT tokens found in ground truth",
        "  Recall     fraction of ground truth tokens in BioGPT answer",
        "  Keyword    domain keyword coverage (disorder,IDR,pLDDT...)",
        "  COMPOSITE  mean of above  [0=none, 1=perfect]",""]

    # summary table
    L+=["="*72,"  SUMMARY STATISTICS","-"*72,
        f"  {'Metric':<18}  {'Mean':>7}  {'Stdev':>7}  {'Min':>7}  {'Median':>7}  {'Max':>7}",
        "  "+"-"*62]
    for lbl,key in [("ROUGE-1 F1","rouge1_stats"),("ROUGE-2 F1","rouge2_stats"),
                    ("Precision","precision_stats"),("Recall","recall_stats"),
                    ("Keyword Cov.","keyword_stats"),("COMPOSITE","composite_stats")]:
        s=res[key]
        L.append(f"  {lbl:<18}  {s['mean']:>7.4f}  {s['stdev']:>7.4f}  "
                 f"{s['min']:>7.4f}  {s['median']:>7.4f}  {s['max']:>7.4f}")
    L.append("")

    cm=res["composite_stats"]["mean"]
    if   cm>=0.60: lvl="MODERATE-STRONG"
    elif cm>=0.40: lvl="WEAK-MODERATE"
    elif cm>=0.20: lvl="POOR"
    else:          lvl="NEAR-ZERO"

    imap={"MODERATE-STRONG":
              "BioGPT answers show meaningful overlap with ground truth. "
              "The model captures relevant biomedical terminology and some "
              "factual content, though answers are short fragments.",
          "WEAK-MODERATE":
              "BioGPT answers show partial overlap. The model generates "
              "contextually related text but misses specific quantitative "
              "claims present in the DisProt-derived ground truth.",
          "POOR":
              "BioGPT answers show low overlap. The model generates "
              "grammatically plausible but semantically shallow one-line "
              "responses that rarely match the factual ground truth content.",
          "NEAR-ZERO":
              "BioGPT answers show near-zero agreement with ground truth. "
              "The model outputs title-style fragments ('A meta-analysis.', "
              "'The effect of disorder on protein function.') with no "
              "factual alignment to DisProt-derived statistics. This "
              "confirms the core motivation for RAG augmentation: BioGPT "
              "alone cannot answer data-specific disorder prediction queries."}

    L+=[f"  Overall agreement level : {lvl}",
        f"  Composite mean          : {cm:.4f}",""]
    L+=wrap(imap[lvl]); L.append("")

    # distribution
    L+=["="*72,"  COMPOSITE SCORE DISTRIBUTION","-"*72]
    for bkt,cnt in res["distribution"].items():
        pct=cnt/n*100 if n else 0
        L.append(f"  {bkt:<26}  {cnt:>4} ({pct:>5.1f}%)  "
                 f"{bar(cnt/n if n else 0, 20)}")
    L.append("")

    # bar chart
    L+=["="*72,"  MEAN SCORE PER METRIC","-"*72]
    for lbl,key in [("ROUGE-1","rouge1_stats"),("ROUGE-2","rouge2_stats"),
                    ("Precision","precision_stats"),("Recall","recall_stats"),
                    ("Keyword","keyword_stats"),("COMPOSITE","composite_stats")]:
        mu=res[key]["mean"]
        L.append(f"  {lbl:<12}  {mu:.4f}  {bar(mu,36)}")
    L.append("")

    # top / bottom
    hdr=(f"  {'Q#':<5}  {'Comp':>6}  {'R1':>6}  {'R2':>6}  "
         f"{'Prec':>6}  {'Rec':>6}  {'KW':>6}  Question")
    def row(s):
        return (f"  Q{s['question_id']:<4}  {s['composite']:>6.4f}  "
                f"{s['rouge1']:>6.4f}  {s['rouge2']:>6.4f}  "
                f"{s['precision']:>6.4f}  {s['recall']:>6.4f}  "
                f"{s['keyword']:>6.4f}  {s['question'][:35]}")

    L+=["="*72,"  TOP 10 -- BEST AGREEMENT","-"*72,hdr,"  "+"-"*68]
    L+=[row(s) for s in res["top10"]]; L.append("")
    L+=["="*72,"  BOTTOM 10 -- WORST AGREEMENT","-"*72,hdr,"  "+"-"*68]
    L+=[row(s) for s in res["bottom10"]]; L.append("")

    # examples
    L+=["="*72,"  ANSWER EXAMPLES","-"*72,""]
    for lbl,items in [("BEST (top 3)",res["top10"][:3]),
                      ("WORST (bottom 3)",list(reversed(res["bottom10"][-3:])))]:
        L.append(f"  -- {lbl} --")
        for s in items:
            L+=[f"  Q{s['question_id']:03d}  composite={s['composite']:.4f}  "
                f"[R1={s['rouge1']:.3f} R2={s['rouge2']:.3f} "
                f"Prec={s['precision']:.3f} Rec={s['recall']:.3f} KW={s['keyword']:.3f}]",
                f"  Q:  {s['question'][:70]}",
                f"  GT: {s['ground_truth'][:110]}",
                f"  BG: {s['biogpt_answer'][:110]}",""]

    # per-question table
    L+=["="*72,"  PER-QUESTION SCORES","-"*72,hdr,"  "+"-"*60]
    for s in res["scored"]:
        c=s["composite"]
        rtg="STRONG" if c>=0.60 else "MODERATE" if c>=0.40 else "WEAK" if c>=0.20 else "POOR"
        L.append(f"  Q{s['question_id']:<4}  {s['composite']:>6.4f}  "
                 f"{s['rouge1']:>6.4f}  {s['rouge2']:>6.4f}  "
                 f"{s['precision']:>6.4f}  {s['recall']:>6.4f}  "
                 f"{s['keyword']:>6.4f}  {rtg}")
    L.append("")

    # diagnostic
    L+=["="*72,"  DIAGNOSTIC NOTES","-"*72]
    L+=wrap("BioGPT (microsoft/biogpt) generates short title-style completions "
            "for most questions. Typical outputs are 5-15 words framed as paper "
            "titles rather than factual answers, explaining the low ROUGE-2 and "
            "recall scores.")
    L.append("")
    L+=wrap("Ground truth answers are DisProt-derived statistics (e.g. "
            "'mean disorder=0.378, 29.1% exceed 0.5 cutoff') requiring specific "
            "numerical knowledge that BioGPT lacks without retrieval augmentation. "
            "This confirms the motivation for the RAG pipeline.")
    L.append("")
    L+=wrap("Keyword coverage is consistently the strongest metric because "
            "BioGPT answers contain relevant domain vocabulary even when factual "
            "content is absent. ROUGE-2 and recall are the weakest.")
    L.append("")
    L+=["="*72,
        "  END OF REPORT",
        "  Project: BMEN-499 Independent Research -- Michelle Ihetu, USC",
        "="*72]

    output="\n".join(L)
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    with open(output_path,"w",encoding="utf-8") as f: f.write(output)
    print(output)
    print(f"\n[SAVED] Report written to: {output_path}\n")

# ── demo data ─────────────────────────────────────────────────────────────────
DEMO = """\
[Q1] Is a disorder score above 0.5 a reliable cutoff for calling a region disordered?

  [Ground Truth]
  Across 13396 DisProt proteins, 29.1% exceed disorder content 0.5 (mean=0.378). A 0.5 cutoff is commonly used but may be conservative - 44.2% exceed 0.3, suggesting many IDRs fall in the mid-range where 0.5 may undercount disorder.

  [BioGPT generating...]
  [BioGPT Answer]
  A meta-analysis.

  [Judge Scores]
    Judge-1 (Strict): [parse error] [API ERROR: HTTP Error 401: Unauthorized]
----------------------------------------------------------------------

[Q2] Do confidence scores drop for IDRs shorter than 10 residues?

  [Ground Truth]
  Of 0 annotated disordered regions, 0.0% are shorter than 10 residues (mean length=0.0 aa). Short IDRs are underrepresented in DisProt, consistent with lower prediction confidence for very short disordered stretches.

  [BioGPT generating...]
  [BioGPT Answer]
  The effect of confidence scores on the accuracy of IDRs.

  [Judge Scores]
    Judge-1 (Strict): [parse error] [API ERROR: HTTP Error 401: Unauthorized]
----------------------------------------------------------------------

[Q3] Do proline and glycine-rich regions consistently score higher disorder confidence than average?

  [Ground Truth]
  Mean proline fraction: 6.0%, mean glycine fraction: 7.0%. Both amino acids promote backbone flexibility and disrupt secondary structure, making Pro/Gly-rich regions strong predictors of intrinsic disorder.

  [BioGPT generating...]
  [BioGPT Answer]
  The effect of proline and glycine-rich regions on the prediction of protein disorder.

  [Judge Scores]
    Judge-1 (Strict): [parse error] [API ERROR: HTTP Error 401: Unauthorized]
----------------------------------------------------------------------

[Q4] Does applying a sliding window smooth out confidence scores without losing true IDR signal?

  [Ground Truth]
  General DisProt stats (13396 proteins): mean disorder=0.378, mean region length=0.0 aa.

  [BioGPT generating...]
  [BioGPT Answer]
  The effect of sliding window size on the detection of protein disorder.

  [Judge Scores]
    Judge-1 (Strict): [parse error] [API ERROR: HTTP Error 401: Unauthorized]
----------------------------------------------------------------------

[Q5] How far off are raw confidence scores from the true disorder rate in DisProt?

  [Ground Truth]
  General DisProt stats (13396 proteins): mean disorder=0.378, mean region length=0.0 aa.

  [BioGPT generating...]
  [BioGPT Answer]
  The AlphaFold score is not a reliable measure of disorder.

  [Judge Scores]
    Judge-1 (Strict): [parse error] [API ERROR: HTTP Error 401: Unauthorized]
----------------------------------------------------------------------
"""

# ── entry point ───────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser(description="Agreement score: BioGPT vs Ground Truth")
    ap.add_argument("--log",type=str,help="Path to BioGPT evaluation log")
    ap.add_argument("--gt", type=str,default=None,
                    help="Optional separate ground_truth_output.txt")
    ap.add_argument("--output",type=str,default=None,
                    help="Output path (default: agreement_score_2.txt next to script)")
    ap.add_argument("--demo",action="store_true",help="Run on 5 built-in samples")
    args=ap.parse_args()

    script_dir=os.path.dirname(os.path.abspath(__file__))
    out=args.output or os.path.join(script_dir,"agreement_score_2.txt")

    if args.demo or not args.log:
        print("[INFO] Running in DEMO mode\n")
        import tempfile
        tmp=tempfile.NamedTemporaryFile(mode="w",suffix=".txt",delete=False,encoding="utf-8")
        tmp.write(DEMO); tmp.close(); log_path=tmp.name
    else:
        log_path=args.log
        if not Path(log_path).exists():
            print(f"[ERROR] Log not found: {log_path}"); sys.exit(1)

    print(f"[INFO] Parsing: {log_path}\n")
    records=parse_log(log_path)
    if not records:
        print("[ERROR] No Q&A pairs found. Check log format."); sys.exit(1)
    print(f"[INFO] Found {len(records)} Q&A pairs\n")

    if args.gt and Path(args.gt).exists():
        print(f"[INFO] Loading GT overrides from: {args.gt}\n")
        gt_dict=parse_gt_file(args.gt)
        for rec in records:
            if rec["question_id"] in gt_dict:
                rec["ground_truth"]=gt_dict[rec["question_id"]]

    print("[INFO] Computing scores...\n")
    results=run_scoring(records)
    write_report(results,out)

    if args.demo or not args.log:
        try: os.unlink(log_path)
        except: pass

if __name__=="__main__":
    main()