import sys, json, torch
import distill_student as D
from distill_student import TinyGPT, QAT
TONES="ˊˇˋ˙"; VAR=str.maketrans({"臺":"台","箇":"個"})
mdir=sys.argv[1]
ck=torch.load(f"{mdir}/student.pt",map_location="cuda"); c=ck["config"]
vocab=json.load(open(f"{mdir}/student_vocab.json",encoding="utf-8"))
id2=  {v:k for k,v in vocab.items()}
QAT["on"]=c.get("qat",False)
D.SSM_TYPE["t"]=c.get("ssm_type","mamba1")
PLM=c.get("prefix_lm",False)
m=TinyGPT(c["vocab"],c["dim"],c["depth"],c["heads"],c["kv"],c["ffn"],c.get("pattern")).cuda()
sd=ck["model"]
try:
    m.load_state_dict(sd)
except Exception:
    import re
    # backward-compat: old Block used .attn.; new attn-block uses .mix.
    rem={}
    for k,v in sd.items():
        nk=re.sub(r'\.attn\.(q|k|v|o|qn|kn)\.', r'.mix.\1.', k)
        rem[nk]=v
    m.load_state_dict(rem)
m.eval()
BOS,SEP,UNK=vocab["<bos>"],vocab["<sep>"],vocab.get("<unk>",1)
tonal={}
for line in open("phonetic_table.tsv",encoding="utf-8"):
    s,_,r=line.rstrip("\n").partition("\t")
    if r: tonal[s]=set(r)
def legal_ids(syl):
    has=any(x in TONES for x in syl); ch=tonal.get(syl) if has else None
    if ch is None:
        base="".join(x for x in syl if x not in TONES); ch=set()
        for k,v in tonal.items():
            if "".join(x for x in k if x not in TONES)==base: ch|=v
    return [vocab[x] for x in (ch or {syl}) if x in vocab]
@torch.no_grad()
def convert(syls):  # INSIDE constrained
    inp=[BOS]+[vocab.get(s,UNK) for s in syls]+[SEP]; gen=[]
    for syl in syls:
        _t=torch.tensor([inp+gen],device="cuda")
        _p=torch.tensor([len(syls)+2],device="cuda") if PLM else None
        lg=m(_t,_p)[0,-1]
        lids=legal_ids(syl)
        if not lids: gen.append(UNK); continue
        mask=torch.full_like(lg,float("-inf")); lt=torch.tensor(lids,device="cuda"); mask[lt]=lg[lt]
        gen.append(int(mask.argmax()))
    return "".join(id2.get(g,"?") for g in gen)
@torch.no_grad()
def predict(syls,extra=8):  # OUTSIDE free continuation past the span
    inp=[BOS]+[vocab.get(s,UNK) for s in syls]+[SEP]; gen=[]
    # first convert inside (constrained), then free-generate
    for syl in syls:
        _t=torch.tensor([inp+gen],device="cuda")
        _p=torch.tensor([len(syls)+2],device="cuda") if PLM else None
        lg=m(_t,_p)[0,-1]
        lids=legal_ids(syl); 
        if lids:
            mask=torch.full_like(lg,float("-inf")); lt=torch.tensor(lids,device="cuda"); mask[lt]=lg[lt]; gen.append(int(mask.argmax()))
    prev=None
    for _ in range(extra):
        _t=torch.tensor([inp+gen],device="cuda")
        _p=torch.tensor([len(syls)+2],device="cuda") if PLM else None
        lg=m(_t,_p)[0,-1]
        nx=int(lg.argmax())
        if nx==vocab["<eos>"] or nx==prev: break
        gen.append(nx); prev=nx
    return "".join(id2.get(g,"?") for g in gen)
# INSIDE accuracy on testset
n=okv=0
for line in open("../eval/testset.tsv",encoding="utf-8"):
    line=line.rstrip("\n")
    if not line or line.startswith("#"): continue
    p=line.split("\t"); exp=p[1]; g=convert(p[0].split())
    n+=1; okv+=(g.translate(VAR)==exp.translate(VAR))
print(f"INSIDE conversion (constrained): {okv}/{n} ({100*okv/n:.1f}%)")
# INSIDE toneless
nt=okt=0
for line in open("../eval/testset.tsv",encoding="utf-8"):
    line=line.rstrip("\n")
    if not line or line.startswith("#"): continue
    p=line.split("\t"); exp=p[1]; syls=["".join(x for x in y if x not in TONES) for y in p[0].split()]
    nt+=1; okt+=(convert(syls).translate(VAR)==exp.translate(VAR))
print(f"INSIDE toneless: {okt}/{nt} ({100*okt/nt:.1f}%)")
print("OUTSIDE prediction (prefix -> convert + continue):")
for s,pr in [("ㄨㄛˇ ㄗㄞˋ","我在"),("ㄐㄧㄣ ㄊㄧㄢ","今天"),("ㄉㄚˋ ㄐㄧㄚ","大家"),("ㄨㄛˇ ㄒㄧㄤˇ","我想")]:
    print("  ",pr,"->",predict(s.split()))
