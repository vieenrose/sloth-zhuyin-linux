import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
m=sys.argv[1]
t=AutoTokenizer.from_pretrained(m)
mdl=AutoModelForCausalLM.from_pretrained(m,dtype=torch.bfloat16).cuda().eval()
print("=== INSIDE conversion still works? ===")
for s,e in [("ㄋㄧˇ ㄏㄠˇ","你好"),("ㄨㄛˇ ㄗㄞˋ ㄔㄨㄥˊ ㄒㄧㄣ ㄎㄠˇ ㄌㄩˋ","我在重新考慮")]:
    ids=t(s+" →",add_special_tokens=False,return_tensors="pt").input_ids.cuda()
    o=mdl.generate(ids,max_new_tokens=len(e)+2,do_sample=False,pad_token_id=t.eos_token_id)
    print("  ",e,"->",t.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()[:len(e)])
print("=== OUTSIDE prediction (partial prefix -> convert + CONTINUE)? ===")
for s,pref in [("ㄨㄛˇ ㄗㄞˋ","我在"),("ㄐㄧㄣ ㄊㄧㄢ","今天"),("ㄒㄧㄝˋ ㄒㄧㄝ˙","謝謝"),("ㄨㄛˇ ㄒㄧㄤˇ","我想"),("ㄉㄚˋ ㄐㄧㄚ","大家")]:
    ids=t(s+" →",add_special_tokens=False,return_tensors="pt").input_ids.cuda()
    o=mdl.generate(ids,max_new_tokens=10,do_sample=False,pad_token_id=t.eos_token_id)
    print("  ",pref,"...->",t.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip())
