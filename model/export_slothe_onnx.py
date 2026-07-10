#!/usr/bin/env python3
"""Export SlothLM-E to ONNX for in-browser (Transformers.js/onnxruntime) serving.

Inputs:  syl [B,T] int64, amask [B,T] bool  ->  logits [B,T,vocab]
Dynamic batch + sequence length. Ships syl_vocab.json alongside.

  python3 model/export_slothe_onnx.py --model model/slothe --out model/slothe_onnx
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothlm_e import SlothE


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="model/slothe")
    ap.add_argument("--out", default="model/slothe_onnx")
    ap.add_argument("--opset", type=int, default=17)
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpt=torch.load(os.path.join(args.model,"slothe.pt"),map_location="cpu")
    c=ckpt["config"]
    hints_on=c.get("char_hints",False)
    model=SlothE(c["n_syl"],c["n_char"],c["dim"],c["depth"],c["heads"],c["kv"],c["ffn"],
                 embed_norm=c.get("embed_norm",False), char_hints=hints_on,
                 tie_hints=c.get("tie_hints",False))
    model.load_state_dict(ckpt["model"]); model.eval()

    syl=torch.zeros(1,6,dtype=torch.long)
    amask=torch.ones(1,6,dtype=torch.bool)
    path=os.path.join(args.out,"model.onnx")
    if hints_on:
        hints=torch.zeros(1,6,dtype=torch.long)   # 0 = no hint, else char_id+1
        torch.onnx.export(
            model,(syl,amask,hints),path,opset_version=args.opset,
            input_names=["syl","amask","hints"],output_names=["logits"],
            dynamic_axes={"syl":{0:"B",1:"T"},"amask":{0:"B",1:"T"},
                          "hints":{0:"B",1:"T"},"logits":{0:"B",1:"T"}},
        )
    else:
        torch.onnx.export(
            model,(syl,amask),path,opset_version=args.opset,
            input_names=["syl","amask"],output_names=["logits"],
            dynamic_axes={"syl":{0:"B",1:"T"},"amask":{0:"B",1:"T"},
                          "logits":{0:"B",1:"T"}},
        )
    # copy the syllable vocab + config for the serving side
    import shutil
    shutil.copy(os.path.join(args.model,"syl_vocab.json"),
                os.path.join(args.out,"syl_vocab.json"))
    json.dump(c, open(os.path.join(args.out,"config.json"),"w"))
    sz=os.path.getsize(path)/1e6
    print(f"exported {path} ({sz:.1f} MB)")


if __name__=="__main__":
    main()
