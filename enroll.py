"""
声纹注册 CLI

用法:
  python enroll.py add 张三 sample1.wav [sample2.wav ...]
  python enroll.py list
  python enroll.py remove 张三

说明:
  - 每个样本建议 5s 以上干净人声 (单一说话人)
  - 提供多个样本时取平均 embedding, 鲁棒性更好
  - 同名再次 add 会覆盖
"""

import os
import argparse

import numpy as np
import librosa

from speaker_db import SpeakerDB, extract_embedding_from_wave


MIN_DURATION = 5.0


def cmd_add(args):
    db = SpeakerDB(args.db)
    embs = []
    for path in args.samples:
        if not os.path.exists(path):
            print(f"[skip] 文件不存在: {path}")
            continue
        wav, _ = librosa.load(path, sr=16000, mono=True)
        dur = len(wav) / 16000.0
        if dur < MIN_DURATION:
            print(f"[warn] {path} 仅 {dur:.1f}s (建议 >= {MIN_DURATION:.0f}s)")
        emb = extract_embedding_from_wave(wav, sr=16000)
        embs.append(emb)
        print(f"[ok] 已处理 {path} ({dur:.1f}s)")

    if not embs:
        print("没有有效样本, 中止")
        return

    avg = np.mean(np.stack(embs, axis=0), axis=0)
    db.add(args.name, avg)
    db.save()


def cmd_list(args):
    db = SpeakerDB(args.db)
    if not db.names:
        print("(空)")
        return
    print(f"声纹库: {args.db}")
    for i, n in enumerate(db.names, 1):
        print(f"  {i}. {n}")


def cmd_remove(args):
    db = SpeakerDB(args.db)
    if db.remove(args.name):
        db.save()
        print(f"已删除: {args.name}")
    else:
        print(f"不存在: {args.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="speakers/db.npz")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="注册或覆盖")
    p_add.add_argument("name")
    p_add.add_argument("samples", nargs="+")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="列出已注册")
    p_list.set_defaults(func=cmd_list)

    p_rm = sub.add_parser("remove", help="删除")
    p_rm.add_argument("name")
    p_rm.set_defaults(func=cmd_remove)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
