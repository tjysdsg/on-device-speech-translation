from kaldiio import ReadHelper
import os
import pickle
import json


def read_data():
    utt2wav = {}
    with ReadHelper(f'scp:wav.scp') as reader:
        for utt, (rate, wav) in reader:
            # print(utt, rate, wav)
            utt2wav[utt] = wav

    utt2src_text = {}
    with open(f'text', encoding='utf-8') as f:
        for line in f:
            utt, text = line.rstrip('\n').split(maxsplit=1)
            utt2src_text[utt] = text

    utt2tgt_text = {}
    with open(f'text.tc.de', encoding='utf-8') as f:
        for line in f:
            utt, text = line.rstrip('\n').split(maxsplit=1)
            utt2tgt_text[utt] = text

    return utt2wav, utt2src_text, utt2tgt_text


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Due to limitation of Kaldi-style data dir, make sure you are running this script in the data dir")

    # Load test data (520 utterances out of MUST_C_v2 TST-COMMON subset)
    utt2wav, utt2src_text, utt2tgt_text = read_data()

    with open(os.path.join(out_dir, 'utt2wav.pkl'), 'wb') as f:
        pickle.dump(utt2wav, f)

    utt2text = {}
    for utt in utt2src_text:
        utt2text[utt] = dict(
            src_text=utt2src_text[utt],
            tgt_text=utt2tgt_text[utt],
        )
    with open(os.path.join(out_dir, 'utt2text.json'), 'w', encoding='utf-8') as f:
        json.dump(utt2text, f, indent=2, ensure_ascii=False)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
