import os
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter


PROMPT_VTA = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Does this video entail the description: "{caption}"?
AI: {label}'''

PROMPT_PHYSICS = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Does this video follow the physical laws?
AI: {label}'''


def main(args):

    df = pd.read_csv(args.input_csv)
    
    vta_seq_data = defaultdict(list)
    physics_seq_data = defaultdict(list)
    for j in tqdm(range(len(df))):
        videopath = df.iloc[j]['videopath']
        caption = df.iloc[j]['caption']
        vta = 'Yes' if int(df.iloc[j]['sa']) else 'No'
        physics = 'Yes' if int(df.iloc[j]['pc']) else 'No'
        split = 'train'
        vta_seq_data['videopath'].append(videopath)
        vta_seq_data['caption'].append(PROMPT_VTA.format(caption=caption, label=vta))
        vta_seq_data['split'].append(split)

        physics_seq_data['videopath'].append(videopath)
        physics_seq_data['caption'].append(PROMPT_PHYSICS.format(label=physics))
        physics_seq_data['split'].append(split)

    seq_data = defaultdict(list)
    for k in vta_seq_data:
        seq_data[k] = vta_seq_data[k] + physics_seq_data[k]
    df = pd.DataFrame(seq_data)
    df.to_csv(os.path.join(args.output_folder, 'videocon_format_train.csv'), index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, help="input csv")
    parser.add_argument('--output_folder', type=str, help="output folder")
    args = parser.parse_args()
    main(args)