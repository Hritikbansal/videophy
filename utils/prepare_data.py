import os
import argparse
import pandas as pd
from constants import *
from tqdm import tqdm
from collections import defaultdict

def main(args):

    df = pd.read_csv(args.input_csv)

    vta_test = defaultdict(list)
    physics_test = defaultdict(list)

    for j in tqdm(range(len(df))):
        videopath = df.iloc[j]['videopath']
        caption = df.iloc[j]['caption']
        vta_test['videopath'].append(videopath)
        vta_test['caption'].append(PROMPT_VTA.format(caption=caption))

        physics_test['videopath'].append(videopath)
        physics_test['caption'].append(PROMPT_PHYSICS)

    df = pd.DataFrame(vta_test)
    print(len(df))
    df.to_csv(os.path.join(args.output_folder, 'sa_testing.csv'), index = False)

    df = pd.DataFrame(physics_test)
    print(len(df))
    df.to_csv(os.path.join(args.output_folder, 'physics_testing.csv'), index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, help="input csv")
    parser.add_argument('--output_folder', type=str, help="output folder")
    args = parser.parse_args()
    main(args)