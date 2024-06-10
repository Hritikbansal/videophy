import os
import csv
import json
import torch
import argparse
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import DataLoader
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from peft import LoraConfig, get_peft_model
from data_utils.xgpt3_dataset import MultiModalDataset
from utils import batchify

parser = argparse.ArgumentParser()

parser.add_argument('--input_csv', type = str, required = True, help = 'input json file')
parser.add_argument('--output_csv', type = str, help = 'output csv with scores')
parser.add_argument('--checkpoint', type = str, required = True, help = 'pretrained ckpt')
parser.add_argument('--batch_size', type = int, default = 16)

args = parser.parse_args()
softmax = nn.Softmax(dim=2)

def get_entail(logits, input_ids, tokenizer):
    logits = softmax(logits)
    token_id_yes = tokenizer.encode('Yes', add_special_tokens = False)[0]
    token_id_no  = tokenizer.encode('No', add_special_tokens = False)[0]
    entailment = []
    for j in range(len(logits)):
        for i in range(len(input_ids[j])):
            if input_ids[j][i] == tokenizer.pad_token_id: # pad token if the answer is not present
                i = i - 1
                break
            elif i == len(input_ids[j]) - 1:
                break
        score = logits[j][i][token_id_yes] / (logits[j][i][token_id_yes] + logits[j][i][token_id_no])
        entailment.append(score)
    entailment = torch.stack(entailment)
    return entailment

def get_scores(model, tokenizer, dataloader):

    with torch.no_grad():
        for index, inputs in tqdm(enumerate(dataloader)):
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if v.dtype == torch.float:
                        inputs[k] = v.bfloat16()
                    inputs[k] = inputs[k].to(model.device)
            outputs = model(pixel_values = inputs['pixel_values'], video_pixel_values = inputs['video_pixel_values'], labels = None, \
                                num_images = inputs['num_images'], num_videos = inputs['num_videos'], input_ids = inputs['input_ids'], non_padding_mask = inputs['non_padding_mask'], \
                                non_media_mask = inputs['non_media_mask'], prompt_mask = inputs['prompt_mask'])
            logits = outputs['logits']
            entail_scores = get_entail(logits, inputs['input_ids'], tokenizer)
            for m in range(len(entail_scores)):
                with open(args.output_csv, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([inputs['videopaths'][m], inputs['captions'][m], entail_scores[m].item()])
            print(f"Batch {index} Done")

def main():
    
    checkpoint = args.checkpoint

    # Processors
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    valid_data = MultiModalDataset(args.input_csv, tokenizer, processor, max_length = 256, loss_objective = 'sequential')
    dataloader = DataLoader(valid_data, batch_size=args.batch_size, pin_memory=True, collate_fn=batchify)
    
    # Instantiate model
    model = MplugOwlForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    print('Model Loaded')
    model.eval()

    get_scores(model, tokenizer, dataloader)

if __name__  == "__main__":
    main()