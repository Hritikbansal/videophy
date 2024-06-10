import torch
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from peft import LoraConfig, get_peft_model

model = MplugOwlForConditionalGeneration.from_pretrained(
        'local_dir/mplug-owl-llama-7b-video',
        torch_dtype=torch.bfloat16,
    ).to('cpu')

for name, param in model.named_parameters():
    param.requires_grad = False

peft_config = LoraConfig(
    target_modules=r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)', 
    inference_mode=True, 
    r=32, 
    lora_alpha=32, 
    lora_dropout=0.05
)
            
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

with open('trained_videocon_physics/checkpoint-xx/pytorch_model.bin', 'rb') as f:
    ckpt = torch.load(f, map_location = torch.device(f"cpu"))
model.load_state_dict(ckpt)
model = model.to(torch.bfloat16)
print('Model loaded!')
merged_model = model.merge_and_unload()
print('Model merged!')

with open('trained_videocon_physics_merged_output_dir/pytorch_model.bin', 'wb') as f:
    torch.save(merged_model.state_dict(), f)