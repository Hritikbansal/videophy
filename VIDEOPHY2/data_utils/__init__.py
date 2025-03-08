# from .processors.builder import build_processors
from .xgpt3_dataset import MultiModalDataset
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

def train_valid_test_datasets_provider(data_path, config, tokenizer, seq_length=1024, loss_objective = 'sequential'):
    """Build train and valid datasets."""
    print('> building train and validation datasets for mPLUG-Owl ...')
    train_ds, valid_ds = build_train_valid_test_datasets(
        input_file=data_path,  
        tokenizer=tokenizer,
        max_length=seq_length, 
        config=config, loss_objective=loss_objective)  
    print("> finished creating mPLUG-Owl datasets ...")

    return train_ds, valid_ds


def build_train_valid_test_datasets(input_file, tokenizer, max_length=80, config=None, loss_objective='sequential'):

    # train_processors = build_processors(config['train_processors'])
    # valid_processors = build_processors(config['valid_processors'])
    
    image_processor = MplugOwlImageProcessor.from_pretrained(config['pretrained_ckpt'])
    processor = MplugOwlProcessor(image_processor, tokenizer)

    assert len(input_file) == 2 # If you have files more than 2, modify code at here or merger them into train and dev
    train_ds = MultiModalDataset(input_file[0], tokenizer, processor, max_length, loss_objective = loss_objective)
    valid_ds = MultiModalDataset(input_file[1], tokenizer, processor, max_length, loss_objective = loss_objective)
    return (train_ds, valid_ds)
