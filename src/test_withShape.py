import omegaconf
import hydra

import  lightning.pytorch as pl
from lightning.pytorch .callbacks import EarlyStopping, ModelCheckpoint

import os.path as osp
from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, AddedToken
from generate_samples import GenerateTextSamplesCallback
from rdflib import Graph
import json

from codecarbon import OfflineEmissionsTracker
from pathlib import Path
def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)

    print(">>>>>>>>>>>>>>>> EXTRACTION FROM SHAPE")
    print(conf.shape_file)

    shacl_g = Graph()
    shacl_g.parse(conf.shape_file)


    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id = 0,
        #early_stopping = False,
        no_repeat_ngram_size = 0,
        dropout=conf.dropout,
        forced_bos_token_id=None,
    )
    
    print("USE FAST >>>>>>>>>>>>",conf.use_fast_tokenizer)
    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        #"add_tokens": all_vocab
    }


    print("LOAD tokenizer >>>>>>>>>>>>")

    if(conf.tokenizer_path != ""):
        print("-->",conf.tokenizer_path )
        if("t5" in conf.dataset_name):
            tokenizer = T5Tokenizer.from_pretrained(conf.tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer_path)
    else:
        if("t5" in conf.dataset_name):
            tokenizer = T5Tokenizer.from_pretrained(
                conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
                **tokenizer_kwargs
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
                **tokenizer_kwargs
            )


    print("LOAD model >>>>>>>>>>>>")
    if("gpt" in conf.model_name_or_path):
        model = AutoModelForCausalLM.from_pretrained(
            conf.model_name_or_path,
            config=config,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            conf.model_name_or_path,
            config=config,
        )

    # if not conf.finetune:
    model.resize_token_embeddings(len(tokenizer))

    # data module declaration
    pl_data_module = BasePLDataModule(conf, tokenizer, model)


    # main module declaration
    pl_module = BasePLModule(conf, config, tokenizer, model, shacl_g)

    print("LOAD checkpoint_path >>>>>>>>>>>>")
 

    print(conf.checkpoint_path)
    p = Path(conf.checkpoint_path)
    print("HEY checkpoint_path >>>>>>>>>>>>")
    trainer = pl.Trainer(
            accumulate_grad_batches=conf.gradient_acc_steps,
            gradient_clip_val=conf.gradient_clip_value,
            devices=conf.gpus,
    )

    new_model = pl_module.load_from_checkpoint_custom(checkpoint_path = p, config = config, tokenizer = tokenizer, model = model)
    print(type(new_model))
    print("HEY checkpoint_path >>>>>>>>>>>>")
 
    print("Test >>>>>>>>>>>>")

    res = trainer.test(model=new_model, datamodule=pl_data_module)
    print(res)
    print("SAVE IT !!!!!!!!!!!!!")
    path = osp.join(trainer.log_dir, "out_of_test_results.json")
    with open(path, "w") as outfile:
            json.dump(results, outfile)




@hydra.main(config_path='../conf', config_name='root')
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == '__main__':
    main()
