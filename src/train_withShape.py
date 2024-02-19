import omegaconf
import hydra
import torch
import os
import os.path as osp
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, AddedToken
import lightning.pytorch as pl
import json
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from kfold.datamodule import KFoldDataModule
from rdflib import Graph
#from pl_crossvalidate import KFoldTrainer
import wandb
##from kfold.trainer import KFoldTrainer
from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from generate_samples import GenerateTextSamplesCallback
from pathlib import Path

from GPUtil import showUtilization as gpu_usage

# torch.cuda.empty_cache()
import gc

from codecarbon import OfflineEmissionsTracker

# gc.collect()
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)
    
    print(">>>>>>>>>>>>>>>> EXTRACTION FROM SHAPE")
    print(conf.shape_file)

    shacl_g = Graph()
    shacl_g.parse(conf.shape_file)

    print("--------")
  
    all_vocab = []
    if(conf.add_vocab==True):
        ns_vocab=[]
        syntax_vocab=[]
        type_vocab=[]
        prop_vocab=[]

        ns_vocab=[]
        syntax_vocab=[]
        type_vocab=[]
        prop_vocab=[]
        dt_vocab=[]
        have_dt=True
        if("list" in conf.train_file):
                have_dt=False
                syntax_vocab=syntax_vocab+["[","]",",","'"]

        elif("turtleS" in conf.train_file):
                have_dt=False
                syntax_vocab=syntax_vocab+[".",",",";","\n",":","<",">"]
                # Literals
                syntax_vocab=syntax_vocab+['"']
                # Languages and datatype
                if(conf.datatype==True):
                    syntax_vocab=syntax_vocab+["^^","@"]
                # List / SET / TRIG
                syntax_vocab=syntax_vocab+["[","]","{","}","(",")"]
                # blanck nodes + a > rdf:type
                syntax_vocab=syntax_vocab+["_:"," a "]
                
        elif("tags" in conf.train_file):
            have_dt=False
            syntax_vocab=syntax_vocab+["<subj>","<rel>","<obj>","<et>"]

        elif("json-ld" in conf.train_file):
            # JSON STANDARDS
            #DELETED NOT FOUND IN JSON "'"
            # FROM https://ecma-international.org/publications-and-standards/standards/ecma-404/
            # we kept \n
            syntax_vocab=syntax_vocab+["[","]","{","}",":",",",'"',"\n"]
            # W3C JSON-LD standard
            # https://www.w3.org/TR/json-ld11/#terms
            #term
            syntax_vocab=syntax_vocab+["@type"]
            # node objects
            syntax_vocab=syntax_vocab+["@set","@list","@value","@context","@id","@included","@graph","@nest","@reverse","@index"]
            # frame objects
            syntax_vocab=syntax_vocab+["@default","@null","@none","@embed","@always","@once","@explicit","@default","@omitDefault","@requiereAll"]
            # values objects
            syntax_vocab=syntax_vocab+["@value","@language","@direction"]
            # propery_based Index Maps        
            syntax_vocab=syntax_vocab+["@container"]
            # included block
            syntax_vocab=syntax_vocab+["@included"]        
            # context def
            syntax_vocab=syntax_vocab+["@import","@base","@propagate","@protected","@version","@vocab"]    
            # other
            syntax_vocab=syntax_vocab+["@json","@prefix"]    
            
        elif("ntriples" in conf.train_file):
      
            # FOLLOWING 
            # https://www.w3.org/TR/n-triples/
            
            syntax_vocab=syntax_vocab+["<",">",".","\n"]
            # quote
            syntax_vocab=syntax_vocab+['"']
            # datatypes 
            syntax_vocab=syntax_vocab+["^^","@"]
            # blank nodes
            syntax_vocab=syntax_vocab+["_:"]

        elif("turtle" in conf.train_file):
            # FOLLOWING  https://www.w3.org/TR/rdf12-turtle/#language-features
            syntax_vocab=syntax_vocab+[".",",",";","\n",":","<",">"]
            # IRI ref
            syntax_vocab=syntax_vocab+["@base","@prefix"]
            # Literals
            syntax_vocab=syntax_vocab+["'",'"',"'''",'"""']
            # Languages and datatype
            syntax_vocab=syntax_vocab+["^^","@"]
            # List / SET / TRIG
            syntax_vocab=syntax_vocab+["[","]","{","}","(",")"]
            # blanck nodes + a > rdf:type
            syntax_vocab=syntax_vocab+["_:"," a "]

        elif("xml" in conf.train_file):
            #https://www.w3.org/TR/rdf-syntax-grammar/
            # "/>" not added could broke URI
            syntax_vocab=syntax_vocab+[":","</","<",">"," />","\n","="]
            # Head and comment
            syntax_vocab=syntax_vocab+["<?","?>","<!--","--!>"]
            # CORE SYNTHAX TERM
            syntax_vocab=syntax_vocab+["rdf:RDF","rdf:ID","rdf:type","rdf:about","rdf:parseType","rdf:resource","rdf:nodeID","rdf:datatype"]
            # LIST
            syntax_vocab=syntax_vocab+["rdf:li","rdf:_"]
            # XML 
            syntax_vocab=syntax_vocab+["xml:lang","xml:base","xmlns:"]


        all_vocab=list(set(prop_vocab+type_vocab+ns_vocab+syntax_vocab+dt_vocab))

        all_vocab=list(set(syntax_vocab))

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

    if("t5" in conf.dataset_name):
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True
        tokenizer = T5Tokenizer.from_pretrained(
            conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
            **tokenizer_kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
            **tokenizer_kwargs
        )

    print("============+>",conf.train_file)
  
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

    #if not conf.finetune:
    print("SIZE BEFORE >",len(tokenizer))

    if("t5" in conf.dataset_name and conf.add_vocab==True):
        # print("ADDDD T5 SPECIAL TOKENS")
        # if("[" not in all_vocab):
        #     all_vocab.append("[")
        # if("]" not in all_vocab):
        #     all_vocab.append("]")
        if("<s>" in all_vocab):
            all_vocab.remove("<s>") 
        if("["  in all_vocab):
            all_vocab.remove("[")
        if("]"  in all_vocab):
            all_vocab.remove("]")
        if("\n" in all_vocab):
            all_vocab.remove('\n')
            
        tokenizer.add_tokens(AddedToken("[", normalized=False))
        tokenizer.add_tokens(AddedToken("]", normalized=False))
        tokenizer.add_tokens(AddedToken("\n", normalized=False))
        tokenizer.add_tokens(AddedToken("<s>", normalized=False))
        # tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("<s>")]})
        #for voc in all_vocab:
        #    tokenizer.add_tokens(AddedToken(voc, normalized=False))

            #https://stackoverflow.com/questions/72214408/why-does-huggingface-t5-tokenizer-ignore-some-of-the-whitespaces
    tokenizer.add_tokens(all_vocab)

    ########### TEST IT
    print("SAVE TOKENIZE VOCAB >")
    tokenizer.save_pretrained( f'experiments/experiments/{conf.model_name}_tokenizer')
    print("SAVED TOKENIZE VOCAB !")


    print("SIZE AFTER >",len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    model_name=conf.config_name.split("/")[-1]

    ### ADDED FOR LOCAL
    #model.to(torch.device('cpu'))
        # data module declaration
    print("TRAIN FILE")
    print(conf.train_file)
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    train_dataloader=pl_data_module.train_dataloader()
    val_dataloader=pl_data_module.val_dataloader()
    

    project=conf.project
    group=conf.syntax_name+"_"+model_name



    tracker = OfflineEmissionsTracker(country_iso_code="FRA")

    if(conf.nb_folds==0):
        pl_module = BasePLModule(conf, config, tokenizer, model, shacl_g)
        wandblogger = WandbLogger(project = conf.dataset_name.split('/')[-1].replace('.py', ''), name = group,group=group)
        callbacks_store = []
        if conf.apply_early_stopping:
            callbacks_store.append(
                EarlyStopping(
                    monitor=conf.monitor_var,
                    mode=conf.monitor_var_mode,
                    patience=conf.patience
                )
            )


        callbacks_store.append(
            ModelCheckpoint(
                monitor=conf.monitor_var,
                # monitor=None,
                dirpath=f'experiments/{conf.model_name}',
                save_top_k=conf.save_top_k,
                save_last=True,
                mode=conf.monitor_var_mode
            )
        )

        callbacks_store.append(GenerateTextSamplesCallback(conf.samples_interval))
        callbacks_store.append(LearningRateMonitor(logging_interval='step'))

        #accelerator = CPUAccelerator()
        trainer = pl.Trainer(
           # accelerator="cpu",#### ADDed
            #accelerator="cpu",
           # gpus=conf.gpus,
            devices=conf.gpus,
            accumulate_grad_batches=conf.gradient_acc_steps,
            gradient_clip_val=conf.gradient_clip_value,
            val_check_interval=conf.val_check_interval,
            callbacks=callbacks_store,
            max_steps=conf.max_steps,
            # max_steps=total_steps,
            precision=conf.precision,
            #amp_level=conf.amp_level,
            logger=wandblogger,
            #ckpt_path=conf.checkpoint_path,
            limit_val_batches=conf.val_percent_check
        )


        tracker.start()
        trainer.fit(pl_module, datamodule=pl_data_module)
        tracker.stop()
        results, paths = [], []
        res = trainer.test(model=pl_module, datamodule=pl_data_module, verbose=False)
        results.append(res)
        wandb.finish()
        path = osp.join(trainer.log_dir, conf.syntax_name+"_"+model_name+"_withoutCVresults.json")
        with open(path, "w") as outfile:
            json.dump(results, outfile)

    elif(conf.nb_folds > 0):

        print("===================> CREATE DIR")
        test_save={"test":"test"}
        with open( 'test_save.json', "w") as outfile:
            json.dump(test_save, outfile)

        kfold_data_module=KFoldDataModule(
                num_folds=conf.nb_folds,
                shuffle=False,
                stratified=False,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader,
                #datamodule=pl_data_module,
            )

        models = [BasePLModule(conf, config, tokenizer, model, shacl_g) for _ in range(conf.nb_folds)]

        results, paths = [], []
        all_data_exp = {}

        for i in range(conf.nb_folds):
            rank_zero_info(f"===== Starting fold {i+1}/{conf.nb_folds} =====")

           

            wandblogger = WandbLogger(project = project, name = group+"_"+str(i),group=group)
            all_data_exp["fold"+str(i)]={}
            callbacks_store = []
        
            if conf.apply_early_stopping:
                callbacks_store.append(
                    EarlyStopping(
                        monitor=conf.monitor_var,
                        mode=conf.monitor_var_mode,
                        patience=conf.patience
                    )
                )

            checkpoint_callback=ModelCheckpoint(
                    monitor=conf.monitor_var,
                    # monitor=None,
                    dirpath='experiments/'+project+'/',
                    filename=group+"_"+str(i)+'-{epoch:02d}-{val_loss:.2f}',
                    save_top_k=conf.save_top_k,
                    verbose=True,
                    save_last=True,
                    mode=conf.monitor_var_mode
                )
            callbacks_store.append(checkpoint_callback)

            callbacks_store.append(GenerateTextSamplesCallback(conf.samples_interval))
            callbacks_store.append(LearningRateMonitor(logging_interval='step'))

            kfold_data_module.fold_index=i
            trainer = pl.Trainer(
               # accelerator="cpu",#### ADDed
                #accelerator="cpu",
               # gpus=conf.gpus,
                devices=conf.gpus,
                accumulate_grad_batches=conf.gradient_acc_steps,
                gradient_clip_val=conf.gradient_clip_value,
                val_check_interval=conf.val_check_interval,
                callbacks=callbacks_store,
                max_steps=conf.max_steps,
                # max_steps=total_steps,
                precision=conf.precision,
                #amp_level=conf.amp_level,
                logger=wandblogger,
                #ckpt_path=conf.checkpoint_path,
                limit_val_batches=conf.val_percent_check
            )
            print(">>>>>>>>>>>>>>>>>> TRAIN")
            tracker.start()
            res=trainer.fit(models[i], datamodule=kfold_data_module)
            tracker.stop()
            

            all_data_exp["fold"+str(i)]["train_data"]=res
            all_data_exp["fold"+str(i)]["carbon_data"]={}

            all_data_exp["fold"+str(i)]["carbon_data"]["train_emissions"] = tracker.final_emissions_data.emissions
            all_data_exp["fold"+str(i)]["carbon_data"]["train_energy_consumed"] = tracker.final_emissions_data.energy_consumed

                      #trainer.save_checkpoint(fold_path)
            #state=trainer.state
          

            print("=====================BEST MODEL")

            all_data_exp["fold"+str(i)]["best_model_path"]=checkpoint_callback.best_model_path
            # with open( f'experiments/{conf.model_name}/{conf.nb_folds}_fold_valid_test_data.json', "w") as outfile:
            # json.dump(test_data, outfile)
            #print("-------------->",state)


            #checkpoint_callback.best_model_path

            print(">>>>>>>>>>>>>>>>>> TEST")
            tracker.start()

            # p_bm = Path(checkpoint_callback.best_model_path)
            # pl_module = BasePLModule(conf, config, tokenizer, model, shacl_g)
            # trainer = pl.Trainer(
            #         accumulate_grad_batches=conf.gradient_acc_steps,
            #         gradient_clip_val=conf.gradient_clip_value,
            #         devices=conf.gpus,
            # )
            # best_model = pl_module.load_from_checkpoint_custom(checkpoint_path = p_bm, config = config, tokenizer = tokenizer, model = models[i])
            #res = trainer.test(model=best_model, datamodule=kfold_data_module)

            res=trainer.test(models[i], datamodule=kfold_data_module)
            tracker.stop()
            all_data_exp["fold"+str(i)]["test_data_last_step"]=res
            all_data_exp["fold"+str(i)]["carbon_data"]["test_emissions"] = tracker.final_emissions_data.emissions
            all_data_exp["fold"+str(i)]["carbon_data"]["test_energy_consumed"] = tracker.final_emissions_data.energy_consumed

            wandb.finish()

        print("SAVE DATA")

        # path = osp.join(trainer.log_dir, conf.syntax_name+"_"+model_name+"_"+conf.nb_folds+"_fold_valid_test_data.json")
        with open( 'all_data.json', "w") as outfile:
            json.dump(all_data_exp, outfile)

@hydra.main(config_path='../conf', config_name='root')
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == '__main__':
   
   # print(torch.cuda.list_gpu_processes())
    #gc.collect()
    #torch.cuda.empty_cache()
    main()
