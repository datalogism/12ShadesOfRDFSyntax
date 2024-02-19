from typing import Any
import nltk
nltk.download('punkt') 
import json
import lightning.pytorch as pl
import torch
import re
import numpy as np
import pandas as pd
from rdflib import Graph
from typing import Optional
from score import  re_score_withShape
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from scheduler import get_inverse_square_root_schedule_with_warmup
from datasets import load_dataset, load_metric
from torch.nn.utils.rnn import pad_sequence
from utils import BartTripletHead, shift_tokens_left, extract_triplets_typed, extract_triplets, extract_str_triplets

import lmppl
from lightning.pytorch.accelerators import CUDAAccelerator, MPSAccelerator
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,Type
)
from pathlib import Path
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from typing_extensions import Self
from lightning.pytorch.utilities.migration import pl_legacy_patch
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint

from lightning.pytorch.core.saving import  _load_state

from math import exp
from typing import List

from tqdm import tqdm


PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
    "inverse_square_root": get_inverse_square_root_schedule_with_warmup
}


from lightning.pytorch.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml
  
def _default_map_location(storage: "UntypedStorage", location: str) -> Optional["UntypedStorage"]:
    if (
        location.startswith("mps")
        and not MPSAccelerator.is_available()
        or location.startswith("cuda")
        and not CUDAAccelerator.is_available()
    ):
        return storage.cpu()
    return None  # default behavior by `torch.load()`

def _load_from_checkpoint_custom(
    obj,
    checkpoint_path: Union[_PATH, IO],
    map_location: _MAP_LOCATION_TYPE = None,
    hparams_file: Optional[_PATH] = None,
    strict: Optional[bool] = None,
    **kwargs: Any,
) -> Union["pl.LightningModule", "pl.LightningDataModule"]:


    map_location = map_location or _default_map_location
    
    with pl_legacy_patch():
        print(checkpoint_path)
        print(map_location)
        checkpoint = pl_load(checkpoint_path, map_location=map_location)

    # convert legacy checkpoints to the new format
    checkpoint = _pl_migrate_checkpoint(
        checkpoint, checkpoint_path=(checkpoint_path if isinstance(checkpoint_path, (str, Path)) else None)
    )


    if hparams_file is not None:
        extension = str(hparams_file).split(".")[-1]
        if extension.lower() == "csv":
            hparams = load_hparams_from_tags_csv(hparams_file)
        elif extension.lower() in ("yml", "yaml"):
            hparams = load_hparams_from_yaml(hparams_file)
        else:
            raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

        # overwrite hparams by the given file
        checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

    # TODO: make this a migration:
    # for past checkpoint need to add the new key
    checkpoint.setdefault(obj.CHECKPOINT_HYPER_PARAMS_KEY, {})
    # override the hparams with values that were passed in

    #### DELETE IT
    #checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)

    if issubclass(type(obj), pl.LightningDataModule):
        return _load_state(type(obj), checkpoint, **kwargs)
    if issubclass(type(obj), pl.LightningModule):
        model = _load_state(type(obj), checkpoint, strict=strict, **kwargs)

    state_dict = checkpoint["state_dict"]
    if not state_dict:
        rank_zero_warn(f"The state dict in {checkpoint_path!r} contains no parameters.")
        return model

    device = next((t for t in state_dict.values() if isinstance(t, torch.Tensor)), torch.tensor(0)).device
    assert isinstance(model, pl.LightningModule)
   
    return model.to(device)

    raise NotImplementedError(f"Unsupported {type(obj)}")
    
def cleanDecoded(list_labels):
    decoded_labels_clean = []
    for label in list_labels:
        clean=label.replace("<s>","").replace("</s>","")
        clean=label.replace('=" ','="').replace(" : ",":").replace(" :",":").replace(": ",":").replace(' #','#').replace("< ","<").replace("</ ","</").replace("= ","=").replace('<? ','<?').replace('# ','#').replace(' >','>').replace(" <","<").replace(' =','=').replace('/ ','/').replace('/ ','/')
        # xml
        clean=clean.replace(' "/>','"/>').replace(' ">','">')
        # turtle
        #.replace(' "','"')
        clean=clean.replace(":<",": <").replace(" ^","^").replace("^ ","^")
        # ntriples
        clean=clean.replace("><","> <")
        #clean=label.strip().replace("<s>","").replace("</s>","")
        decoded_labels_clean.append(clean)        
    return decoded_labels_clean

def getSyntaxConf(dataset_name):
    datatype=None
    inline_mode=None
    facto=None

    if("json-ld" in dataset_name):
        syntax="json-ld"
        datatype=True
    elif("ntriples" in dataset_name):
        syntax="ntriples"
    elif("xml" in dataset_name):
        syntax="xml"  
        datatype=True
    elif("tags" in dataset_name):
        syntax="tags"
    elif("list" in dataset_name):
        syntax="list"
    elif("turtleS" in dataset_name):
        syntax="turtleS"
    elif("turtle" in dataset_name):
        syntax="turtle"
        datatype=True

    if( dataset_name or "turtleS" in dataset_name):
        if("0datatype" in dataset_name):
            datatype=False
        if("1datatype" in dataset_name):
            datatype=True
        if("0inLine" in dataset_name):
            inline_mode=False
        if("1inLine" in dataset_name):
            inline_mode=True
        if("0facto" in dataset_name):
            facto=False
        if("1facto" in dataset_name):
            facto=True
    if("list" in dataset_name or "tags" in dataset_name):
        if("facto" in dataset_name):
            facto=True
        else:
            facto=False
    syntax_conf={"facto":facto,"inLine":inline_mode,"datatype":datatype}

    return syntax, syntax_conf

class BasePLModule(pl.LightningModule):

    def __init__(self, conf, config: AutoConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, shapes: Optional[Graph] = None,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.tokenizer = tokenizer
        self.shapes = shapes
        print("BasePLModule SHAPES >", shapes)
        self.model = model
        self.config = config
        self.conf = conf

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if self.hparams.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        else:
            # dynamically import label_smoothed_nll_loss
            from utils import label_smoothed_nll_loss

            self.loss_fn = label_smoothed_nll_loss

        self.validation_step_outputs = []  ## ADDED
      #  self.validation_step_inputs = []
        self.testing_step_outputs = []
   #     self.testing_step_inputs = []
        #self.training_step_outputs = []

        #print("> INIT : len val step",len(self.validation_step_outputs))
   

    def load_from_checkpoint_custom(
            self,
            checkpoint_path: _PATH,
            map_location: _MAP_LOCATION_TYPE = None,
            hparams_file: Optional[_PATH] = None,
            strict: bool = True,
            **kwargs: Any,
        ) -> Self:
            loaded = _load_from_checkpoint_custom(
                self,
                checkpoint_path,
                map_location,
                hparams_file,
                strict,
                **kwargs,
            )
            loaded_model=cast(Self, loaded)
            loaded_model.shapes=self.shapes
            return loaded_model
            
    def forward(self, inputs, labels, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        print("FOWARD")
        if self.hparams.label_smoothing == 0:
            if self.hparams is not None and self.hparams.ignore_pad_token_for_loss:
                # force training to ignore pad token
                outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True)
                logits = outputs['logits']
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))#, ignore_index=self.config.pad_token_id)
            else:
                # compute usual loss via models
                outputs = self.model(**inputs, labels=labels, use_cache=False, return_dict = True, output_hidden_states=True)
                loss = outputs['loss']
                logits = outputs['logits']
        else:
            # compute label smoothed loss
            outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True)
            logits = outputs['logits']
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            # labels = torch.where(labels != -100, labels, self.config.pad_token_id)
            labels.masked_fill_(labels == -100, self.config.pad_token_id)
            loss, _ = self.loss_fn(lprobs, labels, self.hparams.label_smoothing, ignore_index=self.config.pad_token_id)
        output_dict = {'loss': loss, 'logits': logits}
        print("FOWARD end")
        #print(output_dict)
        # return loss, logits
        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:

        print("TRAINING STEP")
        ## torch.cuda.OutOfMemoryError
        #print("training_step")
        #print(batch_idx)
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        # TEST WITHOUT
        labels = shift_tokens_left(labels, -100)
        forward_output = self.forward(batch, labels)

        self.log('loss', forward_output['loss'])

        #self.training_step_outputs.append(forward_output['loss'])
        batch["labels"] = labels_original

        #print("training_step middle")
        if 'loss_aux' in forward_output:
            self.log('loss_classifier', forward_output['loss_aux'])
            return forward_output['loss'] + forward_output['loss_aux']
        return forward_output['loss']# + forward_output['loss_aux']
    
    # def on_train_epoch_end(self):
    #     # do something with all training_step outputs, for example:
    #     epoch_mean = torch.stack(self.training_step_outputs).mean()
    #     self.log("training_epoch_mean", epoch_mean)
    #     # free up the memory
    #     self.training_step_outputs.clear()

    def _pad_tensors_to_max_len(self, tensor, max_length):

        print("_pad_tensors_to_max_len")
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

  
    def generate_triples(self, batch, labels, ) -> None:
        print("generate_triples")
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache = True,
            **gen_kwargs,
        )

        ## ADDED IT FOR PROBA 
        # transition_scores =  self.model.compute_transition_scores(
        #     generated_tokens.sequences, generated_tokens.scores, normalize_logits=True
        # )

        ### I ADDED HERE TRUNCATION
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, truncation=True, spaces_between_special_tokens = True)
        decoded_labels = self.tokenizer.batch_decode(torch.where(labels != -100, labels, self.config.pad_token_id), skip_special_tokens=True, truncation=True, spaces_between_special_tokens = True)
        
        decoded_preds = cleanDecoded(decoded_preds)
        decoded_labels = cleanDecoded(decoded_labels)
       

        #if 'dbpedia' in self.hparams.dataset_name.split('/')[-1]:
        return [ decoded_preds,  decoded_labels]
            #return [extract_str_triplets(rel) for rel in decoded_preds], [extract_str_triplets(rel) for rel in decoded_labels]

        # elif self.hparams.dataset_name.split('/')[-1] == 'conll04_typed.py':
        #     return [extract_triplets_typed(rel) for rel in decoded_preds], [extract_triplets_typed(rel) for rel in decoded_labels]
        # elif self.hparams.dataset_name.split('/')[-1] == 'nyt_typed.py':
        #     return [extract_triplets_typed(rel, {'<loc>': 'LOCATION', '<org>': 'ORGANIZATION', '<per>': 'PERSON'}) for rel in decoded_preds], [extract_triplets_typed(rel, {'<loc>': 'LOCATION', '<org>': 'ORGANIZATION', '<per>': 'PERSON'}) for rel in decoded_labels]
        # elif self.hparams.dataset_name.split('/')[-1] == 'docred_typed.py':
        #     return [extract_triplets_typed(rel, {'<loc>': 'LOC', '<misc>': 'MISC', '<per>': 'PER', '<num>': 'NUM', '<time>': 'TIME', '<org>': 'ORG'}) for rel in decoded_preds], [extract_triplets_typed(rel, {'<loc>': 'LOC', '<misc>': 'MISC', '<per>': 'PER', '<num>': 'NUM', '<time>': 'TIME', '<org>': 'ORG'}) for rel in decoded_labels]
        
        # return [extract_triplets(rel) for rel in decoded_preds], [extract_triplets(rel) for rel in decoded_labels]

    #(self, # model,         # tokenizer,         batch,         labels,     )
    def generate_samples(self,  labels,   ) -> None:
        # labels = batch.pop("labels")
        # pick the last batch and logits
        # x, y = batch

        print("generate_samples")
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }
        #print(labels)
        relation_start = labels == 50265
        print("--------------relation_start")
        print(relation_start)
        relation_start = torch.roll(relation_start, 1, 1)
        relation_start = torch.cumsum(relation_start, dim=1)
        labels_decoder = torch.where(relation_start == 1, self.tokenizer.pad_token_id, labels)
        labels_decoder[:,-1] = 2
        labels_decoder = torch.roll(labels_decoder, 1, 1)

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device),
            use_cache = False,
            **gen_kwargs,
        )
        print(generated_tokens)
        relation_start = generated_tokens == 50265
        relation_start = torch.roll(relation_start, 2, 1)

        decoded_preds = self.tokenizer.batch_decode(generated_tokens[relation_start==1], skip_special_tokens=True, truncation=True, spaces_between_special_tokens = True)
        
        print(decoded_preds)
        return [rel.strip() for rel in decoded_preds]
    #self,         # model,         # tokenizer,         batch,         labels)
    def forward_samples(self, batch, labels, ) -> None:

        print("forward_samples")
        relation_start = labels == 50265
        relation_start = torch.roll(relation_start, 2, 1)
        labels = torch.where(torch.cumsum(relation_start, dim=1) == 1, self.tokenizer.pad_token_id, labels)
        labels[:,-1] = 0
        labels = torch.roll(labels, 1, 1)
        min_padding = min(torch.sum((labels == 1).int(), 1))
        labels_decoder = torch.randint(60000,(labels.shape[0], labels.shape[1] - min_padding))
        labels_decoder = labels[:, :-min_padding]
        outputs = self.model(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device),
            return_dict=True,
        )
        next_token_logits = outputs.logits[relation_start[:,: -min_padding]==1]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        decoded_preds = self.tokenizer.batch_decode(next_tokens, skip_special_tokens=True, truncation=True, spaces_between_special_tokens = True)

        return [rel.strip() for rel in decoded_preds]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        print("VALIDATION STEP")
        #print(batch_idx)
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        #print("validation_step")
        #print(batch_idx)
        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = batch.pop("labels")

        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        # TEST WITHOUT        
        labels = shift_tokens_left(labels, -100)
        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output['loss'] = forward_output['loss'].mean().detach()

        if self.hparams.prediction_loss_only:
            self.log('val_loss', forward_output['loss'])
            return

        forward_output['logits'] = generated_tokens.detach() if self.hparams.predict_with_generate else forward_output['logits'].detach()

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output['labels'] = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            forward_output['labels'] = labels


        # print("PERPLEXITY")
        # logits = forward_output['logits'].detach().cpu()
        # valid_length = (forward_output["labels"] != PAD_TOKEN_LABEL_ID).sum(dim=-1)

        # #loss = self.loss_fn(logits.view(-1,  len(self.tokenizer)), forward_output["labels"].view(-1))
        # #loss = loss.view(len(logits), -1)
        # loss = torch.sum(forward_output['loss'], -1) / valid_length
        # loss_list = loss.cpu().tolist()
        # # conversion to perplexity
        # ppl = np.mean([exp(i) for i in loss_list])
        # print(ppl)

        # self.log('val_perplexity', ppl)

        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(forward_output['logits'].detach().cpu(), forward_output['labels'].detach().cpu())
        else:
            metrics = {}

        metrics['val_loss'] = forward_output['loss']

        for key in sorted(metrics.keys()):
            self.log(key, metrics[key])

        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)

        #ADDED
        outputs['predictions'] = cleanDecoded(outputs['predictions'])
        outputs['labels'] = cleanDecoded(outputs['labels'])

        self.validation_step_outputs.append(outputs)
        #self.validation_step_inputs=self.validation_step_inputs+list(batch["input_ids"].detach().cpu())
        #print("LEN val outputs>",len(self.validation_step_outputs))
        return outputs

    def test_step(self, batch: dict, batch_idx: int) -> None:

        print("TEST STEP")

        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        ####### TEST WITHOUT         
        labels = shift_tokens_left(labels, -100)
        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output['loss'] = forward_output['loss'].mean().detach()
        if self.hparams.prediction_loss_only:
            self.log('test_loss', forward_output['loss'])
            return

        forward_output['logits'] = generated_tokens.detach() if self.hparams.predict_with_generate else forward_output['logits'].detach()

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output['labels'] = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            forward_output['labels'] = labels

        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(forward_output['logits'].detach().cpu(), forward_output['labels'].detach().cpu())
        else:
            metrics = {}
        metrics['test_loss'] = forward_output['loss']
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key], prog_bar=True)


        ##### ADD PERPLEXITY HERE


        #if self.hparams.finetune:
        #    return {'predictions': self.forward_samples(batch, labels)}
        #else:
            
        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        

        #ADDED
        outputs['predictions'] = cleanDecoded(outputs['predictions'])
        outputs['labels'] = cleanDecoded(outputs['labels'])
        #self.testing_step_outputs.append(forward_output['loss'])

        self.testing_step_outputs.append(outputs)
      #  self.testing_step_inputs.append(batch["input_ids"])
        #print("LEN test outputs>",len(self.testing_step_outputs))
        return outputs

    def on_validation_epoch_end(self) -> Any:
        print(">>>>>>>>>>>>validation_epoch_end")
        #print(self.validation_step_outputs)

        output = self.validation_step_outputs
       # inputs = self.validation_step_inputs
        dataset_name=self.hparams.train_file
        syntax, syntax_conf= getSyntaxConf(dataset_name)

        pred_list=[item for pred in output for item in pred['predictions']]
        gold_list=[item for pred in output for item in pred['labels']]
        
        #if(self.shapes):
        scores, part_parsed, part_subj_ok, part_valid = re_score_withShape(pred_list, gold_list, syntax, syntax_conf, self.shapes)
        # else:
        #     scores = re_score4(pred_list, gold_list ,syntax)
       # print(inputs)
        #val_Perplexity_gold=np.mean(self.get_perplexity(input_texts=inputs, output_texts=gold_list))
         #val_Perplexity_pred=np.mean(self.get_perplexity(input_texts=inputs, output_texts=pred_list))

        # self.log('val_Perplexity_gold', float(val_Perplexity_gold))
        # self.log('val_Perplexity_pred', float(val_Perplexity_pred))

        self.log('val_F1_micro', float(scores["ALL"]["f1"]))
        self.log('val_F1_macro', scores["ALL"]["Macro_f1"])
        self.log('val_prec_macro', scores["ALL"]["Macro_p"])
        self.log('val_recall_macro', scores["ALL"]["Macro_r"])
        self.log('val_dist_edit', scores["ALL"]["AVG_EditDist"])
        if scores["ALL"]["BLEU"]:
            self.log('val_BLEU', float(scores["ALL"]["BLEU"]))
        
        if(self.shapes):
            self.log('val_part_parsed', part_parsed)
            self.log('val_part_subj_ok', part_subj_ok)
            self.log('val_part_valid', part_valid)

        self.validation_step_outputs.clear()
        #self.validation_step_inputs.clear()

    def on_test_epoch_end(self) -> Any:
        #test_epoch_end(self, output: dict)

        output = self.testing_step_outputs
      #  inputs = self.testing_step_inputs
        dataset_name=self.hparams.train_file
        syntax, syntax_conf= getSyntaxConf(dataset_name)

        pred_list=[item for pred in output for item in pred['predictions']]
        gold_list=[item for pred in output for item in pred['labels']]
        
        # if(self.shapes):

        scores, part_parsed, part_subj_ok, part_valid = re_score_withShape(pred_list, gold_list, syntax, syntax_conf, self.shapes)
        # else:
        #     scores = re_score4(pred_list, gold_list ,syntax)

        # scorer = lmppl.EncoderDecoderLM(self.conf.model_name)
        # val_Perplexity_gold=np.mean(scorer.get_perplexity(input_texts=self.validation_step_inputs, output_texts=gold_list))
        # val_Perplexity_pred=np.mean(scorer.get_perplexity(input_texts=self.validation_step_inputs, output_texts=pred_list))

        # self.log('val_Perplexity_gold', float(val_Perplexity_gold))
        # self.log('val_Perplexity_pred', float(val_Perplexity_pred))

        self.log('test_F1_micro', float(scores["ALL"]["f1"]))
        self.log('test_F1_macro', scores["ALL"]["Macro_f1"])
        self.log('test_prec_macro', scores["ALL"]["Macro_p"])
        self.log('test_recall_macro', scores["ALL"]["Macro_r"])
        self.log('test_prec_micro', scores["ALL"]["Micro_p"])
        self.log('test_recall_micro', scores["ALL"]["Micro_r"])
        self.log('test_dist_edit', scores["ALL"]["AVG_EditDist"])
        if(scores["ALL"]["BLEU"]):
            self.log('test_BLEU', float(scores["ALL"]["BLEU"]))

        if(self.shapes):
            self.log('test_part_parsed', part_parsed)
            self.log('test_part_subj_ok', part_subj_ok)
            self.log('test_part_valid', part_valid)
        self.testing_step_outputs.clear()
#        self.testing_step_inputs.clear()

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.hparams.adafactor else AdamW
        if self.hparams.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
                "eps": self.hparams.adam_epsilon,
            }
        
        optimizer_kwargs["lr"] = self.hparams.learning_rate

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = self._get_lr_scheduler(self.hparams.max_steps, optimizer)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_lr_scheduler(self, num_training_steps, optimizer):
        print("_get_lr_scheduler")
        schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = schedule_func(optimizer)
        elif self.hparams.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "inverse_square_root":
            # args = {"warmup_updates": self.hparams.warmup_steps, "lr": [self.hparams.learning_rate]}
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        else:
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler


    def compute_metrics(self, preds, labels):
        print(">>>>>>>>>>>>compute_metrics")
        metric_name = "rouge" # if self.hparams.task.startswith("summarization") else "sacrebleu"
        metric = load_metric(metric_name)
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            if metric_name == "rouge":
                preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
                labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
            else:  # sacrebleu  
                labels = [[label] for label in labels]

            return preds, labels
        # preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, truncation=True, spaces_between_special_tokens = True)
        if self.hparams.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, truncation=True, spaces_between_special_tokens = True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if metric_name == "rouge":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}
            
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX METRICS :")
        print(result)
        return result

