import os
import numpy as np
import yaml
import evaluate
from datasets import Dataset, load_from_disk
import wandb
from transformers import logging
import random
import json
from transformers import AdamW
from torch.optim.lr_scheduler import StepLR
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    T5Config,
    T5ForConditionalGeneration

)
import queue
from transformers.models.gptj import GPTJConfig, GPTJForCausalLM
from data_process import DataUtilSeq
from concurrent.futures import ThreadPoolExecutor, Future
from transformers import TrainerCallback, TrainerState, TrainerControl

logging.set_verbosity_info()
        
class AsyncDatasetCallback(TrainerCallback):
    """Enhanced callback that handles async data reloading"""
    def __init__(self, infill_finetune_instance, num_prefetch=1):
        super().__init__()
        self.infill_finetune = infill_finetune_instance
        self.num_prefetch = num_prefetch
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future_datasets = queue.Queue(maxsize=num_prefetch)
        self._initialize_prefetch()
        self.prefetch_started = False  # Added initialization here

    def _initialize_prefetch(self):
        """Start initial prefetching of datasets"""
        for _ in range(self.num_prefetch):
            future = self.executor.submit(self.infill_finetune.load_training_dataset)
            self.future_datasets.put(future)
            
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Start prefetching when training begins"""
        if not self.prefetch_started:
            # Start prefetching for next epochs while first epoch trains
            self.prefetch_started = True
            
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Update dataset at the beginning of each epoch"""
        # Get the next pre-processed dataset
        
        if state.epoch==0:
            return
        if not self.future_datasets.empty():
            future = self.future_datasets.get()            
            try:
                new_dataset = future.result()
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                return

            future = self.executor.submit(self.infill_finetune.load_training_dataset)
            self.future_datasets.put(future)
            self.infill_finetune.trainer.train_dataset = new_dataset
            
    def __del__(self):
        """Cleanup executor on deletion"""
        self.executor.shutdown(wait=False)
        
class InfillFinetune():
    
    def __init__(self, model_name, model_config_path=None):
        self.model_name = model_name
        self.max_src = 512
        self.max_trg = 128
        self.model_config_path = model_config_path
        self.cache_dir = None
        self.training_src = None
        self.training_trg = None
        self.tokenizer = self.load_tokenizer()
        self.trainer = None
        
    def load_tokenizer(self, model_name=None):
        if model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left',)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, truncation_side='left',)
        special_tokens = {
            "additional_special_tokens": ["<span>", "</span>"] + [f"<len{i}>" for i in range(1, 6)]
        }
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.decoder_start_token_id = 0
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        return tokenizer
    
    def load_casual_model(self, model_name=None):
        
        if model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = model.to('cuda')
        return model
    
    def load_seq_model(self, model_name=None):
        def initialize_weights_with_xavier(model):
        # Apply Xavier initialization
            for _, param in model.named_parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
        config = T5Config(
            decoder_start_token_id = self.tokenizer.pad_token_id,
            # num_layers = 3
        )
        model = T5ForConditionalGeneration(config)
        model.init_weights()
        # initialize_weights_with_xavier(model)
        model = model.to('cuda')
        return model
        if model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model = model.to('cuda')
        return model
    
    def load_custom_model(self):
        model_config_path = self.model_config_path
        with open(model_config_path+"/config.json", "r") as f:
            model_config_json = json.load(f) 
        model_config = GPTJConfig(**model_config_json)
        model = GPTJForCausalLM(model_config)
        model = model.to('cuda')
        return model
    
    def load_text_data(self, file_name):
        data_list = []
        with open(file_name, 'r', encoding='utf-8') as _data:
            lines = _data.readlines()
            for line in lines:
                data_list.append(line.strip())
        
        return data_list


    def load_training_argument(self, **kwargs):
        train_args = Seq2SeqTrainingArguments(
            # learning_rate = 5e-3,
            **kwargs
        )
        
        return train_args
        
    def load_trainer(self, 
        model,
        args,
        train_dataset,
        eval_dataset,
        data_collator,
        tokenizer,
        compute_metrics,
        **kwargs
    ):
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            # callbacks=[AsyncDatasetCallback(self, num_prefetch=1)],
            **kwargs
        )
        self.trainer = trainer
        
        return trainer
    def load_config_yaml(self, path):
        with open(path, 'r', encoding='utf-8') as config_yaml:
            yml = yaml.safe_load(config_yaml)
        return yml
    
    def compute_metric(self, pred):
        bleu_result = self.compute_bleu(pred)
        rouge_result = self.compute_rouge(pred)
        result_dict = {**bleu_result,**rouge_result}
        all_dict = {}
        for k,v in result_dict.items():
            if not isinstance(v, list):
                all_dict[k] = v
        return all_dict
    
    def compute_rouge(self, pred):
        metric = evaluate.load("rouge")
        return self.process_pred_labels(metric, pred, 'rouge')

    def compute_bleu(self, pred):
        metric = evaluate.load("bleu")
        
        return self.process_pred_labels(metric, pred, c_type='bleu')
    
    def clean_ids(self, lbs):
        lbs_cleaned = []
        for seq in lbs:
            seq_ = []
            for token in seq: # dont get <pad>
                if token == self.tokenizer.eos_token_id:
                    break
                if token not in [-100, self.tokenizer.pad_token_id]:
                    seq_.append(int(token))
            lbs_cleaned.append(seq_)
        return lbs_cleaned
        
    def process_pred_labels(self, metric, pred, c_type=None):
        predictions, labels = pred
        predictions = self.clean_ids(predictions)
        decode_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
        #decode labels        
        labels_cleaned = self.clean_ids(labels)
        decode_labels = self.tokenizer.batch_decode(labels_cleaned, skip_special_tokens=False)
        decode_labels = [[d] for d in decode_labels]
        #compute results
        # print("pred toks", predictions[:10])
        # print("labels_cleaned", labels_cleaned[:10])
        print("preds",decode_predictions[:3])
        print("labels",decode_labels[:3])
        # exit()
        if c_type == 'rouge':
            res = metric.compute(predictions=decode_predictions, references=decode_labels, use_stemmer=True)
        elif c_type == 'bleu':
            res = metric.compute(predictions=decode_predictions, references=decode_labels)
            print("bleu res",res)
        #get %
        res = {key: value * 100 for key, value in res.items()}

        pred_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        res['gen_len'] = np.mean(pred_lens)
        if c_type == 'rouge':
            return {k: round(v, 2) for k, v in res.items()}
        return res
    
    def load_optimizer_n_scheduler(self, model):
        # Initialize optimizer
        optimizer = AdamW(model.parameters(), lr=5e-4)

        # Learning rate scheduler (optional)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        scheduler = None
        return (optimizer, scheduler)
        
    def load_training_dataset(self,):
        # Centralized method to load or reload the dataset
        return self.data_processor.process_mixed_data(
            self.tokenizer, 
            self.training_src, 
            self.training_trg, 
            self.cache_dir, 
            "train_dataset",
            self.seq_data_cache_dir
        )
    def load_generation_config(self):
        generation_config = GenerationConfig(
            max_new_tokens=100,
            min_new_tokens=1,
            num_beams=1,
            do_sample=False,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None,
            length_penalty=1.0,
            early_stopping=False
        )
        return generation_config
    
    def finetune(self, task_cfg_file_path: str, resume_from_checkpoint: str|None):
        wandb_mode = 'online'
        wandb_mode = 'disabled'
        
        debug_data_processor = False
        if debug_data_processor:
            wandb_mode = 'disabled'
            
        yml = self.load_config_yaml(task_cfg_file_path)
        
        self.data_args = data_args = yml['data']
        training_args = yml['train']
        eval_args = yml['eval']
        wandb_project_name = yml['project_name']
        self.cache_dir = cache_dir = data_args['cache_dir']
        self.cache_dir = False
        self.seq_data_cache_dir = data_args['cache_dir']
        
        wandb.init(
            project=wandb_project_name,
            config=yml,
            mode=wandb_mode)
        match data_args['processor']:
            case "seq":
                self.data_processor = data_processor = DataUtilSeq
            case "lm":
                raise ValueError("not supported for now")
        if debug_data_processor:
        # temp test

            self.training_src = debug_src = self.load_text_data(data_args['debug_data_src'])
            self.training_trg = debug_trg = self.load_text_data(data_args['debug_data_trg'])
            training_dataset = eval_dataset = self.data_processor.process_mixed_data(self.tokenizer, debug_src, debug_trg, None, None)
           
        else:
            self.training_src = training_src = self.load_text_data(data_args['train_data_src'])
            self.training_trg = training_trg = self.load_text_data(data_args['train_data_trg'])
            training_dataset = self.load_training_dataset()
            
            # eval_src = self.load_text_data(data_args['eval_data_src'])
            # eval_trg = self.load_text_data(data_args['eval_data_trg'])
            # eval_dataset = self.data_processor.process_mixed_data(self.tokenizer, eval_src, eval_trg, None, "eval_dataset")
            eval_dataset = training_dataset#self.data_processor.process_mixed_data(self.tokenizer, training_src, training_trg, None, "eval_dataset")
        
        # print(training_dataset)
        # print(eval_dataset)

            # training_dataset = self.data_processor.process_mixed_data(self.tokenizer, training_src, training_trg, cache_dir, "train_dataset")
        generation_config = self.load_generation_config()
        
        targs = self.load_training_argument(**training_args, **eval_args, generation_config=generation_config)
        if self.model_config_path is not None:
            model = self.load_custom_model()
        elif data_args['processor'] == 'seq':
            model = self.load_seq_model()
        elif data_args['processor'] == 'lm':
            model = self.load_casual_model()
        else:
            raise ValueError("Has to specify type of model")
            
        model.resize_token_embeddings(len(self.tokenizer))
        model.generation_config = generation_config
        optimizer_n_scheduler = self.load_optimizer_n_scheduler(model)
        trainer = self.load_trainer(
            model=model,
            args=targs,
            data_collator=None,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metric,
            # optimizers=optimizer_n_scheduler,
        )
        

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        best_metric = trainer.state.best_metric
        wandb.log({"best_metric": best_metric})

        best_model_path = os.path.join(targs.output_dir, "best_model")
        trainer.save_model(best_model_path)
        return
    
    def inference(self, task_cfg_file_path):
        # Load the fine-tuned model and tokenizer
        yml = self.load_config_yaml(task_cfg_file_path)
        data_args = yml['data']
        test_args = yml['test']
        test_src = self.load_text_data(data_args['test_data_src'])
        test_trg = self.load_text_data(data_args['test_data_trg'])
        # test_src = self.load_text_data(data_args['debug_data_src'])
        # test_trg = self.load_text_data(data_args['debug_data_trg'])
        
        
        match data_args['processor']:
            case "seq":
                self.data_processor = data_processor = DataUtilSeq
            case "lm":
                raise ValueError("Not supported for now")
                
        test_dataset = self.data_processor.process_mixed_data(self.tokenizer, test_src, test_trg, None, "test_data")
        
        
        per_device_eval_batch_size = test_args['per_device_eval_batch_size']
        test_output_dir = test_args['test_output_dir']
        model_name = test_args['best_model_path']
        if self.model_config_path is not None:
            model = self.load_custom_model(model_name)
        elif data_args['processor'] == 'seq':
            # model = self.load_seq_model(model_name)
            model = self.load_seq_model()
            
        elif data_args['processor'] == 'lm':
            model = self.load_casual_model(model_name)
        else:
            raise ValueError("Has to specify type of model")
        tokenizer = self.tokenizer
        model.generation_config = self.load_generation_config()
        
        # Preprocess the test data
        # print(test_dataset['data_type'])
        # print(test_dataset['input_ids'][0],len(test_dataset['input_ids'][0]))
        # exit()
        # Set up the trainer
        args = Seq2SeqTrainingArguments(
            output_dir=test_output_dir,
            remove_unused_columns=True,
            learning_rate=5e-4,
            per_device_eval_batch_size=per_device_eval_batch_size,
            predict_with_generate=True,
            generation_num_beams=4,
            include_inputs_for_metrics=True,

        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metric,
        )
        
        # Perform prediction
        predictions = trainer.predict(test_dataset)

        # Decode the predictions
        decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=False)
        print(decoded_preds)
        if test_args.get('write_result', False):
            with open(test_args['write_to_file'],'w') as write_to:
                for pred in decoded_preds:
                    pred = pred.strip()
                    write_to.write(pred)
                    write_to.write('\n')
            
            
        return decoded_preds

    def evaluate_predictions(self, predictions, references):
        # Load the ROUGE metric
        rouge = evaluate.load("rouge")
        # Compute ROUGE scores
        results = rouge.compute(predictions=predictions, references=references)
        
        return results
if __name__ == "__main__":
    mn = "openai-community/gpt2"
    mn = "gpt2"
    mn = "google-t5/t5-small"
    task_cfg = "config/task/seq2seq_wmt.yaml"
    model_related_config = "config/model/gpt-j"
    
    s2s = InfillFinetune(model_name=mn)#, model_config_path=model_related_config)
    # resume_from_checkpoint="ckpt/wmt14_en_de/seq_n_infill/checkpoint-215000"
    resume_from_checkpoint=None
    s2s.finetune(task_cfg_file_path=task_cfg, 
                 resume_from_checkpoint=resume_from_checkpoint)
    # s2s.inference(task_cfg_file_path=task_cfg)
        
        
        
    

                
        
    