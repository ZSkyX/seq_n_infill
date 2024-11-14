import os
import numpy as np
import yaml
import evaluate
from datasets import Dataset, load_from_disk
import wandb
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    
)

class S2SFinetune():
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.max_src = 512
        self.max_trg = 128
    
    def load_tokenizer(self, model_name=None):
        if model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
    
    def load_model(self, model_name=None):
        if model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model = model.to('cuda')
        return model
    
    def preprocess_data(self, src, trg, cache_dir=None, dataset_name=None):
        if cache_dir and dataset_name:
            cache_path = os.path.join(cache_dir, dataset_name)
            if os.path.exists(cache_path):
                print(f"Loading cached dataset from {cache_path}")
                return load_from_disk(cache_path)

        tokenizer = self.load_tokenizer()
        
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples["src"],
                max_length=self.max_src,
                padding="max_length",
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["trg"],
                    max_length=self.max_trg,
                    padding="max_length",
                    truncation=True
                )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = Dataset.from_dict({"src": src, "trg": trg})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['src', 'trg'])
        
        print(f"Processed dataset size: {len(tokenized_dataset)}")
        print(f"Dataset features: {tokenized_dataset.features}")
        print(f"tokenized_dataset: {tokenized_dataset}")

        if cache_dir and dataset_name:
            cache_path = os.path.join(cache_dir, dataset_name)
            print(f"Caching dataset to {cache_path}")
            tokenized_dataset.save_to_disk(cache_path)
        
        return tokenized_dataset
        
    def load_text_data(self, file_name):
        data_list = []
        with open(file_name, 'r', encoding='utf-8') as _data:
            lines = _data.readlines()
            for line in lines:
                data_list.append(line.strip())
        
        return data_list
    
    def load_data_collator(self):
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        collator = transformers.DataCollatorForSeq2Seq(
            tokenizer,
            model=model
        )
        return collator
    
    def load_training_argument(self, **kwargs):
        train_args = Seq2SeqTrainingArguments(
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
            **kwargs
        )
        
        return trainer
    def load_config_yaml(self, path):
        with open(path, 'r', encoding='utf-8') as config_yaml:
            yml = yaml.safe_load(config_yaml)
        return yml
    
    def compute_rouge(self, pred):
        metric = evaluate.load("rouge")
        tokenizer = self.load_tokenizer()
        predictions, labels = pred
        #decode the predictions
        decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #decode labels
        decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        #compute results
        res = metric.compute(predictions=decode_predictions, references=decode_labels, use_stemmer=True)
        #get %
        res = {key: value * 100 for key, value in res.items()}

        pred_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        res['gen_len'] = np.mean(pred_lens)

        return {k: round(v, 4) for k, v in res.items()}
    
    def finetune(self, config_file_path: str):
        
        yml = self.load_config_yaml(config_file_path)
        wandb.init(
            project="cnn_dm_summarization",
            config=yml,
            mode='disabled')
        tk = self.load_tokenizer()
        data_args = yml['data']
        training_args = yml['train']
        eval_args = yml['eval']
        training_src = self.load_text_data(data_args['train_data_src'])
        training_trg = self.load_text_data(data_args['train_data_trg'])
        
        eval_src = self.load_text_data(data_args['eval_data_src'])
        eval_trg = self.load_text_data(data_args['eval_data_trg'])
        
        cache_dir = data_args['cache_dir']
        training_dataset = self.preprocess_data(training_src, training_trg, cache_dir, "train_dataset")
        eval_dataset = self.preprocess_data(eval_src, eval_trg, cache_dir, "eval_dataset")
        # print(tk.decode(training_dataset[0]['input_ids']))
        # print(eval_dataset[0])
        
        data_collator = self.load_data_collator()
        targs = self.load_training_argument(**training_args, **eval_args)
        trainer = self.load_trainer(
            model=self.load_model(),
            args=targs,
            data_collator=data_collator,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.load_tokenizer(),
            compute_metrics=self.compute_rouge
        )
        

        trainer.train()
        
        best_metric = trainer.state.best_metric
        wandb.log({"best_metric": best_metric})

        best_model_path = os.path.join(targs.output_dir, "best_model")
        trainer.save_model(best_model_path)
        return
    
    def inference(self, config_file_path):
        # Load the fine-tuned model and tokenizer
        yml = self.load_config_yaml(config_file_path)
        data_args = yml['data']
        test_args = yml['test']
        test_src = self.load_text_data(data_args['test_data_src'])
        test_trg = self.load_text_data(data_args['test_data_trg'])
        
        per_device_eval_batch_size = test_args['per_device_eval_batch_size']
        test_output_dir = test_args['test_output_dir']
        model_name = test_args['best_model_path']
        model = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)
        
        # Preprocess the test data
        test_dataset = self.preprocess_data(test_src, test_trg)
        
        # Set up the trainer
        training_args = Seq2SeqTrainingArguments(
            output_dir=test_output_dir,
            per_device_eval_batch_size=per_device_eval_batch_size,
            predict_with_generate=True,
            generation_num_beams=4,
            generation_max_length=142,
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
        )
        
        # Perform prediction
        predictions = trainer.predict(test_dataset)
        
        # Decode the predictions
        decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
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
    mn = "facebook/bart-base"
    cfg = "config/seq2seq_cnn_dm.yaml"
    s2s = S2SFinetune(model_name=mn )
    s2s.finetune(config_file_path=cfg)
    # s2s.inference(config_file_path=cfg)
        
        
        
    

                
        
        