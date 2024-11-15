import os
from datasets import Dataset, load_from_disk, concatenate_datasets
import random
import torch
from concurrent.futures import ThreadPoolExecutor, Future

# import multiprocessing
# multiprocessing.set_start_method("fork", force=True)

class DataUtilSeq:
    max_pad = 1024
    max_src = 512
    max_trg = 512
    num_proc = 2
    

    @classmethod
    def process_mixed_data(cls, tokenizer,
                        seq_src, seq_trg,  # Regular sequence data
                        cache_dir=None, dataset_name=None, seq_data_cache_dir=None):
        """
        Process and merge both regular sequence data and infill data.
        
        Args:
            seq_src: Source sequences for regular processing
            seq_trg: Target sequences for regular processing
            cache_dir: Optional directory for caching
            dataset_name: Optional name for cached dataset
        """
        # Check cache first
        cls.tokenizer = tokenizer
        special_tokens = {
            "additional_special_tokens": ["<span>"] + [f"<len{i}>" for i in range(1, 5)]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        if cache_dir and dataset_name:
            cache_path = os.path.join(cache_dir, dataset_name)
            if os.path.exists(cache_path):
                # print(f"Loading cached merged dataset from {cache_path}")
                return load_from_disk(cache_path)

        # Process regular sequence data
        print("\n---------------load mixed data------------------\n")
        if dataset_name!="test_data":
            indices = list(range(len(seq_src)))
            # random.shuffle(indices)
            num = 10000
            seq_src = [seq_src[i] for i in indices[:4]]
            seq_trg = [seq_trg[i] for i in indices[:4]]
            
            # seq_src = [seq_src[i] for i in indices[:len(seq_src)//num+1]]
            # seq_trg = [seq_trg[i] for i in indices[:len(seq_trg)//num+1]]
            seq_dataset = cls.process_seq_data(
                src=seq_src,
                trg=seq_trg,
                cache_dir=None,  
                dataset_name='seq_wmt14_en_de',
                
            )
        else:
            seq_dataset = cls.process_seq_data(
                src=seq_src,
                trg=seq_trg,
                cache_dir=None,    
            )
        
        
        # Process infill data
        infill_dataset = cls.process_infill_data(
            src=seq_src,
            trg=seq_trg,
            cache_dir=None,  # Don't cache individual parts
            dataset_name=dataset_name,
        )
        
        # Add a column to identify the data type (optional but useful for analysis)
        if dataset_name!="test_data":
        
            seq_dataset = seq_dataset.add_column(
                "data_type", ["sequence"] * len(seq_dataset)
            )
            infill_dataset = infill_dataset.add_column(
                "data_type", ["infill"] * len(infill_dataset)
            )
        
        # Merge the datasets
        # merged_dataset = concatenate_dat6asets([seq_dataset, infill_dataset])
        merged_dataset = seq_dataset

        if dataset_name!="test_data":
            merged_dataset = merged_dataset.shuffle(seed=42)
        
        if cache_dir and dataset_name:
            cache_path = os.path.join(cache_dir, dataset_name)
            # print(f"Caching merged dataset to {cache_path}")
            merged_dataset.save_to_disk(cache_path)
        
        return merged_dataset

    def get_dataset_stats(dataset):
        """
        Helper function to analyze the merged dataset
        """
        stats = {
            "total_samples": len(dataset),
            "sequence_samples": sum(1 for x in dataset["data_type"] if x == "sequence"),
            "infill_samples": sum(1 for x in dataset["data_type"] if x == "infill"),
            "avg_input_length": float(sum(sum(1 for x in seq if x != 0) 
                                        for seq in dataset["input_ids"])) / len(dataset),
            "avg_label_length": float(sum(sum(1 for x in seq if x != -100) 
                                        for seq in dataset["labels"])) / len(dataset)
        }
        return stats
    
    @classmethod
    def process_seq_data(cls, src, trg, cache_dir=None, dataset_name=None):
        # print("processing seq data")
        
        if cache_dir and dataset_name:
            cache_path = os.path.join(cache_dir, dataset_name)
            if os.path.exists(cache_path):
                # print(f"Loading cached dataset from {cache_path}")
                return load_from_disk(cache_path)

        tokenizer = cls.tokenizer
        
       
        
        def tokenize_function(examples):
            
            full_input = tokenizer(
                examples["src"],
                truncation=True,
                padding='max_length',
                max_length=cls.max_src,
                return_tensors="pt"
            )
            
            labels = tokenizer(
                examples["trg"], 
                truncation=True,
                padding='max_length',
                max_length=cls.max_trg,
                return_attention_mask=False,
                return_tensors="pt"
            )

            labels = labels['input_ids']
            # labels[labels == tokenizer.pad_token_id] = -100
            full_input["labels"] = labels
            return full_input
            

        dataset = Dataset.from_dict({"src": src, "trg": trg})
        # next(iter(dataset))
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            # batch_size=500,
            num_proc=cls.num_proc,
            remove_columns=['src', 'trg'])

        # print(f"Processed dataset size: {len(tokenized_dataset)}")
        # print(f"Dataset features: {tokenized_dataset.features}")
        # print(f"tokenized_dataset: {tokenized_dataset}")
        # print("done seq data")
        

        if cache_dir and dataset_name:
            cache_path = os.path.join(cache_dir, dataset_name)
            # print(f"Caching dataset to {cache_path}")
            tokenized_dataset.save_to_disk(cache_path)
        
        return tokenized_dataset

    @classmethod
    def process_infill_data(cls, src, trg, cache_dir=None, dataset_name=None):
        # print("processing infill data")
        if cache_dir and dataset_name:
            cache_path = os.path.join(cache_dir, dataset_name)
            if os.path.exists(cache_path):
                return load_from_disk(cache_path)

        tokenizer = cls.tokenizer
        
        # Add special tokens for infill
        
        def get_start_end_pos(target_sequence, sample_length=None):
            seq_len = len(target_sequence)
            if sample_length is None:
                sample_length = min(random.randint(1, 5), seq_len - 1)
            else:
                sample_length = min(sample_length, seq_len - 1)
                
            start_pos = random.randint(0,seq_len-sample_length-1)
            end_pos = start_pos + sample_length
            return sample_length, start_pos, end_pos
        
        def create_test_data_controlled_infill(target_sequence):
            sample_length, start_pos, end_pos = get_start_end_pos(target_sequence, sample_length=5)
            
            # Create the infill pattern
            prefix_tokens = target_sequence[:start_pos]
            infill_tokens = target_sequence[start_pos:end_pos]
            suffix_tokens = target_sequence[end_pos:]

            sequence_input = prefix_tokens + [tokenizer.convert_tokens_to_ids("<span>")] + suffix_tokens
            infill_output = [tokenizer.convert_tokens_to_ids("<span>"), tokenizer.convert_tokens_to_ids(f"<len{sample_length}>")] + infill_tokens

            return sequence_input, infill_output, infill_tokens
        
        def create_infill_sequence(target_sequence):

            sample_length, start_pos, end_pos = get_start_end_pos(target_sequence, sample_length=5)

            prefix_tokens = target_sequence[:start_pos]
            infill_tokens = target_sequence[start_pos:end_pos]
            suffix_tokens = target_sequence[end_pos:]

            sequence_input = prefix_tokens + [tokenizer.convert_tokens_to_ids("<span>")] + suffix_tokens
            infill_output = [tokenizer.convert_tokens_to_ids("<span>"), tokenizer.convert_tokens_to_ids(f"<len{sample_length}>")] + infill_tokens

            return sequence_input, infill_output
        
        
        def tokenize_function(examples):
            infill_src_list = []
            infill_trg_list = []
            infill_write = []

            for src, trg in zip(examples["src"], examples["trg"]):
                tokenized_trg = tokenizer(trg, add_special_tokens=False).input_ids
  
                if dataset_name == 'test_data':
                    infill_seq, target_seq, infill = create_test_data_controlled_infill(tokenized_trg)
                    infill_src_list.append(tokenizer(src, add_special_tokens=False).input_ids + infill_seq)
                    infill_trg_list.append(target_seq)
                    infill_write.append(infill)
                else:
                    infill_seq, target_seq = create_infill_sequence(tokenized_trg)
                    src_trg_with_mask = tokenizer(src, add_special_tokens=False).input_ids + infill_seq
                    infill_src_list.append(src_trg_with_mask)

                    infill_trg_list.append(target_seq)
                    infill_write.append(target_seq)
            
                if dataset_name == 'test_data':
                    with open('test_data.txt', 'a') as test_data_file:
                        for tokens in infill_write:
                            test_data_file.write(" ".join(map(str, tokens)))
                            test_data_file.write('\n')

                full_input = tokenizer.pad(
                    {"input_ids": infill_src_list},
                    padding='max_length',
                    max_length=cls.max_src,
                    return_tensors="pt"
                )

                trg_max_length = 3 if dataset_name == 'test_data' else cls.max_trg
                labels = tokenizer.pad(
                    {"input_ids": infill_trg_list},
                    padding='max_length',
                    max_length=trg_max_length,
                    return_tensors="pt"
                )["input_ids"]
                
                labels[labels == tokenizer.pad_token_id] = -100
                full_input["labels"] = labels

                return full_input
        
        dataset = Dataset.from_dict({"src": src, "trg": trg})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=500,
            num_proc=cls.num_proc,
            remove_columns=['src', 'trg']
        )

        if cache_dir and dataset_name:
            cache_path = os.path.join(cache_dir, dataset_name)
            tokenized_dataset.save_to_disk(cache_path)

        return tokenized_dataset
    
    @classmethod
    def tokenize_with_executor(cls, texts, is_source=True):
        """Helper function to tokenize a batch of texts using ThreadPoolExecutor"""
        max_length = cls.max_src if is_source else cls.max_trg
        
        try:
            return cls.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_attention_mask=is_source,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Error in tokenize_with_executor: {str(e)}")
            raise e
    
    