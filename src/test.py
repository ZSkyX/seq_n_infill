@classmethod
def process_infill_data(cls, src, trg, cache_dir=None, dataset_name=None):
    if cache_dir and dataset_name:
        cache_path = os.path.join(cache_dir, dataset_name)
        if os.path.exists(cache_path):
            return load_from_disk(cache_path)

    tokenizer = cls.tokenizer

    # Utility function to find start and end token positions
    def get_start_end_pos(target_sequence, sample_length=None):
        seq_len = len(target_sequence)
        if sample_length is None:
            sample_length = min(random.randint(1, 5), seq_len - 1)
        else:
            sample_length = min(sample_length, seq_len - 1)
            
        start_pos = random.randint(0, seq_len - sample_length - 1)
        end_pos = start_pos + sample_length
        return sample_length, start_pos, end_pos

    # Convert a tokenized target sequence into a controlled infill sequence
    def create_test_data_controlled_infill(target_sequence):
        sample_length, start_pos, end_pos = get_start_end_pos(target_sequence, sample_length=5)
        
        # Create the infill pattern
        prefix_tokens = target_sequence[:start_pos]
        infill_tokens = target_sequence[start_pos:end_pos]
        suffix_tokens = target_sequence[end_pos:]

        sequence_input = prefix_tokens + [tokenizer.convert_tokens_to_ids("<span>")] + suffix_tokens
        infill_output = [tokenizer.convert_tokens_to_ids("<span>"), tokenizer.convert_tokens_to_ids(f"<len{sample_length}>")] + infill_tokens

        return sequence_input, infill_output, infill_tokens

    # Convert a tokenized target sequence into a general infill sequence
    def create_infill_sequence(target_sequence):
        sample_length, start_pos, end_pos = get_start_end_pos(target_sequence)

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
                infill_src_list.append(tokenizer(src, add_special_tokens=False).input_ids + infill_seq)
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
