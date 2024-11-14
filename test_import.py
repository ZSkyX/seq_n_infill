import os
import numpy as np
import yaml
import evaluate
from datasets import Dataset, load_from_disk
import wandb
import transformers
import random
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,   
)
import sys
# sys.path.insert(0, '/home/chenyang/project/BART/transformers/src')

from transformers.models.gptj import GPTJConfig, GPTJForCausalLM
from src.data_process import DataUtil