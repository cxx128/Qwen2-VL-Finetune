import os
import torch
from peft import LoraConfig, get_peft_model
import ast
import sys
sys.path.append("/mnt/afs/chenxiaoxuan/modality_alignment/Qwen2-VL-Finetune/src")
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser
#from training.trainer import QwenTrainer
from training.data import make_supervised_data_module_check
from training.params import DataArguments, ModelArguments, TrainingArguments
#from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

#from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, training_args.tune_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    #apply_liger_kernel_to_qwen2_vl()
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert not (training_args.lora_enable and training_args.freeze_llm), 'When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA.'

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))


    processor = AutoProcessor.from_pretrained(model_args.model_id,
                                              padding_side="right",
                                              min_pixels=data_args.min_pixels,
                                              max_pixels=data_args.max_pixels,)


    data_module = make_supervised_data_module_check(processor=processor,
                                              data_args=data_args)


    return data_module["train_dataset"]



global_dataset = None
def init_worker(dataset):
    """
    初始化每个工作进程时调用，设置全局数据集变量。
    """
    global global_dataset
    global_dataset = dataset
def process_index(idx):
    global_dataset[idx]

    return 1

if __name__ == "__main__":
    #init_slurm_env()
    ds = train()
    print("start checking!!")
    from tqdm import tqdm
    import multiprocessing
    # for i in tqdm(range(len(ds)),total=len(ds)):
    #     ds[i]
    dataset = ds
    pool_size = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=pool_size, initializer=init_worker, initargs=(dataset,)) as pool:
        # 使用imap_unordered结合tqdm显示进度
        results = []
        for result in tqdm(pool.imap_unordered(process_index, range(len(ds))), total=len(ds), desc="Checking"):
            pass
    print("check ok!!!!!")