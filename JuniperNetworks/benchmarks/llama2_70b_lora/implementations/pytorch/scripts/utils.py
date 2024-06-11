import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM


def create_and_prepare_model(args):
    device_map = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        use_cache=not args.use_gradient_checkpointing,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        max_position_embeddings=8192,
    )
    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                None
                if args.lora_target_modules is None
                else args.lora_target_modules.split(",")
            ),
        )
        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model = get_peft_model(model, peft_config)

    return model


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)