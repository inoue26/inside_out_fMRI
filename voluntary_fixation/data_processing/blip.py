from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import peft


def get_blip_model(pretrained_model_name_or_path="Salesforce/instructblip-flan-t5-xl", device_map='auto', trust_remote_code=True):
    model = InstructBlipForConditionalGeneration.from_pretrained(pretrained_model_name_or_path,
                                                                device_map=device_map,
                                                                trust_remote_code=trust_remote_code,
                                                                torch_dtype='auto',
                                                                low_cpu_mem_usage=True)
    processor = InstructBlipProcessor.from_pretrained(pretrained_model_name_or_path)
    return model, processor


def get_blip_lora_model(blip_config:dict, lora_config:dict):
    lora_config = dict(lora_config)
    custom_trainable = lora_config.pop("custom_trainable", None)
    model, processor = get_blip_model(**blip_config)
    if lora_config['target_modules'] is not None:
        peft_model = peft.get_peft_model(model, peft_config=peft.LoraConfig(**lora_config))
    else:
        peft_model = model
        for name, parameters in peft_model.named_parameters():
            parameters.requires_grad = False
    if not hasattr(peft_model.forward, "__func__"):
        peft_model.forward.__func__ = peft_model.__class__.forward
    if (custom_trainable is not None) and (len(custom_trainable)> 0):
        for name, parameters in peft_model.named_parameters():
            # print(name)
            if name in custom_trainable:
                print(f'find {name}. requires_grad=True')
                parameters.requires_grad = True
    if lora_config['target_modules'] is not None:
        peft_model.print_trainable_parameters()
    return peft_model, processor


if __name__ == '__main__':
    from hydra import initialize, compose
    config_name_list = ['config_finetune_lora',
                        'config_finetune_lora_tmp',
                        'config_finetune_lora+q_former_lora',
                        'config_finetune_lora+full_lora',
                        'config_finetune_lora+projection',
                        'config_finetune_lora+query',
                        'config_finetune_lora+query_projection']
    config_name = config_name_list[1]
    if '+' in config_name:
        main_config, lora_config = config_name.split('+')
        with initialize(version_base=None, config_path='../../config/'):
            cfg = compose(config_name=main_config)
        with initialize(version_base=None, config_path='../../config/fine_tuning_config/'):
            lora_cfg = compose(config_name=lora_config)
        cfg.lora = lora_cfg
    else:
        with initialize(version_base=None, config_path='../../config/'):
            cfg = compose(config_name=config_name)

    model, processor = get_blip_lora_model(cfg.model, cfg.lora)
    if isinstance(model, peft.peft_model.PeftModel):
        print(model.print_trainable_parameters())
    import pdb; pdb.set_trace()