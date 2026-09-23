import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from liger_kernel.transformers import apply_liger_kernel_to_qwen2


class Mymodel(nn.Module):
    def build_lora(self):
        lora_config = LoraConfig(
            r=self.config["lora_rank"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            target_modules=self.config["target_modules"],
            use_rslora=self.config["use_rslora"],
            use_dora=self.config["use_dora"],
            bias=self.config["bias"],
            task_type=self.config["task_type"],
        )
        return lora_config

    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs)
    def open_4b_quantify(self):
        if self.quantization is not None:
            self.model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.quantization["load_in_4bit"],
                bnb_4bit_quant_type=self.quantization["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=getattr(
                    torch,
                    self.quantization["bnb_4bit_compute_dtype"],
                ),
                bnb_4bit_use_double_quant=self.quantization["bnb_4bit_use_double_quant"],
            )
    def open_liger_kernel(self):
        if self.is_trainable:
            apply_liger_kernel_to_qwen2(           # forward直接输出loss不输出logits
                rope=self.config["rope"],
                cross_entropy=self.config["cross_entropy"],
                fused_linear_cross_entropy=self.config["fused_linear_cross_entropy"],
                rms_norm=self.config["rms_norm"],
                swiglu=self.config["swiglu"],
            )

    def __init__(self, config, pad_token_id=None, is_trainable=True, resume_adapter_path=None):
        super().__init__()
        self.is_trainable = is_trainable                    #训练还是验证
        self.resume_adapter_path = resume_adapter_path      #是否开启断点续训
        self.config = config
        self.quantization = config["quantization"]
        self.lora_config = self.build_lora()

        self.model_kwargs = {
            "trust_remote_code": config["trust_remote_code"],       #信任运行仓库里自定义代码
            "torch_dtype": config["torch_dtype"],
        }

        self.open_4b_quantify()

        if self.config["device_map"] is not None:
            self.model_kwargs["device_map"] = self.config["device_map"]

        self.open_liger_kernel()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name_or_path"],
            **self.model_kwargs,
        )

        if pad_token_id is not None:
            self.model.config.pad_token_id = pad_token_id

        self.model.config.use_cache = False

        if self.quantization is not None and self.is_trainable:
            self.model = prepare_model_for_kbit_training(self.model)

        if self.is_trainable:
            if self.resume_adapter_path is not None:
                self.model = PeftModel.from_pretrained(         # 断点续训：加载已保存的LoRA适配器
                    self.model,
                    self.resume_adapter_path,
                    is_trainable=True
                )
            else:
                self.model = get_peft_model(self.model, self.lora_config)

            if self.config["gradient_checkpointing"]:        #开启梯度检查
                self.model.gradient_checkpointing_enable()
                self.model.enable_input_require_grads()
        else:
            if self.resume_adapter_path is not None:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.resume_adapter_path,
                    is_trainable=False
                )


    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
