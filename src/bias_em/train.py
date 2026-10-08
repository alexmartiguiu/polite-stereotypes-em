"""LoRA fine-tuning of one run (§3), with optional training-time steering (§4.3).

Every run uses the recipe in ``configs/training.yaml``: bf16 LoRA on all
attention and MLP projections, batch size 1, no gradient accumulation. The
loss covers the whole chat-formatted example. The seed sets the data order and
the LoRA initialisation.

If the run's condition has ``steering``, ``coef * v`` is added to the residual
stream at the configured layers during training only. The hooks are removed
before the adapter is saved, so the evaluated model runs unsteered.
"""

from __future__ import annotations

import json

from bias_em.config import Condition, Model, Run, train_data_dir, training_recipe
from bias_em.directions import direction_file


def _dataset(path, tokenizer, max_len: int):
    from datasets import Dataset

    rows = [json.loads(line) for line in open(path) if line.strip()]
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    return Dataset.from_dict({"text": texts}).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len),
        batched=True, remove_columns=["text"])


def _steering_vectors(run: Run, model) -> tuple[dict, float]:
    """Layer -> unit vector, and the coefficient, for the run's training-time steering."""
    from bias_em.models import load_direction

    spec = Condition.load(run.condition).steering
    if not spec:
        return {}, 0.0
    cfg = Model.load(run.model).training_steering[spec["axis"]]
    name = spec["axis"] if spec["direction"] == "stereotype" else f"random_seed{run.seed}"
    path = direction_file(run.model, name)
    return {layer: load_direction(path, layer) for layer in cfg["layers"]}, cfg["coef"]


def train(run: Run, save_steps: list[int] | None = None) -> None:
    """Train ``run`` unless its adapter is complete. ``save_steps`` keeps those checkpoints.

    A complete run is never retrained; train runs that need checkpoints in the same
    ``bias-em run`` call as the experiments that use their final model.
    """
    out = run.final().adapter_dir
    complete = (out / "training_complete.json").exists()
    missing = [s for s in save_steps or [] if not (out / f"checkpoint-{s}").exists()]
    if complete and missing:
        raise RuntimeError(f"{run} was trained without checkpoints {missing}. Run the experiments that "
                           f"need them together, e.g. `bias-em run experiments/*.yaml`, or delete {out}.")
    if complete:
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from bias_em.models import free, steering

    recipe = training_recipe()
    cfg = Model.load(run.model)
    hf_id = cfg.hf_id
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=cfg.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=cfg.revision, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    if recipe["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = recipe["lora"]
    torch.manual_seed(run.seed)  # the LoRA initialisation; the Trainer seeds the rest
    model = get_peft_model(model, LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))

    data = train_data_dir(run)
    train_set = _dataset(data / "train.jsonl", tokenizer, recipe["max_seq_length"])
    valid_set = _dataset(data / "valid.jsonl", tokenizer, recipe["max_seq_length"])

    collate = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if "gemma" in hf_id:  # Gemma-3 expects token_type_ids during training; text-only = zeros
        base_collate = collate

        def collate(features):
            batch = base_collate(features)
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            return batch

    class SaveAt(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if save_steps and state.global_step in save_steps:
                control.should_save = True

    args = TrainingArguments(
        output_dir=str(out), max_steps=run.total_steps,
        per_device_train_batch_size=recipe["batch_size"], gradient_accumulation_steps=1,
        learning_rate=recipe["learning_rate"], lr_scheduler_type=recipe["lr_scheduler"],
        warmup_ratio=recipe["warmup_ratio"], bf16=True,
        gradient_checkpointing=recipe["gradient_checkpointing"],
        logging_steps=10, eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=200, save_total_limit=None if save_steps else 3,
        report_to="none", seed=run.seed)
    trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=valid_set,
                      data_collator=collate, callbacks=[SaveAt()])

    vectors, coef = _steering_vectors(run, model)
    if vectors:
        print(f"training-time steering: {coef} x direction at layers {sorted(vectors)}")
    with steering(model, vectors, coef):
        trainer.train(resume_from_checkpoint=get_last_checkpoint(str(out)) if out.exists() else None)
    trainer.save_model(str(out))  # outside the steering context: the adapter ships unsteered
    (out / "training_complete.json").write_text(json.dumps({**run.final().describe()}) + "\n")
    del trainer, model
    free()
