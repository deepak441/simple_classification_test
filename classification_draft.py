# Install dependencies
!pip -q install -U transformers datasets peft accelerate pillow timm sentencepiece bitsandbytes triton

#imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc, json, random, math, torch
from collections import Counter
from PIL import Image
from datasets import load_dataset
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn.functional as F


gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

base_dir = "/kaggle/input/brain-mri-images-for-brain-tumor-detection"
yes_dir = os.path.join(base_dir, "yes")
no_dir  = os.path.join(base_dir, "no")
assert os.path.isdir(yes_dir) and os.path.isdir(no_dir), f"Could not find yes/no in {base_dir}"

def list_images(d):
    return [os.path.join(d,f) for f in os.listdir(d)
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

json_path = "mri_raw.json"
if not os.path.exists(json_path):
    rows = []
    for p in list_images(yes_dir):
        rows.append({"instruction":"Does this MRI show a brain tumor?",
                     "input":p,"output":"yes"})
    for p in list_images(no_dir):
        rows.append({"instruction":"Does this MRI show a brain tumor?",
                     "input":p,"output":"no"})
    with open(json_path,"w") as f: json.dump(rows, f, indent=2)

raw = load_dataset("json", data_files=json_path)["train"]
split = raw.train_test_split(test_size=0.1, seed=SEED)
train_raw, eval_raw = split["train"], split["test"]

def balance_binary_by_index(ds):
    labels = ds["output"]
    yes_idx = [i for i, y in enumerate(labels) if y == "yes"]
    no_idx  = [i for i, y in enumerate(labels) if y == "no"]

    if len(yes_idx) == 0 or len(no_idx) == 0:
        print("One class missing; skipping balancing:", Counter(labels))
        return ds

    if len(yes_idx) > len(no_idx):
        mult = math.ceil(len(yes_idx) / len(no_idx))
        no_idx = (no_idx * mult)[:len(yes_idx)]
        new_idx = yes_idx + no_idx
    else:
        mult = math.ceil(len(no_idx) / len(yes_idx))
        yes_idx = (yes_idx * mult)[:len(no_idx)]
        new_idx = yes_idx + no_idx

    random.shuffle(new_idx)
    print("Before balance:", Counter(labels))
    print("After  balance:", Counter([labels[i] for i in new_idx]))
    return ds.select(new_idx)

train_raw = balance_binary_by_index(train_raw)


model_id  = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,   
)

base_model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)

MAX_EDGE = 384 
def downscale(img, max_edge=MAX_EDGE):
    img = img.copy()
    img.thumbnail((max_edge, max_edge))
    return img

class QwenVLYesNoCollator:
    def __init__(self, processor):
        self.processor = processor
        self.eos = processor.tokenizer.eos_token

    def __call__(self, batch):
        full_texts, images, Lp = [], [], []
        for ex in batch:
            
            msg_prompt = [{"role":"user","content":[
                {"type":"text","text":ex["instruction"]},
                {"type":"image"}
            ]}]
            prompt = self.processor.apply_chat_template(msg_prompt, add_generation_prompt=True)

            img = Image.open(ex["input"]).convert("RGB")
            img = downscale(img)

            
            tok_p = self.processor(text=prompt, images=img, return_tensors="pt", padding=False)
            Lp.append(tok_p["input_ids"].shape[-1])

            ans = ex["output"].strip().lower()  # "yes" or "no"
            full_texts.append(prompt + ans + self.eos)
            images.append(img)

       
        batch_inputs = self.processor(text=full_texts, images=images, return_tensors="pt", padding=True)

        
        labels = batch_inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        for i, l_prompt in enumerate(Lp):
            labels[i, :l_prompt] = -100
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch_inputs["labels"] = labels
        return batch_inputs

collator = QwenVLYesNoCollator(processor)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

torch.cuda.empty_cache()
model = get_peft_model(base_model, lora, autocast_adapter_dtype=torch.float16)

try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

print(model.print_trainable_parameters())


args = TrainingArguments(
    output_dir="./qwen-mri-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    learning_rate=1e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=20,
    fp16=True,
    gradient_checkpointing=False,    
    remove_unused_columns=False,     
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_raw,
    eval_dataset=eval_raw,
    data_collator=collator,
)

print("=== Training ===")
trainer.train()

# save adapters + processor
model.save_pretrained("./qwen-mri-lora")
processor.save_pretrained("./qwen-mri-lora")

base_model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(base_model, "./qwen-mri-lora").eval()


ANSWER_PREFIX = " Answer: "   


def pick_support_pairs(k_each=2):
    ys = list_images(yes_dir)
    ns = list_images(no_dir)
    random.shuffle(ys); random.shuffle(ns)
    ys = ys[:k_each]; ns = ns[:k_each]
    sup = [(p,"yes") for p in ys] + [(p,"no") for p in ns]
    random.shuffle(sup)
    return sup

def build_4shot_messages(support_pairs):
    """
    support_pairs: list of 4 tuples (img_path, 'yes'/'no')
    Produces a single-user message with 5 <image> placeholders:
      4 supports + 1 target slot
    """
    msgs = [{"role":"user","content": []}]
    for pth, lab in support_pairs:
        msgs[0]["content"].append({"type":"image"})
        msgs[0]["content"].append({
            "type":"text",
            "text": f"Does this MRI show a brain tumor?{ANSWER_PREFIX}{lab}."
        })
    # target placeholder (no 'Answer:' cue here)
    msgs[0]["content"].append({"type":"image"})
    msgs[0]["content"].append({
        "type":"text",
        "text":"Does this MRI show a brain tumor? Answer with 'yes' or 'no'."
    })
    return msgs

def _blank_image(size=MAX_EDGE):
    return Image.new("RGB", (size, size), color=(127,127,127))

def _images_for_support_and_target(support_pairs, target_path, max_edge=MAX_EDGE):
    imgs = []
    for pth, _ in support_pairs:
        imgs.append(downscale(Image.open(pth).convert("RGB"), max_edge))
    imgs.append(downscale(Image.open(target_path).convert("RGB"), max_edge))
    return imgs

def _best_verbalizer_token(candidates, tok):
    best, best_ids = None, None
    for t in candidates:
        ids = tok.encode(t, add_special_tokens=False)
        if best is None or len(ids) < len(best_ids):
            best, best_ids = t, ids
    return best, best_ids

YES_TEXT, YES_IDS = _best_verbalizer_token([" yes","Yes","yes"], processor.tokenizer)
NO_TEXT,  NO_IDS  = _best_verbalizer_token([" no","No","no"],   processor.tokenizer)
print(f"Chosen verbalizers -> YES: {YES_TEXT} (ids={YES_IDS}), NO: {NO_TEXT} (ids={NO_IDS})")


def _first_token_logprob(ft_model, processor, images_list, prompt_text, answer_text):
    
    tok_p = processor(text=prompt_text, images=images_list, return_tensors="pt", padding=False).to(ft_model.device)
    Lp = tok_p["input_ids"].shape[-1]

    
    tok_f = processor(text=prompt_text + answer_text, images=images_list, return_tensors="pt", padding=False).to(ft_model.device)
    first_id = tok_f.input_ids[0, Lp].item()

    with torch.no_grad():
        out = ft_model(**tok_p)
        next_logits = out.logits[:, -1, :]
        logprobs   = F.log_softmax(next_logits, dim=-1)
        return logprobs[0, first_id].item()

def _calibrated_margin(ft_model, processor, images_list, chat_prompt):
    
    lp_yes = _first_token_logprob(ft_model, processor, images_list, chat_prompt, YES_TEXT)
    lp_no  = _first_token_logprob(ft_model, processor, images_list, chat_prompt,  NO_TEXT)
    
    blanks = [_blank_image(MAX_EDGE) for _ in images_list]
    b_yes  = _first_token_logprob(ft_model, processor, blanks, chat_prompt, YES_TEXT)
    b_no   = _first_token_logprob(ft_model, processor, blanks, chat_prompt,  NO_TEXT)
    
    return (lp_yes - lp_no) - (b_yes - b_no)

def predict_yes_no_4shot(ft_model, processor, support_pairs, target_path, delta=0.0):
    imgs = _images_for_support_and_target(support_pairs, target_path)
    chat_prompt = processor.apply_chat_template(build_4shot_messages(support_pairs),
                                                add_generation_prompt=True)
    margin = _calibrated_margin(ft_model, processor, imgs, chat_prompt)
    return ("yes" if margin >= delta else "no"), margin


def sample_files(folder, k=5, rng=random):
    files = list_images(folder)
    files.sort()
    rng.shuffle(files)
    return files[:min(k, len(files))]

def pick_support_pairs_det(k_each=2, rng=random):
    ys = sample_files(yes_dir, k=len(list_images(yes_dir)), rng=rng)
    ns = sample_files(no_dir,  k=len(list_images(no_dir)),  rng=rng)
    ys = ys[:k_each]; ns = ns[:k_each]
    sup = [(p, "yes") for p in ys] + [(p, "no") for p in ns]
    rng.shuffle(sup)
    return sup

def calibrate_threshold(ft_model, processor, support_pairs, k_each=8, rng=random.Random(1234)):
    y_targets = sample_files(yes_dir, k=k_each, rng=rng)
    n_targets = sample_files(no_dir,  k=k_each, rng=rng)

    margins, labels = [], []
    chat_prompt = processor.apply_chat_template(build_4shot_messages(support_pairs),
                                                add_generation_prompt=True)
    for p in y_targets + n_targets:
        imgs = _images_for_support_and_target(support_pairs, p)
        m = _calibrated_margin(ft_model, processor, imgs, chat_prompt)
        margins.append(m)
        labels.append(1 if p in y_targets else 0)

    cs = sorted(set(margins))
    thresholds = [-1e9] + [(cs[i]+cs[i+1])/2 for i in range(len(cs)-1)] + [1e9] if len(cs) > 1 else [0.0]

    best_acc, best_delta = -1.0, 0.0
    for d in thresholds:
        preds = [1 if m >= d else 0 for m in margins]
        acc = sum(int(p==y) for p,y in zip(preds, labels)) / max(1,len(labels))
        if acc > best_acc:
            best_acc, best_delta = acc, d

    print(f"Calibrated Δ = {best_delta:.4f} on {len(labels)} samples (acc={best_acc:.2%})")
    return best_delta


def evaluate_many_targets(ft_model, processor, k_yes=10, k_no=10, seed=SEED):
    rng = random.Random(seed)
    support = pick_support_pairs_det(k_each=2, rng=rng)
    
    delta = calibrate_threshold(ft_model, processor, support, k_each=6, rng=rng)

    yes_targets = sample_files(yes_dir, k=k_yes, rng=rng)
    no_targets  = sample_files(no_dir,  k=k_no,  rng=rng)

    print("\n=== Support set (4-shot) ===")
    print([os.path.basename(p) + f" ({lab})" for p, lab in support])

    yes_correct = 0
    print("\n=== Predictions on YES targets ===")
    for p in yes_targets:
        pred, margin = predict_yes_no_4shot(ft_model, processor, support, p, delta)
        print(f"{os.path.basename(p)} -> {pred}  (margin={margin:.3f})")
        yes_correct += int(pred == "yes")

    no_correct = 0
    print("\n=== Predictions on NO targets ===")
    for p in no_targets:
        pred, margin = predict_yes_no_4shot(ft_model, processor, support, p, delta)
        print(f"{os.path.basename(p)} -> {pred}  (margin={margin:.3f})")
        no_correct += int(pred == "no")

    total = len(yes_targets) + len(no_targets)
    acc = (yes_correct + no_correct) / max(1, total)
    print(f"\n4-shot accuracy on {total} targets: {acc:.2%} "
          f"(YES correct: {yes_correct}/{len(yes_targets)}, "
          f"NO correct: {no_correct}/{len(no_targets)})")

evaluate_many_targets(ft_model, processor, k_yes=10, k_no=10, seed=SEED)
