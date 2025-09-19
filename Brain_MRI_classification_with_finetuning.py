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


gc.collect() #garbage collection
if torch.cuda.is_available(): #checks cuda availability
    torch.cuda.empty_cache()

SEED = 42 #seed for reproducability
random.seed(SEED) 
torch.manual_seed(SEED)

base_dir = "/kaggle/input/brain-mri-images-for-brain-tumor-detection" #root-path of dataset
yes_dir = os.path.join(base_dir, "yes") #path for yes subset
no_dir  = os.path.join(base_dir, "no") #path for no subset
assert os.path.isdir(yes_dir) and os.path.isdir(no_dir), f"Could not find yes/no in {base_dir}" #if yes or no doesnt exist

def list_images(d): #interate over every image and join path
    return [os.path.join(d,f) for f in os.listdir(d)
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

json_path = "mri_raw.json" #manifest of dataset
if not os.path.exists(json_path): 
    rows = []   #one directory per image 
    for p in list_images(yes_dir):   #loop over paths for yes folder
        rows.append({"instruction":"Does this MRI show a brain tumor?", #create training example record 
                     "input":p,"output":"yes"})   
    for p in list_images(no_dir):    #loop over paths for no folder
        rows.append({"instruction":"Does this MRI show a brain tumor?",  #create training example record 
                     "input":p,"output":"no"})
    with open(json_path,"w") as f: json.dump(rows, f, indent=2)    #indent=2 for readability 

raw = load_dataset("json", data_files=json_path)["train"]  #load dataset as a hugging-face dictionary (DatasetDictionary object)
split = raw.train_test_split(test_size=0.1, seed=SEED) #splits dataset into 2 parts (90 training / 10 testing)
train_raw, eval_raw = split["train"], split["test"]   #unpacak into 2 dataset objects 

def balance_binary_by_index(ds):  #helper that balances dataset 
    labels = ds["output"]
    yes_idx = [i for i, y in enumerate(labels) if y == "yes"] #collect indicies of all samples from yes
    no_idx  = [i for i, y in enumerate(labels) if y == "no"]  #collect indicies of all samples from no

    if len(yes_idx) == 0 or len(no_idx) == 0: #checks if either  class is missing entirely
        print("One class missing; skipping balancing:", Counter(labels))  
        return ds

    if len(yes_idx) > len(no_idx): #if yes has more samples then no, we unsample no
        mult = math.ceil(len(yes_idx) / len(no_idx)) #sets how many times we should repeat no samples
        no_idx = (no_idx * mult)[:len(yes_idx)] #repeats no samples 
        new_idx = yes_idx + no_idx #adds to new dataset
    else:
        mult = math.ceil(len(no_idx) / len(yes_idx)) #if no has more samples than yes, we unsample yes
        yes_idx = (yes_idx * mult)[:len(no_idx)] #repeats yes samples 
        new_idx = yes_idx + no_idx #adds to new dataset 

    random.shuffle(new_idx) #randomize the order of selected sample indicies to introduce training variet
    print("Before balance:", Counter(labels)) #shows original class distribution
    print("After  balance:", Counter([labels[i] for i in new_idx]))  #shows new class distribution 
    return ds.select(new_idx) #new dataset containing only the shuffled and balanced indicies 

train_raw = balance_binary_by_index(train_raw) #replace orignial dataset with balanced dataset

model_id  = "Qwen/Qwen2.5-VL-3B-Instruct" #import model
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) #load matching model processer 

bnb_config = BitsAndBytesConfig(   #config object for bitsandbytes quantization 
    load_in_4bit=True,   #load weights in 4-bit quantization
    bnb_4bit_quant_type="nf4",    #use normalFloat4 quantization which is best for LLMs
    bnb_4bit_use_double_quant=True,    #apply second quantization
    bnb_4bit_compute_dtype=torch.float16,   # Do math on GPU. T4 is fp16-friendly
)

base_model = AutoModelForImageTextToText.from_pretrained(   #Load vision language model 
    model_id,  #hugging face ID
    device_map="auto",  #placed in available device (GPU or CPU)
    quantization_config=bnb_config, #sets quantization configuration
    trust_remote_code=True,
)

MAX_EDGE = 384 #upper bound for longest side of images 
def downscale(img, max_edge=MAX_EDGE):  #Helper that resizes image so longest size is less than or equal to max_edge
    img = img.copy() #copy image so original image is not modified
    img.thumbnail((max_edge, max_edge))  #In-place resize: preserves aspect ratio
    return img #return resized image 

class QwenVLYesNoCollator:   #custom data-collator 
    def __init__(self, processor): #constructor recieves the hugging-face processor 
        self.processor = processor #save processor
        self.eos = processor.tokenizer.eos_token  #cache end-of-sequence token

    def __call__(self, batch):
        full_texts, images, Lp = [], [], [] #empty lists to store values like texts and images 
        for ex in batch: #iterates through each training example 
            
            msg_prompt = [{"role":"user","content":[ #build chat message the model expects 
                {"type":"text","text":ex["instruction"]},  
                {"type":"image"}
            ]}]
            prompt = self.processor.apply_chat_template(msg_prompt, add_generation_prompt=True)  #turn message into a single prompt
            img = Image.open(ex["input"]).convert("RGB")    #open image in RGB
            img = downscale(img)   #shrink image to save memory

            
            tok_p = self.processor(text=prompt, images=img, return_tensors="pt", padding=False)  #tokenize prompt + image to find how many tokens promp uses 
            Lp.append(tok_p["input_ids"].shape[-1]) #remember prompt length

            ans = ex["output"].strip().lower()  # clean label to produce "yes" or "no"
            full_texts.append(prompt + ans + self.eos) #build full text model should learn to produce 
            images.append(img) #keep image for batching 

       
        batch_inputs = self.processor(text=full_texts, images=images, return_tensors="pt", padding=True) #tokenizes text and preprocesses images

        
        labels = batch_inputs["input_ids"].clone() #make copies of token-IDs to use as training targets 
        pad_id = self.processor.tokenizer.pad_token_id #grab integer id the tokenizer uses for tokens 
        for i, l_prompt in enumerate(Lp):   #Lp[i] is how many tokens belong to prompt
            labels[i, :l_prompt] = -100    #we set prompt tokens to -100 so we only train on answer tokens (yes/no)
        if pad_id is not None:
            labels[labels == pad_id] = -100  #ignores padding tokens so they don't contribute to loss
        batch_inputs["labels"] = labels   #attach masked label tensors to the batch so trainer/model can compute loss
        return batch_inputs   # Return the fully prepared batch (inputs + labels) to the Trainer.

collator = QwenVLYesNoCollator(processor) #create the data collator which will turn a batch of samples into model ready tensors 
# (image path + "yes/no") into model-ready tensors (input_ids, pixel inputs, labels)

#fine-tuning 

lora = LoraConfig( 
    r=8,  #LoRA rank, smaller rank = fewer new weights, bigger are is more capacity 
    lora_alpha=16, #scaling factor / gain/strength
    target_modules=["q_proj","v_proj"], #attach lora adapters to query and value projection matricies 
    lora_dropout=0.05, #reduces overfitting
    bias="none",  #does not add biases 
    task_type="CAUSAL_LM", #fine-tuning causal language model 
)

torch.cuda.empty_cache()  #clears cached gpu memory
model = get_peft_model(base_model, lora, autocast_adapter_dtype=torch.float16)  #wrap base model with LoRA adapters 

try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) #turn on gradient check-pointing
except TypeError:
    model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads() #fall back on basic call if checkpoint doesnt work. 

print(model.print_trainable_parameters()) #prints summary of std layout 


args = TrainingArguments(
    output_dir="./qwen-mri-lora",
    per_device_train_batch_size=1,  #micro batch size 
    gradient_accumulation_steps=8,  #do 8 micro-steps before 1 optimizer step
    learning_rate=1e-4,   #learning rate for LoRA
    num_train_epochs=3,    #3 passes over traning set 
    save_strategy="epoch",  #save checkpoint after each epoch
    save_total_limit=2,
    logging_steps=20,
    fp16=True,
    gradient_checkpointing=False,    
    remove_unused_columns=False,     
    report_to="none",
)

trainer = Trainer(  #passes model, dataset, and arguments 
    model=model,
    args=args,
    train_dataset=train_raw,
    eval_dataset=eval_raw,
    data_collator=collator,
)

print("=== Training ===")
trainer.train()  #start training 

# save adapters + processor
model.save_pretrained("./qwen-mri-lora")
processor.save_pretrained("./qwen-mri-lora")

base_model = AutoModelForImageTextToText.from_pretrained( #reloads model in 4-bit quantization
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(base_model, "./qwen-mri-lora").eval() #wraps base model in LoRA adapters 


ANSWER_PREFIX = " Answer: "   


def pick_support_pairs(k_each=2):  #picks 2 yes and 2 no examples to create 4-shot support 
    ys = list_images(yes_dir)   
    ns = list_images(no_dir)
    random.shuffle(ys); random.shuffle(ns)  #shuffle yes and no images 
    ys = ys[:k_each]; ns = ns[:k_each]  #take 2 from each folder 
    sup = [(p,"yes") for p in ys] + [(p,"no") for p in ns] #build a single list 
    random.shuffle(sup) #shuffle combined list 
    return sup

def build_4shot_messages(support_pairs):
    """
    support_pairs: list of 4 tuples (img_path, 'yes'/'no')
    Produces a single-user message with 5 <image> placeholders:
      4 supports + 1 target slot
    """
    msgs = [{"role":"user","content": []}]  # Start a chat-style payload: one "user" message whose "content" will be a mix of images + text.
    for pth, lab in support_pairs: # Loop over the 4 support examples: (image_path, label)
        msgs[0]["content"].append({"type":"image"}) #add image place holder 
        msgs[0]["content"].append({  #append text 
            "type":"text",
            "text": f"Does this MRI show a brain tumor?{ANSWER_PREFIX}{lab}."
        })
    # target placeholder (no 'Answer:' cue here)
    msgs[0]["content"].append({"type":"image"})
    msgs[0]["content"].append({
        "type":"text",
        "text":"Does this MRI show a brain tumor? Answer with 'yes' or 'no'."
    }) # Add the question for the target image, but WITHOUT revealing the answer.
    return msgs

def _blank_image(size=MAX_EDGE): #creates blank image 
    return Image.new("RGB", (size, size), color=(127,127,127))

def _images_for_support_and_target(support_pairs, target_path, max_edge=MAX_EDGE):
     # Build the ordered list of PIL images the model will see:
    # 4 support images (with labels shown in the text) + 1 target image to classify.
    imgs = []
    for pth, _ in support_pairs:
        # Loop over the 4 support examples; each is (image_path, label).
        imgs.append(downscale(Image.open(pth).convert("RGB"), max_edge))
        # Load the support image from disk, force RGB mode, shrink it to fit (max_edge),
        # then append it to the list.
    imgs.append(downscale(Image.open(target_path).convert("RGB"), max_edge))
    #load target image
    return imgs #return full list 

def _best_verbalizer_token(candidates, tok):
    # Define a helper that picks the candidate string that tokenizes to the FEWEST tokens.
    # Fewer tokens = cleaner, more stable scoring for the “first token” method.
    best, best_ids = None, None
    # Keep track of the best text and its token ids so far.
    for t in candidates:
        ids = tok.encode(t, add_special_tokens=False)
        # Use the tokenizer to convert the string into token IDs 
        if best is None or len(ids) < len(best_ids):
            best, best_ids = t, ids
            # Update the current best choice and its tokenization.
    return best, best_ids
    # Return the chosen text (e.g., " yes") and its token IDs.

YES_TEXT, YES_IDS = _best_verbalizer_token([" yes","Yes","yes"], processor.tokenizer)
# "yes"—pick the variant that becomes the fewest tokens.

NO_TEXT,  NO_IDS  = _best_verbalizer_token([" no","No","no"],   processor.tokenizer)
# Do the same for "no"—pick the variant that becomes the fewest tokens.

print(f"Chosen verbalizers -> YES: {YES_TEXT} (ids={YES_IDS}), NO: {NO_TEXT} (ids={NO_IDS})")


def _first_token_logprob(ft_model, processor, images_list, prompt_text, answer_text):
    # We want: how likely is the FIRST word of the answer?

    tok_p = processor(text=prompt_text, images=images_list, return_tensors="pt", padding=False).to(ft_model.device)
    # Turn the prompt + images into model inputs.
    Lp = tok_p["input_ids"].shape[-1]
# Count how many tokens are in the prompt
    
    tok_f = processor(text=prompt_text + answer_text, images=images_list, return_tensors="pt", padding=False).to(ft_model.device)
    # Turn the prompt + the answer into inputs too
    first_id = tok_f.input_ids[0, Lp].item()
    # Grab the token id of the FIRST answer token 

    with torch.no_grad():
        out = ft_model(**tok_p)
        # Run the model on the prompt only 
        next_logits = out.logits[:, -1, :]
        # Take the scores for the very next token position.
        logprobs   = F.log_softmax(next_logits, dim=-1)
        return logprobs[0, first_id].item()

def _calibrated_margin(ft_model, processor, images_list, chat_prompt):
    
    lp_yes = _first_token_logprob(ft_model, processor, images_list, chat_prompt, YES_TEXT)
    lp_no  = _first_token_logprob(ft_model, processor, images_list, chat_prompt,  NO_TEXT)
    #Get how likely the model thinks the first answer token is “yes” vs “no” for the real images you provided
    blanks = [_blank_image(MAX_EDGE) for _ in images_list]
    b_yes  = _first_token_logprob(ft_model, processor, blanks, chat_prompt, YES_TEXT)
    b_no   = _first_token_logprob(ft_model, processor, blanks, chat_prompt,  NO_TEXT)
    #Get how likely “yes” vs “no” are when the images carry no info
    return (lp_yes - lp_no) - (b_yes - b_no)

def predict_yes_no_4shot(ft_model, processor, support_pairs, target_path, delta=0.0):
    #function that predicts “yes” or “no” using 4-shot context
    imgs = _images_for_support_and_target(support_pairs, target_path)
    #Load and downscale the five images in the exact order the prompt expects
    chat_prompt = processor.apply_chat_template(build_4shot_messages(support_pairs),
                                                add_generation_prompt=True)
    #Build the chat-style text prompt
    margin = _calibrated_margin(ft_model, processor, imgs, chat_prompt)
    #Compute the debiased decision score (
    return ("yes" if margin >= delta else "no"), margin


def sample_files(folder, k=5, rng=random):
    #picks 5 images for 4 shot
    files = list_images(folder)
    files.sort()
    rng.shuffle(files)
    return files[:min(k, len(files))]

def pick_support_pairs_det(k_each=2, rng=random):
    #picks a fixed (deterministic) set of support examples for few-shot prompts.
    ys = sample_files(yes_dir, k=len(list_images(yes_dir)), rng=rng)
    ns = sample_files(no_dir,  k=len(list_images(no_dir)),  rng=rng)
    ys = ys[:k_each]; ns = ns[:k_each]
    sup = [(p, "yes") for p in ys] + [(p, "no") for p in ns]
    rng.shuffle(sup)
    return sup
    #Returns the final list of 4 tuples by default (2 yes + 2 no), ready to feed into your 4-shot prompt builder.

def calibrate_threshold(ft_model, processor, support_pairs, k_each=8, rng=random.Random(1234)):
    y_targets = sample_files(yes_dir, k=k_each, rng=rng)
    n_targets = sample_files(no_dir,  k=k_each, rng=rng)

    margins, labels = [], [] ## Prepare lists to store margins (model scores) and ground-truth labels
    chat_prompt = processor.apply_chat_template(build_4shot_messages(support_pairs),
                                                add_generation_prompt=True)
    for p in y_targets + n_targets:
        imgs = _images_for_support_and_target(support_pairs, p)
        # Compute the calibrated log-probability margin for this target (positive = tends toward "yes")
        m = _calibrated_margin(ft_model, processor, imgs, chat_prompt)
        margins.append(m)
        # Save label: 1 for YES targets, 0 for NO targets
        labels.append(1 if p in y_targets else 0)

    # Get sorted unique margins
    cs = sorted(set(margins))
    # Build a list of candidate thresholds:
    thresholds = [-1e9] + [(cs[i]+cs[i+1])/2 for i in range(len(cs)-1)] + [1e9] if len(cs) > 1 else [0.0]

    # Track the best accuracy and the corresponding threshold (Δ)
    best_acc, best_delta = -1.0, 0.0
    for d in thresholds:
        # Convert each margin into a binary prediction using threshold d (>= d => predict YES)
        preds = [1 if m >= d else 0 for m in margins]
        acc = sum(int(p==y) for p,y in zip(preds, labels)) / max(1,len(labels))
        if acc > best_acc:
            best_acc, best_delta = acc, d

    print(f"Calibrated Δ = {best_delta:.4f} on {len(labels)} samples (acc={best_acc:.2%})")
    # Return the optimal threshold Δ to use at inference time
    return best_delta
#this function learns the decision threshold Δ that best separates YES from NO on a small, 
#balanced calibration set by trying midpoints between observed margins and picking the one that 
#yields the highest accuracy.


def evaluate_many_targets(ft_model, processor, k_yes=10, k_no=10, seed=SEED):
    rng = random.Random(seed)
    support = pick_support_pairs_det(k_each=2, rng=rng) # Pick a fixed 4-shot support set: 2 YES + 2 NO.

    
    delta = calibrate_threshold(ft_model, processor, support, k_each=6, rng=rng)

    yes_targets = sample_files(yes_dir, k=k_yes, rng=rng)
    no_targets  = sample_files(no_dir,  k=k_no,  rng=rng)

    print("\n=== Support set (4-shot) ===")
    print([os.path.basename(p) + f" ({lab})" for p, lab in support])
# Show which 4 images (and labels) are used as few-shot examples.
    yes_correct = 0
    print("\n=== Predictions on YES targets ===")
    for p in yes_targets:
        pred, margin = predict_yes_no_4shot(ft_model, processor, support, p, delta) 
        # Make a 4-shot prediction on image p using threshold Δ; also get the confidence margin.
        print(f"{os.path.basename(p)} -> {pred}  (margin={margin:.3f})")
        yes_correct += int(pred == "yes")

    no_correct = 0
    print("\n=== Predictions on NO targets ===")
    for p in no_targets:
        pred, margin = predict_yes_no_4shot(ft_model, processor, support, p, delta)
        # Make a 4-shot prediction on image p using the same Δ and support set.
        print(f"{os.path.basename(p)} -> {pred}  (margin={margin:.3f})")
        no_correct += int(pred == "no")

    total = len(yes_targets) + len(no_targets)
    acc = (yes_correct + no_correct) / max(1, total)
    print(f"\n4-shot accuracy on {total} targets: {acc:.2%} "
          f"(YES correct: {yes_correct}/{len(yes_targets)}, "
          f"NO correct: {no_correct}/{len(no_targets)})")

evaluate_many_targets(ft_model, processor, k_yes=10, k_no=10, seed=SEED)
# Run the evaluation:
#  - Uses your fine-tuned model (ft_model) and processor
#  - Tests on 10 YES and 10 NO images
#  - Uses SEED for reproducible sampling
