import os
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_a_mask

MODEL_PATH = "models/Qwen2.5-32B-Instruct-GPTQ-Int4"
OUTPUT_DIR = "./descriptors_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_GROUPS = {
    "natural_objects": [
        "ImageNet", "ImageNet-V2", "ImageNet-A", "Caltech101", 
        "OxfordPets", "Flowers102", "Food101", "StanfordCars", "FGVC-Aircraft"
    ],
    "sketch": ["ImageNet-Sketch"],
    "rendition": ["ImageNet-R"],
    "texture": ["DTD"],
    "scene_action": ["SUN397", "EuroSAT", "UCF101"]
}

SYSTEM_PROMPTS = {
    "natural_objects": {
        "prefix": "A photo of a",
        "instruction": """You are a Computer Vision expert. 
Task: Generate a visual caption for the provided class name from a natural image dataset.
Format: JSON key-value pair {"class_name": "caption"}.
Length: Under 20 words.
Content: Focus on DISTINCTIVE visual features (color, shape, specific parts like 'wings', 'wheels', 'fur', 'petals').
Template: "A photo of a [CLASS], featuring [FEATURE 1] and [FEATURE 2]." """
    },
    
    "sketch": {
        "prefix": "A sketch of a",
        "instruction": """You are an expert in analyzing pencil sketches. The dataset consists of BLACK AND WHITE sketches.
Task: Generate a visual caption for the class as it appears in a sketch.
Format: JSON key-value pair {"class_name": "caption"}.
CRITICAL: Do NOT use color words (red, blue, etc.). Focus on OUTLINE, SHAPE, and LINE WORK.
Length: Under 20 words.
Template: "A sketch of a [CLASS], depicted with [STRUCTURAL FEATURE] and [LINE STYLE]." """
    },

    "rendition": {
        "prefix": "A rendering of a",
        "instruction": """You are analyzing art renditions (cartoons, origami, paintings) of objects.
Task: Generate a visual caption.
Format: JSON key-value pair {"class_name": "caption"}.
Focus: Describe the INTRINSIC STRUCTURE that makes the object recognizable across styles. Avoid realistic textures unless defining.
Length: Under 20 words.
Template: "A rendering of a [CLASS], displaying its characteristic [STRUCTURAL FEATURE 1] and [STRUCTURAL FEATURE 2]." """
    },

    "texture": {
        "prefix": "A texture of",
        "instruction": """You are describing surface textures.
Task: Generate a visual caption for the texture pattern.
Format: JSON key-value pair {"class_name": "caption"}.
Focus: Repetitive patterns, surface feel, and geometric arrangement.
Length: Under 20 words.
Template: "A texture of [CLASS] surface, characterized by [PATTERN DETAILS]." """
    },

    "scene_action": {
        "prefix": "A photo of",
        "instruction": """You are creating captions for Scenes, Satellite Views, or Human Actions.
Task: Generate a visual caption.
Format: JSON key-value pair {"class_name": "caption"}.
Logic: 
- If Scene/Satellite: Describe layout/environment.
- If Action: Describe pose and interaction.
Length: Under 20 words.
Template: "A photo of [CLASS], showing [LAYOUT OR ACTION DETAILS]." """
    }
}

def get_strategy(dataset_name):
    for group, names in DATASET_GROUPS.items():
        if dataset_name in names:
            return SYSTEM_PROMPTS[group]
    return SYSTEM_PROMPTS["natural_objects"]

def clean_json_output(text, class_name):
    data = None
    try:
        data = json.loads(text)
    except:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except:
                pass
        
        if data is None:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except:
                    pass

    if not isinstance(data, dict):
        print(f"  [Parse Error] Raw output for {class_name}: {text[:50]}...")
        return {class_name: f"A photo of a {class_name}."}

    if class_name in data:
        return {class_name: data[class_name]}

    for k, v in data.items():
        if k.lower() == class_name.lower():
            return {class_name: v}

    if len(data) == 1:
        return {class_name: list(data.values())[0]}

    common_keys = ["caption", "description", "visual_description", "text"]
    for key in common_keys:
        if key in data:
            return {class_name: data[key]}
        
    print(f"  [Key Error] Could not find key for {class_name} in {data}")
    return {class_name: f"A photo of a {class_name}."}
    
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto", 
    trust_remote_code=True,
    torch_dtype=torch.float16
)

def generate_dataset_captions(dataset_name, class_list):
    print(f"\nProcessing Dataset: {dataset_name} ({len(class_list)} classes)")
    
    strategy = get_strategy(dataset_name)
    system_prompt = strategy["instruction"]
    results = {}

    for class_name in tqdm(class_list):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": class_name}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                temperature=0.7, 
                top_p=0.9,
                do_sample=True
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        json_data = clean_json_output(response_text, class_name)
        
        caption = json_data.get(class_name, json_data.get(class_name.lower(), ""))

        if not caption:
             caption = f"{strategy['prefix']} {class_name}."
             
        results[class_name] = caption

    save_path = os.path.join(OUTPUT_DIR, f"descriptions_{dataset_name}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    imagenet_a_classes = [imagenet_classes[i] for i in imagenet_a_mask]
    generate_dataset_captions("ImageNet-A", imagenet_a_classes)