import json, math, os

from transformers import LlamaTokenizer, LlamaModel
import torch
from tqdm import tqdm


HOME = os.environ["HOME"]
RESULTS_PATH = os.path.join(HOME, "results")

print("Load llama.")
tokenizer = LlamaTokenizer.from_pretrained("/groups/gac50547/edison/data/LLaMA-HF/llama-7b")
tokenizer.pad_token='[PAD]'

language_model = LlamaModel.from_pretrained(
                "/groups/gac50547/edison/data/LLaMA-HF/llama-7b", cache_dir=RESULTS_PATH,
                )
language_model.eval()
# Charades-STA
print("Extract embeddings for CharadesSTA.")

## Training
## Load annotations
print("Load train annotations.")
ann_file_path = "/home/aad13446xj/projects/TVG_adapters/preprocessing/charades-sta/charades_sta_train_tokens_w_objects.json"

aux = json.load(open(ann_file_path, 'r'))
dataset = aux['annotations']

anns = {}
size = int(round(len(dataset) * 1.))
counter = 0

for row in dataset[:size]:
    if float(row['feature_start']) > float(row['feature_end']):
        print(row)
        continue

    if math.floor(float(row['feature_end'])) >= float(row['number_features']):
        row['feature_end'] = float(row['number_features'])-1

    row['augmentation'] = 0
    anns[counter] = row
    counter+=1
print(" Ok! {}".format(len(anns.keys())))

llama_emb_path = "/groups/gac50547/aad13446xj/charades_sta/llama_embeddings"
## Get embeddings
print("Get embeddings.")
counter = 0
queries = []
videos = []
import pdb; pdb.set_trace()
for key, ann in tqdm(anns.items()):
    queries.append(ann['description'])
    videos.append(ann['video'])
    counter += 1 

    if counter % 64 == 0:
        tokenizer_output = tokenizer(
            queries,
            return_tensors="pt",
            max_length=30,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_length=True,
        )

        with torch.no_grad():
            output = language_model(
                            input_ids      = tokenizer_output["input_ids"],
                            attention_mask = tokenizer_output["attention_mask"], 
                            output_hidden_states=True
                        ) 
            
        sequences = output.last_hidden_state
        senquences_lenght = tokenizer_output["length"]

        for i in range(len(queries)):

            filename ="{}/{}.pth".format(llama_emb_path, videos[i])
            this_sequence = sequences[i][:int(senquences_lenght[i])]
            torch.save(this_sequence, filename)
        
        queries = []
        videos = []
        counter = 0

tokenizer_output = tokenizer(
    queries,
    return_tensors="pt",
    max_length=30,
    truncation=True,
    padding=True,
    add_special_tokens=True,
    return_length=True,
)

output = language_model(
                input_ids      = tokenizer_output["input_ids"],
                attention_mask = tokenizer_output["attention_mask"], 
                output_hidden_states=True
            ) 

sequences = output.last_hidden_state
senquences_lenght = tokenizer_output["length"]

for i in range(len(queries)):

    filename ="{}/{}.pth".format(llama_emb_path, videos[i])
    this_sequence = torch.squeeze(sequences[i])
    torch.save(sequences, filename)


import pdb; pdb.set_trace()

## Test
## Load annotations
print("Load test annotations.")
ann_file_path = "/home/aad13446xj/projects/TVG_adapters/preprocessing/charades-sta/charades_sta_test_tokens_w_objects.json"

aux = json.load(open(ann_file_path, 'r'))
dataset = aux['annotations']

anns = {}
size = int(round(len(dataset) * 1.))
counter = 0

for row in dataset[:size]:
    if float(row['feature_start']) > float(row['feature_end']):
        print(row)
        continue

    if math.floor(float(row['feature_end'])) >= float(row['number_features']):
        row['feature_end'] = float(row['number_features'])-1

    row['augmentation'] = 0
    anns[counter] = row
    counter+=1
print(" Ok! {}".format(len(anns.keys())))

llama_emb_path = "/groups/gac50547/aad13446xj/charades_sta/llama_embeddings"
print("Get embeddings.")
## Get embeddings
for key, ann in anns.items():
    query = ann['description']

    tokenizer_output = tokenizer(
        query,
        return_tensors="pt",
        max_length=30,
        truncation=True,
        padding=True,
        add_special_tokens=True,
        return_length=True,
    )

    query_info = {}
    query_info["tokens_lengths"]  = tokenizer_output["length"]


    output = language_model(
                    input_ids      = tokenizer_output["input_ids"],
                    attention_mask = tokenizer_output["attention_mask"], 
                    output_hidden_states=True
                ) 
    
    sequences = output.last_hidden_state
    sequences = torch.squeeze(sequences)
    
    filename ="{}/{}.pth".format(llama_emb_path, ann['video'])
    torch.save(sequences, filename)

print("End of script.")

