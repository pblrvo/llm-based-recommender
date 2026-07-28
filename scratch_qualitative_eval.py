import sys, random
sys.path.insert(0, 'src')
from pathlib import Path

from evaluate_ranking_metrics import load_model, load_val_examples_by_task
from constrained_decoding import build_sid_trie, load_catalog, semantic_id_to_tokens, constrained_beam_search

project_root = Path('.').resolve()
adapter_path = project_root / 'models' / 'qwen3-4b-qlora-v2'

model, tokenizer = load_model(adapter_path)
catalog = load_catalog(project_root)
sid_trie = build_sid_trie(tokenizer, catalog)

sid_to_name = {}
for row in catalog.iter_rows(named=True):
    sid_to_name[semantic_id_to_tokens(row['semantic_ids'])] = row['Name']


def split_sids(text):
    parts = [p.strip() for p in text.split('<|sid_end|>') if p.strip()]
    return [p + '<|sid_end|>' for p in parts]


def names_for(text):
    return [sid_to_name.get(s, f'UNKNOWN({s})') for s in split_sids(text)]


examples_by_task = load_val_examples_by_task(project_root / 'data' / 'output' / 'sft_val.jsonl')
random.seed(1)

for task in ['sequential', 'similar_item']:
    print(f'\n{"=" * 70}\n{task.upper()} -- 10 examples\n{"=" * 70}')
    sample = random.sample(examples_by_task[task], min(10, len(examples_by_task[task])))
    hits = 0
    for i, ex in enumerate(sample, 1):
        messages = [{'role': 'user', 'content': f"{ex['instruction']}\n{ex['input']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        candidates = constrained_beam_search(model, tokenizer, prompt, sid_trie, num_beams=10, temperature=0.8)
        top1 = candidates[0] if candidates else None

        input_names = names_for(ex['input'])
        expected_name = sid_to_name.get(ex['output'], f"UNKNOWN({ex['output']})")
        predicted_name = sid_to_name.get(top1, f"UNKNOWN({top1})") if top1 else "NO CANDIDATE"
        match = predicted_name == expected_name
        hits += match

        print(f'\n--- example {i} {"[HIT]" if match else "[MISS]"} ---')
        if task == 'sequential':
            print(f'  history (most->least recent... actually oldest->newest per input order): {input_names}')
        else:
            print(f'  seed item: {input_names}')
        print(f'  expected : {expected_name}')
        print(f'  predicted: {predicted_name}')
        if not match and len(candidates) > 1:
            other_names = [sid_to_name.get(c, f'UNKNOWN({c})') for c in candidates[1:5]]
            print(f'  next candidates (rank 2-5): {other_names}')

    print(f'\n{task}: {hits}/10 top-1 exact matches')
