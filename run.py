import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel, ViTModel, Qwen2VLForConditionalGeneration
from data import M3DVQA
from model import GraphVQA


def required(config, key):
    value = config.get(key)
    if value is None:
        raise ValueError(f'Configuration requires an explicit value: {key}')
    return value


def validate(config, mode):
    for key in ('vision_checkpoint', 'vision_revision', 'text_checkpoint', 'text_revision', 'language_checkpoint', 'language_revision', 'graph_layers', 'dtype', 'seed', 'graph_special_tokens', 'question_special_tokens', 'answer_eos', 'preprocessing'):
        required(config, key)
    if config['dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('dtype must be float32 or bfloat16')
    if not isinstance(config['graph_layers'], int) or config['graph_layers'] < 1:
        raise ValueError('graph_layers must be a positive integer')
    for key in ('graph_special_tokens', 'question_special_tokens', 'answer_eos'):
        if not isinstance(config[key], bool):
            raise ValueError(f'{key} must be a boolean')
    if mode == 'train':
        training = required(config, 'training')
        for key in ('optimizer', 'learning_rate', 'betas', 'epsilon', 'weight_decay', 'batch_size', 'epochs', 'schedule'):
            required(training, key)
        if training['optimizer'] not in ('Adam', 'AdamW') or training['schedule'] != 'constant':
            raise ValueError('Select Adam or AdamW with a constant schedule')
        if training['batch_size'] < 1 or training['epochs'] < 1 or training['learning_rate'] <= 0:
            raise ValueError('Positive batch_size, epochs and learning_rate are required')
    else:
        if required(config, 'max_new_tokens') < 1:
            raise ValueError('max_new_tokens must be positive')
        if not isinstance(required(config, 'eos_ids'), list):
            raise ValueError('eos_ids must be a list')


def load(config, device):
    vision_path = required(config, 'vision_checkpoint')
    text_path = required(config, 'text_checkpoint')
    language_path = required(config, 'language_checkpoint')
    vision = ViTModel.from_pretrained(vision_path, revision=required(config, 'vision_revision'), add_pooling_layer=False)
    text = BertModel.from_pretrained(text_path, revision=required(config, 'text_revision'), add_pooling_layer=False)
    language = Qwen2VLForConditionalGeneration.from_pretrained(language_path, revision=required(config, 'language_revision'))
    if language.config.hidden_size != 1536 or language.config.num_hidden_layers != 28:
        raise ValueError('This entry point requires the paper-listed Qwen2-VL 2B backbone')
    model = GraphVQA(vision, text, language, required(config, 'graph_layers'))
    model.to(device=device, dtype=getattr(torch, required(config, 'dtype')))
    graph_tokenizer = AutoTokenizer.from_pretrained(text_path, revision=config['text_revision'])
    language_tokenizer = AutoTokenizer.from_pretrained(language_path, revision=config['language_revision'])
    return model, graph_tokenizer, language_tokenizer


def encode(item, model, graph_tokenizer, language_tokenizer, config, device):
    graph_ids = graph_tokenizer.encode(item['question'], add_special_tokens=required(config, 'graph_special_tokens'))
    question_ids = language_tokenizer.encode(item['question'], add_special_tokens=required(config, 'question_special_tokens'))
    answer_ids = language_tokenizer.encode(item['answer'], add_special_tokens=False)
    if required(config, 'answer_eos'):
        answer_ids.append(language_tokenizer.eos_token_id)
    if not graph_ids or not question_ids or not answer_ids:
        raise ValueError('Question and answer token sequences must be nonempty')
    inputs = {'pixels': item['pixels'].to(device=device, dtype=next(model.parameters()).dtype), 'graph_question_ids': torch.tensor(graph_ids, device=device), 'question_ids': torch.tensor(question_ids, device=device)}
    return inputs, torch.tensor(answer_ids, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate(config, args.mode)
    seed = required(config, 'seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = M3DVQA(args.csv, args.data_root, required(config, 'preprocessing'))
    if not len(dataset):
        raise ValueError('Dataset is empty')
    model, graph_tokenizer, language_tokenizer = load(config, args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if checkpoint['config'] != config:
            raise ValueError('Checkpoint configuration differs from requested configuration')
        model.load_state_dict(checkpoint['model'], strict=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'train':
        training = required(config, 'training')
        optimizer_name = required(training, 'optimizer')
        if optimizer_name not in ('Adam', 'AdamW'):
            raise ValueError('Supported optimizers: Adam, AdamW')
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=required(training, 'learning_rate'), betas=tuple(required(training, 'betas')), eps=required(training, 'epsilon'), weight_decay=required(training, 'weight_decay'))
        batch_size = required(training, 'batch_size')
        epochs = required(training, 'epochs')
        if batch_size < 1 or epochs < 1:
            raise ValueError('batch_size and epochs must be positive')
        if required(training, 'schedule') != 'constant':
            raise ValueError('The implemented optimizer schedule is constant')
        model.train()
        for epoch in range(epochs):
            indices = torch.randperm(len(dataset)).tolist()
            total = 0.0
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                for index in batch:
                    inputs, answer = encode(dataset[index], model, graph_tokenizer, language_tokenizer, config, args.device)
                    loss = model(**inputs, answer_ids=answer)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    (loss / len(batch)).backward()
                    total += loss.detach().item()
                optimizer.step()
            print(json.dumps({'epoch': epoch + 1, 'loss': total / len(dataset)}), flush=True)
            torch.save({'model': model.state_dict(), 'config': config}, output)
    else:
        if not args.checkpoint:
            raise ValueError('Prediction requires a trained --checkpoint')
        model.eval()
        eos_ids = required(config, 'eos_ids')
        with output.open('w') as stream:
            for index in range(len(dataset)):
                item = dataset[index]
                inputs, _ = encode(item, model, graph_tokenizer, language_tokenizer, config, args.device)
                ids = model.generate(**inputs, max_new_tokens=required(config, 'max_new_tokens'), eos_ids=eos_ids)
                record = {key: item[key] for key in ('index', 'question', 'answer', 'category')}
                record['prediction'] = language_tokenizer.decode(ids, skip_special_tokens=True)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
