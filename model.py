import torch
from torch import nn


class AttentiveGraphLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = nn.Parameter(torch.empty(width, width))
        self.transform = nn.Linear(width, width)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, nodes, adjacency):
        if not adjacency.any(-1).all():
            raise ValueError('Every graph node must have a neighbor')
        scores = (nodes @ self.attention) @ nodes.transpose(-1, -2)
        stable_scores = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
        weights = stable_scores.masked_fill(~adjacency, -torch.inf).softmax(-1)
        return weights.to(nodes.dtype) @ self.transform(nodes)


class CrossModalGraph(nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        if layers < 1:
            raise ValueError('graph_layers must be positive')
        self.layers = nn.ModuleList(AttentiveGraphLayer(width) for _ in range(layers))

    @staticmethod
    def adjacency(slices, tokens, device):
        if slices < 1 or tokens < 1:
            raise ValueError('A volume and question must each contain at least one node')
        edges = torch.zeros(slices + tokens, slices + tokens, dtype=torch.bool, device=device)
        index = torch.arange(slices - 1, device=device)
        edges[index, index + 1] = True
        edges[index + 1, index] = True
        edges[:slices, slices:] = True
        edges[slices:, :slices] = True
        return edges

    def forward(self, slices, tokens):
        if slices.shape[-1] != tokens.shape[-1]:
            raise ValueError('The paper requires a common slice/token feature dimension')
        nodes = torch.cat((slices, tokens), dim=0)
        adjacency = self.adjacency(len(slices), len(tokens), nodes.device)
        for layer in self.layers:
            nodes = layer(nodes, adjacency)
        return nodes


class GraphVQA(nn.Module):
    def __init__(self, vision, text, language, graph_layers):
        super().__init__()
        if vision.config.hidden_size != text.config.hidden_size:
            raise ValueError('Vision and text widths must match; no extra alignment layer is specified')
        self.vision = vision
        self.text = text
        self.language = language
        self.graph = CrossModalGraph(text.config.hidden_size, graph_layers)
        self.projection = nn.Linear(text.config.hidden_size, language.get_input_embeddings().embedding_dim)

    def soft_prompt(self, pixels, graph_question_ids):
        patch_states = self.vision(pixel_values=pixels).last_hidden_state[:, 1:]
        slices = patch_states.amax(dim=1)
        tokens = self.text(input_ids=graph_question_ids.unsqueeze(0)).last_hidden_state[0]
        return self.projection(self.graph(slices, tokens))

    def context(self, pixels, graph_question_ids, question_ids):
        prompt = self.soft_prompt(pixels, graph_question_ids)
        question = self.language.get_input_embeddings()(question_ids)
        return torch.cat((prompt, question), dim=0)

    def forward(self, pixels, graph_question_ids, question_ids, answer_ids):
        context = self.context(pixels, graph_question_ids, question_ids)
        answer = self.language.get_input_embeddings()(answer_ids)
        embeddings = torch.cat((context, answer), dim=0).unsqueeze(0)
        labels = torch.cat((torch.full((len(context),), -100, device=answer_ids.device), answer_ids)).long().unsqueeze(0)
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)[None, None].expand(3, 1, -1)
        return self.language(inputs_embeds=embeddings, attention_mask=torch.ones_like(labels), position_ids=positions, labels=labels, use_cache=False).loss

    @torch.no_grad()
    def generate(self, pixels, graph_question_ids, question_ids, max_new_tokens, eos_ids):
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        embeddings = self.context(pixels, graph_question_ids, question_ids).unsqueeze(0)
        result = []
        for _ in range(max_new_tokens):
            positions = torch.arange(embeddings.shape[1], device=embeddings.device)[None, None].expand(3, 1, -1)
            output = self.language(inputs_embeds=embeddings, attention_mask=torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long), position_ids=positions, use_cache=False)
            token = output.logits[0, -1].argmax()
            result.append(token.item())
            if token.item() in eos_ids:
                break
            embeddings = torch.cat((embeddings, self.language.get_input_embeddings()(token).reshape(1, 1, -1)), dim=1)
        return result
