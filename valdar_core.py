import torch
from collections import deque
from transformers import AutoTokenizer, BrainGPTForCausalLM


class BrainCog(torch.nn.Module):
    """Very small SNN-like block encoding sensor signals."""

    def __init__(self, input_dim: int = 8, spike_dim: int = 32) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, spike_dim)
        self.spike_dim = spike_dim
        self.state = torch.zeros(spike_dim)

    def process_input(self, sensor_input):
        tensor = torch.tensor(sensor_input, dtype=torch.float32)
        encoded = torch.relu(self.encoder(tensor))
        spikes = (encoded > 0.5).float()
        self.state = spikes
        return spikes

    def apply_feedback(self, text_output: str) -> None:
        sentiment = self._sentiment_score(text_output)
        self.state = torch.clamp(self.state + sentiment, 0.0, 1.0)

    def _sentiment_score(self, text: str) -> torch.Tensor:
        positive = {"good", "great", "happy", "love", "excellent"}
        negative = {"bad", "sad", "hate", "terrible", "awful"}
        tokens = text.lower().split()
        score = sum(1 for t in tokens if t in positive) - sum(1 for t in tokens if t in negative)
        return torch.full((self.spike_dim,), 0.01 * score)


class BCBTBridge(torch.nn.Module):
    """Fuse BC spikes with BT token embeddings."""

    def __init__(self, spike_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(spike_dim, embed_dim)

    def merge(self, spikes: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
        injection = self.linear(spikes.to(token_embeddings.device))
        return token_embeddings + injection


class UnifiedMemory:
    """Manage short and long term memories."""

    def __init__(self, short_term_size: int = 5) -> None:
        self.short_term = deque(maxlen=short_term_size)
        self.long_term: list[str] = []

    def retrieve_context(self) -> str:
        return " ".join(self.short_term)

    def store_interaction(self, user_text: str, response_text: str) -> None:
        entry = f"U:{user_text} A:{response_text}"
        self.short_term.append(entry)
        self.long_term.append(entry)


class CrossSTDP:
    """Naive cross-modal STDP weight updater."""

    def __init__(self, spike_dim: int, embed_dim: int, lr: float = 0.001) -> None:
        self.lr = lr
        self.weights = torch.zeros(embed_dim, spike_dim)
        self.spike_dim = spike_dim
        self.embed_dim = embed_dim

    def transform(self, spikes: torch.Tensor) -> torch.Tensor:
        return torch.mv(self.weights.to(spikes.device), spikes)

    def update(self, pre_spikes: torch.Tensor, post_signal: torch.Tensor) -> None:
        delta = torch.ger(post_signal, pre_spikes)
        self.weights += self.lr * delta


class ValdarCore:
    """Central class coordinating BrainCog and BrainGPT."""

    def __init__(self, model_path: str, personality: str = "Valdar", moral_rules=None) -> None:
        self.bc = BrainCog()
        self.bt = BrainGPTForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.stdp = CrossSTDP(self.bc.spike_dim, self.bt.config.hidden_size)
        self.bridge = BCBTBridge(self.bc.spike_dim, self.bt.config.hidden_size)
        self.memory = UnifiedMemory()
        self.personality = personality
        self.moral_rules = moral_rules or []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bt.to(self.device)

    def _prepare(self, messages, bc_spikes: torch.Tensor):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        embeds = self.bt.get_input_embeddings()(model_inputs.input_ids)
        embeds = self.bridge.merge(bc_spikes, embeds[0]).unsqueeze(0)
        embeds[:, 0] += self.stdp.transform(bc_spikes)
        return embeds, model_inputs.attention_mask, model_inputs.input_ids

    def _generate_text(self, embeds, attention_mask, input_ids, max_new_tokens=50):
        with torch.no_grad():
            out = self.bt.generate(inputs_embeds=embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        new_tokens = out[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def process_cycle(self, sensor_input, user_text=None, max_new_tokens=50):
        # 1. Sensory processing via BrainCog
        bc_spikes = self.bc.process_input(sensor_input).to(self.device)

        # 2. Memory retrieval
        context = self.memory.retrieve_context()
        system_prompt = f"You are {self.personality}."
        if self.moral_rules:
            system_prompt += " Rules: " + "; ".join(self.moral_rules)
        if context:
            system_prompt += " Context: " + context

        # 3. Text generation with BC spikes injected
        user_text = user_text or ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        embeds, mask, ids = self._prepare(messages, bc_spikes)
        response = self._generate_text(embeds, mask, ids, max_new_tokens=max_new_tokens)

        # 4. Feedback to BrainCog
        self.bc.apply_feedback(response)

        # 5. Store interaction in memory
        self.memory.store_interaction(user_text, response)

        # 6. STDP adaptation
        self.stdp.update(bc_spikes, embeds[0].mean(dim=0))

        return response
