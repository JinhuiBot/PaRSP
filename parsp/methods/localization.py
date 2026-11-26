import gc

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from parsp.methods.utils import (
    _get_layers,
    transfer_output,
    get_bsvalues,
    get_fc2_params
)

from parsp.utils.common import (
    cuda_memory_cleanup,
    get_prob
)


class NeuronLocalization:

    def __init__(
            self,
            model,
            tokenizer: PreTrainedTokenizerBase,
            device: str = "cuda:0"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.config = model.config
        self.layers = _get_layers(model)

        self.model_type = self.config.model_type.lower()
        self.is_llama = self.model_type == "llama"
        self.is_qwen = self.model_type.startswith("qwen")

        if not (self.is_llama or self.is_qwen):
            raise ValueError(
                f"Unsupported model type: {self.model_type}"
            )

        self.n_layers = len(self.layers)
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size

        if self.n_layers == 0:
            raise ValueError("No Transformer layers found.")

        self.attn_size = (
            self.layers[0].self_attn.o_proj.weight.shape[1]
        )

        if self.attn_size % self.n_head != 0:
            raise ValueError(
                "Attention dimension is incompatible with "
                "the number of attention heads."
            )

        self.head_dim = self.attn_size // self.n_head

        self.baseline_activations = None
        self.transformer_layers_attr = "model.layers"
        self.act_ff_attr = "mlp.act_fn"
        self.input_ff_attr = "mlp.up_proj"
        self.output_ff_attr = "mlp.down_proj.weight"
        self.ff_attr_format = "mlp.{}"
        self.attn_attr_format = "self_attn.{}"
        self.word_embeddings_attr = "model.embed_tokens.weight"

    def _prepare_inputs(
            self,
            prompt,
            ground_truth,
            max_length
    ):

        if self.is_llama:
            ground_truth = ground_truth.rstrip('." )\n,]')

            input_ids = self.tokenizer.encode(
                prompt + ground_truth,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

        else:
            input_ids = self.tokenizer(
                prompt + ground_truth,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).input_ids

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "Attribution expects a single input sequence."
            )

        if input_ids.shape[-1] < 2:
            raise ValueError(
                "Attribution requires at least two tokens."
            )

        predict_index = input_ids[0, -1].item()
        input_tokens = input_ids[:, :-1].to(self.device)

        return input_tokens, predict_index

    @torch.no_grad()
    def _capture_activations(
            self,
            prompt,
            ground_truth,
            max_length,
            stage
    ):

        input_tokens, predict_index = self._prepare_inputs(
            prompt,
            ground_truth,
            max_length
        )

        outputs = self.model(
            input_tokens,
            use_cache=True
        )

        captured = transfer_output(
            outputs[2],
            self.n_layers
        )

        del outputs

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        last_state = torch.as_tensor(
            captured[4][-1],
            dtype=torch.float32
        )
        if self.is_qwen or stage == "attn":
            last_state = last_state.to(self.device)

        last_state = last_state.reshape(
            -1, last_state.shape[-1]
        )[-1]

        final_var = torch.sqrt(
            torch.var(
                last_state,
                dim=-1,
                unbiased=False
            ) + 1e-5
        ).to(self.device)

        return captured, predict_index, final_var

    def _validate_layers(self, layer_idxs):

        layer_idxs = list(layer_idxs)

        for idx in layer_idxs:
            if not 0 <= idx < self.n_layers:
                raise IndexError(
                    f"Invalid layer index: {idx}."
                )

        if len(layer_idxs) != len(set(layer_idxs)):
            raise ValueError(
                "Duplicate layer indices are not allowed."
            )

        return layer_idxs

    @staticmethod
    def _validate_ratio(ratio):

        if not 0 <= ratio <= 1:
            raise ValueError(
                f"Ratio must be within [0, 1], got {ratio}."
            )

    @staticmethod
    def _validate_samples(prompts, ground_truths):

        if len(prompts) != len(ground_truths):
            raise ValueError(
                "Prompts and ground truths must have equal lengths."
            )

        if len(prompts) == 0:
            raise ValueError(
                "Attribution requires at least one sample."
            )

    def _target_log_prob(
            self,
            hidden_states,
            predict_index,
            final_var
    ):

        logits = get_bsvalues(
            hidden_states,
            self.model,
            final_var
        )

        probs = get_prob(logits)

        return torch.log(
            probs[..., predict_index]
        )

    def _log_prob_gain(
            self,
            base,
            contribution,
            predict_index,
            final_var
    ):

        base_log_prob = self._target_log_prob(
            base,
            predict_index,
            final_var
        )

        plus_log_prob = self._target_log_prob(
            base + contribution,
            predict_index,
            final_var
        )

        return plus_log_prob - base_log_prob

    @cuda_memory_cleanup(debug=True)
    @torch.no_grad()
    def get_top_attribute_layers(
            self,
            prompts: list,
            ground_truths: list,
            layer_ratio: float,
            max_length: int = 1024
    ):

        self._validate_ratio(layer_ratio)
        self._validate_samples(prompts, ground_truths)

        top_cnt = max(
            1,
            round(self.n_layers * layer_ratio)
        )

        total_attn_log = [0.0] * self.n_layers
        total_ffn_log = [0.0] * self.n_layers

        for prompt, ground_truth in tqdm(
                zip(prompts, ground_truths),
                total=len(prompts),
                desc="Calculating layer attribution"
        ):
            captured, predict_index, final_var = (
                self._capture_activations(
                    prompt,
                    ground_truth,
                    max_length,
                    stage="layer"
                )
            )

            (
                all_pos_layer_input,
                all_pos_attn_output,
                all_pos_residual_output,
                all_pos_ffn_output,
                _,
                _,
                _,
                _
            ) = captured

            for layer_idx in range(self.n_layers):

                layer_input = torch.as_tensor(
                    all_pos_layer_input[layer_idx],
                    device=self.device
                )

                attn_output = torch.as_tensor(
                    all_pos_attn_output[layer_idx],
                    device=self.device
                )

                residual = torch.as_tensor(
                    all_pos_residual_output[layer_idx],
                    device=self.device
                )

                ffn_output = torch.as_tensor(
                    all_pos_ffn_output[layer_idx],
                    device=self.device
                )

                if self.is_llama:
                    layer_input = layer_input.reshape(-1)
                    attn_output = attn_output.reshape(-1)
                    residual = residual.reshape(-1)
                    ffn_output = ffn_output.reshape(-1)

                else:
                    layer_input = layer_input.reshape(1, -1)
                    attn_output = attn_output.reshape(1, -1)
                    residual = residual.reshape(1, -1)
                    ffn_output = ffn_output.reshape(1, -1)

                attn_gain = self._log_prob_gain(
                    layer_input,
                    attn_output,
                    predict_index,
                    final_var
                )

                total_attn_log[layer_idx] += attn_gain.item()

                ffn_gain = self._log_prob_gain(
                    residual,
                    ffn_output,
                    predict_index,
                    final_var
                )

                total_ffn_log[layer_idx] += ffn_gain.item()

        if self.is_llama:
            attn_sort = sorted(
                enumerate(total_attn_log),
                key=lambda x: x[1]
            )[::-1]

            ffn_sort = sorted(
                enumerate(total_ffn_log),
                key=lambda x: x[1]
            )[::-1]

        else:
            attn_sort = sorted(
                enumerate(total_attn_log),
                key=lambda x: x[1],
                reverse=True
            )

            ffn_sort = sorted(
                enumerate(total_ffn_log),
                key=lambda x: x[1],
                reverse=True
            )

        print(f"Top Attention layers: {attn_sort[:top_cnt]}")
        print(f"Top FFN layers: {ffn_sort[:top_cnt]}")

        top_attn_layer_idx = [
            idx for idx, _ in attn_sort[:top_cnt]
        ]

        top_ffn_layer_idx = [
            idx for idx, _ in ffn_sort[:top_cnt]
        ]

        return top_attn_layer_idx, top_ffn_layer_idx

    @torch.no_grad()
    def _get_top_attribute_neurons(
            self,
            prompts,
            ground_truths,
            layer_idxs,
            neuron_ratio,
            top,
            batch_size,
            max_length,
            module_type
    ):

        self._validate_ratio(neuron_ratio)
        self._validate_samples(prompts, ground_truths)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        layer_idxs = self._validate_layers(layer_idxs)

        if not layer_idxs:
            return []

        if module_type == "ffn":
            total_neurons = self.intermediate_size

        elif module_type == "attn":
            total_neurons = self.attn_size

        else:
            raise ValueError(
                f"Unsupported module type: {module_type}"
            )

        neuron_num = max(
            1,
            round(total_neurons * neuron_ratio)
        )

        scores = torch.zeros(
            (self.n_layers, total_neurons),
            device=self.device,
            dtype=torch.float32
        )

        for prompt, ground_truth in tqdm(
                zip(prompts, ground_truths),
                total=len(prompts),
                desc=f"Analyzing {module_type.upper()} neurons"
        ):
            captured, predict_index, final_var = (
                self._capture_activations(
                    prompt,
                    ground_truth,
                    max_length,
                    stage=module_type
                )
            )

            (
                all_pos_layer_input,
                _,
                all_pos_residual_output,
                _,
                _,
                all_last_attn_subvalues,
                all_pos_coefficient_scores,
                _
            ) = captured

            for layer_idx in layer_idxs:

                if module_type == "ffn":

                    coefficients = torch.as_tensor(
                        all_pos_coefficient_scores[layer_idx],
                        device=self.device
                    ).reshape(-1)

                    weight_vectors = get_fc2_params(
                        self.model,
                        layer_idx
                    ).T.to(self.device)

                    base = torch.as_tensor(
                        all_pos_residual_output[layer_idx],
                        device=self.device
                    ).reshape(1, -1)

                else:

                    coefficients = torch.as_tensor(
                        all_last_attn_subvalues[layer_idx],
                        device=self.device
                    ).reshape(-1)

                    weight_vectors = (
                        self.layers[layer_idx]
                        .self_attn.o_proj.weight
                        .detach().T.to(self.device)
                    )

                    base = torch.as_tensor(
                        all_pos_layer_input[layer_idx],
                        device=self.device
                    ).reshape(1, -1)

                if coefficients.numel() != total_neurons:
                    raise ValueError(
                        f"Activation dimension mismatch in "
                        f"{module_type} layer {layer_idx}: "
                        f"expected {total_neurons}, "
                        f"got {coefficients.numel()}."
                    )

                if weight_vectors.shape != (
                        total_neurons,
                        base.shape[-1]
                ):
                    raise ValueError(
                        f"Projection weight dimension mismatch "
                        f"in {module_type} layer {layer_idx}: "
                        f"expected "
                        f"{(total_neurons, base.shape[-1])}, "
                        f"got {tuple(weight_vectors.shape)}."
                    )

                base_log_prob = self._target_log_prob(
                    base,
                    predict_index,
                    final_var
                ).reshape(())

                for start in range(
                        0,
                        total_neurons,
                        batch_size
                ):
                    end = min(
                        start + batch_size,
                        total_neurons
                    )

                    batch_coefficients = (
                        coefficients[start:end].unsqueeze(1)
                    )

                    batch_weights = weight_vectors[start:end]

                    batch_contributions = (
                        batch_coefficients * batch_weights
                    )

                    batch_input = base + batch_contributions

                    batch_log_prob = self._target_log_prob(
                        batch_input,
                        predict_index,
                        final_var
                    ).reshape(-1)

                    batch_gain = (
                        batch_log_prob - base_log_prob
                    )

                    scores[layer_idx, start:end] += (
                        batch_gain.to(
                            device=scores.device,
                            dtype=scores.dtype
                        )
                    )

                    del batch_coefficients
                    del batch_weights
                    del batch_contributions
                    del batch_input
                    del batch_log_prob
                    del batch_gain

                del coefficients
                del weight_vectors
                del base
                del base_log_prob

        scores = scores.cpu()

        selected_indices = torch.argsort(
            scores[layer_idxs],
            dim=-1,
            descending=top
        )[:, :neuron_num]

        selected_neurons = []

        for i, layer_idx in enumerate(layer_idxs):
            for neuron_idx in selected_indices[i].tolist():
                selected_neurons.append(
                    [layer_idx, neuron_idx]
                )

        return selected_neurons

    @torch.no_grad()
    def get_top_attribute_neurons_ffn(
            self,
            prompts,
            ground_truths,
            layer_idxs,
            neuron_ratio,
            top=True,
            batch_size=4096,
            max_length=1024
    ):

        return self._get_top_attribute_neurons(
            prompts=prompts,
            ground_truths=ground_truths,
            layer_idxs=layer_idxs,
            neuron_ratio=neuron_ratio,
            top=top,
            batch_size=batch_size,
            max_length=max_length,
            module_type="ffn"
        )

    @torch.no_grad()
    def get_top_attribute_neurons_attn(
            self,
            prompts,
            ground_truths,
            layer_idxs,
            neuron_ratio,
            top=True,
            batch_size=4096,
            max_length=1024
    ):

        return self._get_top_attribute_neurons(
            prompts=prompts,
            ground_truths=ground_truths,
            layer_idxs=layer_idxs,
            neuron_ratio=neuron_ratio,
            top=top,
            batch_size=batch_size,
            max_length=max_length,
            module_type="attn"
        )