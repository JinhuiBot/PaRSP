import torch.nn as nn
class ModelIntermediatesWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.captured_data = {}
        self.hooks = []
        self.layers = self._find_layers()
        self._register_hooks()

    def _find_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers

        if hasattr(self.model, "layers"):
            return self.model.layers

        raise ValueError(
            f"Unsupported model structure: "
            f"{self.model.__class__.__name__}"
        )

    def _register_hooks(self):

        def get_activation_hook(layer_idx, key, capture_input=False):

            def hook(module, inputs, output):
                if layer_idx not in self.captured_data:
                    self.captured_data[layer_idx] = {}

                if capture_input:
                    data = inputs[0]
                else:
                    data = output[0] if isinstance(output, tuple) else output

                self.captured_data[layer_idx][key] = data.detach().cpu()

            return hook

        self.remove_hooks()

        for i, layer in enumerate(self.layers):

            hook_configs = [
                (layer.input_layernorm, "layer_input", True),
                (layer.self_attn.o_proj, "attn_subvalues", True),
                (layer.self_attn.o_proj, "attn_output", False),
                (layer.mlp.down_proj, "ffn_inter", True),
                (layer.mlp.down_proj, "ffn_output", False),
                (layer, "layer_output", False),
            ]

            for module, key, capture_input in hook_configs:
                handle = module.register_forward_hook(
                    get_activation_hook(i, key, capture_input)
                )
                self.hooks.append(handle)

    def forward(self, *args, **kwargs):
        self.captured_data = {}

        outputs = self.model(*args, **kwargs)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
            past_key_values = getattr(outputs, "past_key_values", None)
        else:
            logits = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None

        return logits, past_key_values, self.captured_data

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return self.model.device

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()

        self.hooks = []

    def __enter__(self):
        if not self.hooks:
            self._register_hooks()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()