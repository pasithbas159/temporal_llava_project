import torch
from transformers import LlavaForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class LlavaWithLoss(LlavaForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, visual_token_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_token_mask=visual_token_mask,
            labels=labels,
            **kwargs
        )
        logits = outputs.get('logits', outputs[0] if isinstance(outputs, dict) else outputs[0])
        loss = None
        if labels is not None and logits is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get('past_key_values', None),
            hidden_states=outputs.get('hidden_states', None),
            attentions=outputs.get('attentions', None),
            cross_attentions=outputs.get('cross_attentions', None)
        )
