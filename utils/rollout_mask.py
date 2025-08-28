import types


def apply_masked_generate(trainer, logits_processor):
    """Bind a generate method on the unwrapped policy that enforces logits_processor.

    Works for GRPO where generation happens via the unwrapped model.
    """
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    _orig_generate = unwrapped.generate

    def _generate_with_mask(self, *g_args, **g_kwargs):
        lps = g_kwargs.get("logits_processor", None)
        if lps is None:
            g_kwargs["logits_processor"] = [logits_processor]
        else:
            l = list(lps)
            if logits_processor not in l:
                l = [logits_processor] + l
            g_kwargs["logits_processor"] = l
        return _orig_generate(*g_args, **g_kwargs)

    unwrapped.generate = types.MethodType(_generate_with_mask, unwrapped)


