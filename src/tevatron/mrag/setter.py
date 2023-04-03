def fid_setter(model, passage_ids, passage_masks, target_ids, n_context, gates, placeholder):
    # change encoder embedding output
    def hook(module, inputs, outputs=None):

        # encoder embedding layer outputs bsz*n_context,passage_len,768

        temp, passage_len, dim = outputs.shape
        bsz = int(temp / n_context)
        new_encoder_embedding = outputs.view(bsz, n_context, -1)
        # bsz,n_context,len*dim
        new_encoder_embedding = new_encoder_embedding * gates.unsqueeze(-1) + placeholder * (1 - gates).unsqueeze(
            -1)
        new_encoder_embedding = new_encoder_embedding.view(bsz * n_context, passage_len, dim)
        return new_encoder_embedding

    handles = (
        [model.encoder.encoder.embed_tokens.register_forward_hook(hook)]
    )

    try:
        loss, logits = model(input_ids=passage_ids, attention_mask=passage_masks, labels=target_ids)[:2]
    finally:
        for handle in handles:
            handle.remove()

    return loss, logits
