def get_batch_data(batch, device):
    input_ids = batch[0]['input_ids'].to(device)
    indexes = batch[0]['indexes'][0].to(device)
    cu_seqlens = batch[0]['cu_seqlens'].to(device)[0].squeeze(0)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    labels = batch[1].to(device)
    kwargs = {"indexes": indexes,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen}
    
    return input_ids, labels, kwargs
    