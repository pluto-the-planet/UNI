import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm

from ..get_encoder import get_encoder

torch.multiprocessing.set_sharing_strategy("file_system")

@torch.no_grad()
def extract_patch_features_from_dataloader(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, (2) [N x 1]-dim np.array of labels,
              and (3) [N x 1]-dim list of patient names.
    """
    all_embeddings, all_labels, all_patient_names = [], [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters()).device

    for batch_idx, (patient_names, labels, images) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = images.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + images.shape[1:]).type(
                images.type()
            )
            images = torch.vstack([images, _])

        images = images.to(device)
        with torch.inference_mode():
            embeddings = model(images).detach().cpu()[:remaining, :]
            labels = labels.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_patient_names.extend(patient_names[:remaining])  # Collect patient names

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
        "patient_names": all_patient_names,
    }

    return asset_dict
