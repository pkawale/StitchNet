import timm
from torch import nn
from timm.models.resnet import ResNet
import copy


def split_resnet(model: ResNet, split_idx: int):
    children = [copy.deepcopy(child) for child in model.children()]
    preprocessing = children[:4]
    blocks = [block for seq in children[4:-2] for block in seq.children()]
    head = children[-2:]
    if split_idx == 0:
        part1 = nn.Sequential(*preprocessing)
        part2 = nn.Sequential(*blocks, *head)
    elif split_idx == len(blocks):
        part1 = nn.Sequential(*preprocessing, *blocks)
        part2 = nn.Sequential(*head)
    elif 0 < split_idx < len(blocks):
        part1 = nn.Sequential(*preprocessing, *blocks[:split_idx])
        part2 = nn.Sequential(*blocks[split_idx:], *head)
    else:
        raise ValueError(
            f"Invalid split index {split_idx} for ResNet model containing {len(blocks)} blocks."
        )
    return part1, part2


def get_splittable_range(model_name: str):
    """Get the range of valid split indices for the given model, inclusive. Returns min_idx, max_idx
    integers.
    """
    # TODO - this is a horrible hack. It shouldn't need to load the model or use try/except.
    model = timm.create_model(model_name, pretrained=False, num_classes=10)
    first_idx = 0
    last_idx = 0
    for idx in range(len(list(model.modules()))):
        try:
            split_resnet(model, idx)
            last_idx = idx
        except ValueError:
            break
    return first_idx, last_idx


def split_model(model: nn.Module, split_idx: int):
    if isinstance(model, ResNet):
        return split_resnet(model, split_idx)
    else:
        raise TypeError(f"Model type {type(model)} not (yet) supported.")


__all__ = [
    "split_model",
    "get_splittable_range",
]
