from typing import Callable
import torch
from torch.utils.data import Dataset
import pyarrow as pa

from src.data.datatypes import (
    Batch,
    Lookups,
)


class BaseDataset(Dataset):
    def __init__(
        self,
        data: list[pa.RecordBatch],
        text_transform: Callable,
        label_transform: Callable,
        lookups: Lookups,
        split_name: str = "train",
    ) -> None:
        super().__init__()
        self.data = data
        self.split_name = split_name
        self.text_transform = text_transform
        self.label_transform = label_transform
        self.lookups = lookups

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        token_ids, targets, id, num_tokens, _ = self.data[idx]
        targets = self.label_transform(targets)
        return token_ids, targets, id, num_tokens

    def collate_fn(self, batch: tuple[list, list, list, list]) -> Batch:
        data, targets, ids, num_tokens = zip(*batch)
        data = self.text_transform.seq2batch(data)
        targets = self.label_transform.seq2batch(targets)
        ids = torch.tensor(ids)
        num_tokens = torch.tensor(num_tokens)
        return Batch(data=data, targets=targets, ids=ids, num_tokens=num_tokens)


class HuggingfaceDataset(Dataset):
    def __init__(
        self,
        data: list[pa.RecordBatch],
        text_transform: Callable,
        label_transform: Callable,
        lookups: Lookups,
        chunk_size: int = 512,
        split_name: str = "train",
    ) -> None:
        super().__init__()
        self.data = data
        self.split_name = split_name
        self.text_transform = text_transform
        self.label_transform = label_transform
        self.chunk_size = chunk_size
        self.lookups = lookups

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        token_ids, targets, id, num_tokens, attention_mask = self.data[idx]
        targets = self.label_transform(targets)
        return token_ids, targets, id, num_tokens, attention_mask

    def collate_fn(self, batch: tuple[list, list, list, list]) -> Batch:
        token_ids, targets, ids, num_tokens, attention_mask = zip(*batch)
        data = self.text_transform.seq2batch(token_ids, chunk_size=self.chunk_size)
        attention_mask = self.text_transform.seq2batch(
            attention_mask, chunk_size=self.chunk_size
        )
        targets = self.label_transform.seq2batch(targets)
        ids = torch.tensor(ids)
        num_tokens = torch.tensor(num_tokens)
        return Batch(
            data=data,
            targets=targets,
            ids=ids,
            num_tokens=num_tokens,
            attention_mask=attention_mask,
        )

