from __future__ import annotations

import random

import torch


class ForgeryCueBank:
    """Per-forward historical forgery cue bank."""

    def __init__(
        self,
        bank_len: int = 3,
        detach_bank: bool = True,
        shuffle_bank: bool = False,
        zero_bank: bool = False,
    ) -> None:
        self.bank_len = int(bank_len)
        self.detach_bank = bool(detach_bank)
        self.shuffle_bank = bool(shuffle_bank)
        self.zero_bank = bool(zero_bank)
        self._bank: list[torch.Tensor] = []

    def append(self, cue: torch.Tensor) -> None:
        if self.detach_bank:
            cue = cue.detach()
        self._bank.append(cue)
        if len(self._bank) > self.bank_len:
            self._bank = self._bank[-self.bank_len:]

    def last(self) -> torch.Tensor | None:
        if not self._bank:
            return None
        return self._bank[-1]

    def items(self, hist_len: int | None = None) -> list[torch.Tensor]:
        items = list(self._bank[-int(hist_len):] if hist_len else self._bank)
        if self.zero_bank:
            items = [torch.zeros_like(item) for item in items]
        if self.shuffle_bank and len(items) > 1:
            items = list(items)
            random.shuffle(items)
        return items

    def __len__(self) -> int:
        return len(self._bank)
