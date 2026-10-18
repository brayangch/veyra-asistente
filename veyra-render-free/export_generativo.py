"""Exporta a JSON compacto sólo los pesos públicos del generador de Veyra."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA = ROOT / "data"
PUBLIC = Path(__file__).resolve().parent
PUBLIC_DATA = PUBLIC / "data"


def main() -> None:
    package = torch.load(LOCAL_DATA / "modelo_generativo_veyra.pt", map_location="cpu", weights_only=True)
    weights = package["state_dict"]
    exported = {
        "format": "veyra-tiny-gru-numpy-v1",
        "trained_from_scratch": bool(package["trained_from_scratch"]),
        "vocabulary": package["vocabulary"],
        "config": package["config"],
        "embedding": weights["embedding.weight"].tolist(),
        "weight_ih": weights["rnn.weight_ih_l0"].tolist(),
        "weight_hh": weights["rnn.weight_hh_l0"].tolist(),
        "bias_ih": weights["rnn.bias_ih_l0"].tolist(),
        "bias_hh": weights["rnn.bias_hh_l0"].tolist(),
        "head_weight": weights["head.weight"].tolist(),
        "head_bias": weights["head.bias"].tolist(),
    }
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DATA / "modelo_generativo_publico.json").write_text(json.dumps(exported, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    shutil.copy2(LOCAL_DATA / "modelo_veyra_propio.json", PUBLIC_DATA / "modelo_veyra_propio.json")
    shutil.copy2(LOCAL_DATA / "modelo_generativo_veyra.json", PUBLIC_DATA / "modelo_generativo_veyra.json")
    print(f"Exportación pública lista: {PUBLIC_DATA}; no se copiaron bases de datos ni documentos.")


if __name__ == "__main__":
    main()
