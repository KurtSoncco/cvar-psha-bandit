"""CLI training entry point: trains EQ-JEPA and writes CSEP-style rates.

Trains with a horizon-conditioned negative-binomial head (see model.py): the same
issue-time windows are replayed at several ``horizons`` (days) via a ConcatDataset so
the model actually sees varying horizon_days during training and learns a real
horizon response, rather than the single constant it would otherwise memorize.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, DataLoader
import yaml
from cvar_psha.eq_jepa.catalog import CatalogQuery, get_catalog_client
from cvar_psha.eq_jepa.data import CatalogWindowDataset, RegionGrid
from cvar_psha.eq_jepa.model import EQJEPA, EQJEPAConfig

def load_events(cfg):
    source=cfg["catalog"]["source"]; client=get_catalog_client(source)
    if source.lower() in {"kiknet","k-net","knet","local","cached"}: return client.fetch_csv(cfg["catalog"]["csv"])
    q=CatalogQuery(**{**cfg["catalog"]["query"], "start_time": datetime.fromisoformat(cfg["catalog"]["query"]["start_time"]).replace(tzinfo=timezone.utc), "end_time": datetime.fromisoformat(cfg["catalog"]["query"]["end_time"]).replace(tzinfo=timezone.utc)})
    return client.fetch(q)

def build_multi_horizon_dataset(events, g, times, windows, horizons):
    datasets=[]
    for h in horizons:
        w=dict(windows); w["horizon_days"]=h
        datasets.append(CatalogWindowDataset(events,g,times,**w))
    return ConcatDataset(datasets)

def main(argv=None):
    p=argparse.ArgumentParser(description="Train EQ-JEPA from an earthquake catalogue")
    p.add_argument("--config", required=True); args=p.parse_args(argv)
    cfg=yaml.safe_load(Path(args.config).read_text())
    events=load_events(cfg)
    g=RegionGrid(**cfg["grid"]); times=[datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in cfg["training"]["issue_times"]]
    windows=cfg["training"].get("windows",{}); horizons=cfg["training"].get("horizons",[windows.get("horizon_days",1)])
    ds=build_multi_horizon_dataset(events,g,times,windows,horizons)
    dl=DataLoader(ds,batch_size=cfg["training"].get("batch_size",8),shuffle=True)
    model=EQJEPA(EQJEPAConfig(g.n_cells,g.n_mag,dim=cfg["model"].get("dim",128))); opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"].get("lr",3e-4))
    for epoch in range(cfg["training"].get("epochs",10)):
        for batch in dl:
            loss, metrics=model.loss(batch,batch); opt.zero_grad(); loss.backward(); opt.step(); model.update_target()
        print(f"epoch={epoch+1} {metrics}")
    out=Path(cfg.get("output_dir","results/eq_jepa")); out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"config":model.cfg},out/"model.pt")
    write_forecast(model, next(iter(dl)), g, horizons, out/"forecast.dat")

def write_forecast(model, batch, g, horizons, path):
    """Write one CSEP-style forecast file per horizon from a single L0 forward pass
    (same latent, different horizon-conditioned decode -- the horizon-coherence check)."""
    model.eval()
    with torch.no_grad():
        for h in horizons:
            horizon_tensor = torch.full_like(batch["horizon_days"], float(h))
            rates, dispersion = model.forecast_rates(batch, horizon_days=horizon_tensor)
            rates = rates.mean(0).numpy(); dispersion = dispersion.mean(0).numpy()
            rows=[]
            for c in range(g.n_cells):
                i,j=divmod(c,g.n_lon); la0=g.min_lat+i*(g.max_lat-g.min_lat)/g.n_lat; lo0=g.min_lon+j*(g.max_lon-g.min_lon)/g.n_lon
                for m in range(g.n_mag):
                    rows.append([lo0,lo0+(g.max_lon-g.min_lon)/g.n_lon,la0,la0+(g.max_lat-g.min_lat)/g.n_lat,0,700,
                                 g.magnitude_edges[m],g.magnitude_edges[m+1],float(rates[c,m]),float(dispersion[c,m])])
            hpath = path.with_name(f"{path.stem}_h{h}d{path.suffix}")
            hpath.write_text("# lon_min lon_max lat_min lat_max depth_min depth_max mag_min mag_max rate nb_dispersion\n"
                              +"\n".join(" ".join(map(str,r)) for r in rows)+"\n")
    model.train()
if __name__ == "__main__": main()
