"""CLI training entry point: trains EQ-JEPA and writes CSEP-style rates."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
from cvar_psha.eq_jepa.catalog import CatalogQuery, get_catalog_client
from cvar_psha.eq_jepa.data import CatalogWindowDataset, RegionGrid
from cvar_psha.eq_jepa.model import EQJEPA, EQJEPAConfig

def main(argv=None):
    p=argparse.ArgumentParser(description="Train EQ-JEPA from an earthquake catalogue")
    p.add_argument("--config", required=True); args=p.parse_args(argv)
    cfg=yaml.safe_load(Path(args.config).read_text())
    source=cfg["catalog"]["source"]; client=get_catalog_client(source)
    if source.lower()=="kiknet": events=client.fetch_csv(cfg["catalog"]["csv"])
    else:
        q=CatalogQuery(**{**cfg["catalog"]["query"], "start_time": datetime.fromisoformat(cfg["catalog"]["query"]["start_time"]).replace(tzinfo=timezone.utc), "end_time": datetime.fromisoformat(cfg["catalog"]["query"]["end_time"]).replace(tzinfo=timezone.utc)})
        events=client.fetch(q)
    g=RegionGrid(**cfg["grid"]); times=[datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in cfg["training"]["issue_times"]]
    ds=CatalogWindowDataset(events,g,times,**cfg["training"].get("windows",{})); dl=DataLoader(ds,batch_size=cfg["training"].get("batch_size",8),shuffle=True)
    model=EQJEPA(EQJEPAConfig(g.n_cells,g.n_mag,dim=cfg["model"].get("dim",128))); opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"].get("lr",3e-4))
    for epoch in range(cfg["training"].get("epochs",10)):
        for batch in dl:
            loss, metrics=model.loss(batch,batch); opt.zero_grad(); loss.backward(); opt.step(); model.update_target()
        print(f"epoch={epoch+1} {metrics}")
    out=Path(cfg.get("output_dir","results/eq_jepa")); out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"config":model.cfg},out/"model.pt")
    with torch.no_grad(): rates=model.forecast_rates(next(iter(dl))).mean(0).numpy()
    rows=[]
    for c in range(g.n_cells):
        i,j=divmod(c,g.n_lon); la0=g.min_lat+i*(g.max_lat-g.min_lat)/g.n_lat; lo0=g.min_lon+j*(g.max_lon-g.min_lon)/g.n_lon
        for m in range(g.n_mag): rows.append([lo0,lo0+(g.max_lon-g.min_lon)/g.n_lon,la0,la0+(g.max_lat-g.min_lat)/g.n_lat,0,700,g.magnitude_edges[m],g.magnitude_edges[m+1],float(rates[c,m])])
    (out/"forecast.dat").write_text("# lon_min lon_max lat_min lat_max depth_min depth_max mag_min mag_max rate\n"+"\n".join(" ".join(map(str,r)) for r in rows)+"\n")
if __name__ == "__main__": main()
