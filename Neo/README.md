# Neo – Warehouse Slotting Optimizer

Neo is a lightweight tool for real warehouses to upload their warehouse layout, current SKU locations, and sales/order history, then get data-driven slotting recommendations using Affinity Slotting, ABC Slotting, or other strategies from Ant‑Farm.

## Features
- CSV upload for layout, SKU locations, and sales/orders
- Policy engine: Affinity, ABC, RoundRobin, Random (reuses Ant‑Farm policies)
- Suggested re-slot plan with distance/time rationale
- Simple visualization of before/after placement

## Quick start
1. Place your CSVs in `Neo/sample_data/` or upload via the UI.
2. Run the app:
   - As a module: `python -m Neo.neo_app.app`
   - Or directly: `python Neo/neo_app/app.py`
3. Open the URL printed in the terminal.

## Folder structure
- `neo_app/app.py` – Dash app entry point
- `neo_app/data_io.py` – CSV schemas and parsing
- `neo_app/policies.py` – Thin adapters that import from Ant‑Farm’s `slotting.py`
- `neo_app/visuals.py` – Basic figures (optional)
- `sample_data/` – Example CSV templates

## CSV templates
- `layout.csv`: aisle, side_id, section, tote_id
- `sku_locations.csv`: sku_id, tote_id
- `sales.csv`: order_id, sku_id, quantity, order_timestamp

## Notes
- Neo imports policies from the project root; no duplication needed.
- If imports fail when running directly, use module mode or add project root to `sys.path` in `app.py`.
