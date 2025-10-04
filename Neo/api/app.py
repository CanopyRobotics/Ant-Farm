from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import pandas as pd
import io
from .registry import list_policies
from .runner import run_slotting, build_move_plan_csv, build_mapping_csv

app = FastAPI(title="Neo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/policies")
def get_policies():
    return {"policies": list_policies()}


@app.post("/api/run")
async def api_run(
    policy: str = Form("affinity"),
    simulate_correlated: bool = Form(False),
    optimization: str = Form("max"),
    layout: UploadFile = File(...),
    sku_locations: UploadFile = File(...),
    sales: UploadFile | None = File(None),
):
    def _read_csv(f: UploadFile) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(f.file.read()))

    layout_df = _read_csv(layout)
    sku_df = _read_csv(sku_locations)
    sales_df = _read_csv(sales) if sales is not None else None

    result = run_slotting(
        layout_df=layout_df,
        sku_df=sku_df,
        sales_df=sales_df,
        policy=policy,
        simulate_correlated=simulate_correlated,
        optimization=optimization,
    )
    return JSONResponse(result)


@app.post("/api/export/move_plan")
async def api_export_move_plan(
    proposed_map_json: str = Form(...),
    current_map_json: str = Form(...),
    layout: UploadFile | None = File(None),
    sales: UploadFile | None = File(None),
    top_n: int = Form(1000),
):
    layout_df = pd.read_csv(io.BytesIO(layout.file.read())) if layout is not None else None
    sales_df = pd.read_csv(io.BytesIO(sales.file.read())) if sales is not None else None
    csv_text = build_move_plan_csv(proposed_map_json, current_map_json, sales_df, layout_df, top_n)
    return PlainTextResponse(content=csv_text, media_type="text/csv")


@app.post("/api/export/mapping")
async def api_export_mapping(
    proposed_map_json: str = Form(...),
    layout: UploadFile | None = File(None),
):
    layout_df = pd.read_csv(io.BytesIO(layout.file.read())) if layout is not None else None
    csv_text = build_mapping_csv(proposed_map_json, layout_df)
    return PlainTextResponse(content=csv_text, media_type="text/csv")
