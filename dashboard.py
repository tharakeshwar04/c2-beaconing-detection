!pip -q install gradio pandas numpy scikit-learn joblib

import gradio as gr
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime


# =========================
# 1. Convert input CSV schema
# =========================

def to_standard_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Timestamp
    if "timestamp" not in out.columns:
        if "StartTime" in out.columns:
            out["timestamp"] = out["StartTime"]
        else:
            raise ValueError("Missing timestamp column. Need `timestamp` or `StartTime`.")

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # Source IP
    if "src_ip" not in out.columns:
        if "SrcAddr" in out.columns:
            out["src_ip"] = out["SrcAddr"].astype(str)
        else:
            raise ValueError("Missing source IP column. Need `src_ip` or `SrcAddr`.")

    # Destination IP
    if "dst_ip" not in out.columns:
        if "DstAddr" in out.columns:
            out["dst_ip"] = out["DstAddr"].astype(str)
        else:
            raise ValueError("Missing destination IP column. Need `dst_ip` or `DstAddr`.")

    # Destination Port
    if "dst_port" not in out.columns:
        if "Dport" in out.columns:
            out["dst_port"] = pd.to_numeric(out["Dport"], errors="coerce")
        else:
            raise ValueError("Missing destination port column. Need `dst_port` or `Dport`.")

    out["dst_port"] = pd.to_numeric(out["dst_port"], errors="coerce")

    # Protocol
    if "protocol" not in out.columns:
        if "Proto" in out.columns:
            out["protocol"] = out["Proto"].astype(str)
        else:
            out["protocol"] = "tcp"

    out["protocol"] = out["protocol"].astype(str).str.lower()

    # Bytes
    if "bytes" not in out.columns:
        if "TotBytes" in out.columns:
            out["bytes"] = pd.to_numeric(out["TotBytes"], errors="coerce")
        else:
            out["bytes"] = 0

    # Packets
    if "packets" not in out.columns:
        if "TotPkts" in out.columns:
            out["packets"] = pd.to_numeric(out["TotPkts"], errors="coerce")
        else:
            out["packets"] = 0

    out["bytes"] = pd.to_numeric(out["bytes"], errors="coerce").fillna(0)
    out["packets"] = pd.to_numeric(out["packets"], errors="coerce").fillna(0)

    out = out.dropna(subset=["timestamp", "src_ip", "dst_ip", "dst_port", "protocol"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    return out[[
        "timestamp",
        "src_ip",
        "dst_ip",
        "dst_port",
        "protocol",
        "bytes",
        "packets"
    ]]


# =========================
# 2. Build pair-level features
# =========================

PAIR_COLS = ["src_ip", "dst_ip", "dst_port", "protocol"]

def build_pair_features(flows: pd.DataFrame, min_flows_per_pair: int = 10) -> pd.DataFrame:
    rows = []

    for key, g in flows.groupby(PAIR_COLS):
        g = g.sort_values("timestamp")

        if len(g) < min_flows_per_pair:
            continue

        t = g["timestamp"].values.astype("datetime64[s]").astype("int64")
        iat = np.diff(t)
        iat = iat[iat > 0]

        if len(iat) < 5:
            continue

        mean_iat = float(np.mean(iat))
        std_iat = float(np.std(iat))
        cv_iat = float(std_iat / mean_iat) if mean_iat > 0 else np.nan
        median_iat = float(np.median(iat))

        near_median_frac = float(
            np.mean(np.abs(iat - median_iat) <= max(1.0, 0.10 * median_iat))
        )

        duration_sec = float(
            (g["timestamp"].max() - g["timestamp"].min()).total_seconds()
        )

        flow_count = int(len(g))

        bytes_mean = float(g["bytes"].mean())
        bytes_std = float(g["bytes"].std())
        bytes_cv = float(bytes_std / bytes_mean) if bytes_mean > 0 else np.nan

        pkts_mean = float(g["packets"].mean())
        pkts_std = float(g["packets"].std())
        pkts_cv = float(pkts_std / pkts_mean) if pkts_mean > 0 else np.nan

        src_ip, dst_ip, dst_port, proto = key

        rows.append({
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "dst_port": int(dst_port),
            "protocol": proto,
            "flow_count": flow_count,
            "duration_sec": duration_sec,
            "mean_iat": mean_iat,
            "std_iat": std_iat,
            "cv_iat": cv_iat,
            "median_iat": median_iat,
            "near_median_frac": near_median_frac,
            "bytes_mean": bytes_mean,
            "bytes_cv": bytes_cv,
            "pkts_mean": pkts_mean,
            "pkts_cv": pkts_cv,
        })

    return pd.DataFrame(rows)


# =========================
# 3. Reason codes and severity
# =========================

def build_reasons(row):
    reasons = []

    if row["cv_iat"] < 0.25:
        reasons.append("Very regular timing")
    elif row["cv_iat"] < 0.40:
        reasons.append("Regular timing")

    if row["near_median_frac"] > 0.70:
        reasons.append("Intervals repeat very consistently")
    elif row["near_median_frac"] > 0.55:
        reasons.append("Intervals are fairly consistent")

    if row["flow_count"] >= 50:
        reasons.append("Many repeated connections")

    if 5 <= row["median_iat"] <= 300:
        reasons.append("Typical beacon interval range")

    if row.get("allowlisted", False):
        reasons.append("Known DNS server traffic suppressed")

    if not reasons:
        reasons.append("Flagged by model-based multi-feature pattern")

    return " | ".join(reasons)


def severity_from_prob(p: float) -> str:
    if p >= 0.90:
        return "Critical"
    elif p >= 0.70:
        return "High"
    elif p >= 0.50:
        return "Medium"
    else:
        return "Low"


def severity_badge_html(sev: str) -> str:
    cls = str(sev).lower()
    return f'<span class="sev sev-{cls}">{sev}</span>'


# =========================
# 4. HTML output helpers
# =========================

def df_to_html(df: pd.DataFrame, title: str, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="muted">No results found.</div>
        </div>
        """

    view = df.head(max_rows).copy()

    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    if "Severity" in view.columns:
        view["Severity"] = view["Severity"].apply(severity_badge_html)

    html = view.to_html(index=False, escape=False)

    return f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="table-wrap">{html}</div>
        <div class="muted">Showing {min(len(view), max_rows)} of {len(df)} rows</div>
    </div>
    """


def summary_html(text: str) -> str:
    safe_text = str(text).replace("\n", "<br>")
    return f"""
    <div class="card">
        <div class="card-title">Summary</div>
        <div class="summary">{safe_text}</div>
    </div>
    """


# =========================
# 5. Final fixed light-theme CSS
# =========================

css = """
:root{
  --page-bg:#f1f5f9;
  --card-bg:#ffffff;
  --soft-bg:#f8fafc;
  --border:#cbd5e1;
  --text:#0f172a;
  --text-soft:#334155;
  --muted:#64748b;
  --accent:#2563eb;
  --accent-dark:#1d4ed8;
}

/* Whole dashboard */
body,
.gradio-container{
  background:var(--page-bg)!important;
  color:var(--text)!important;
  font-family:Inter, Arial, sans-serif!important;
}

/* Main container */
.gradio-container{
  max-width:1400px!important;
}

/* Important text */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container p,
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .label-wrap span{
  color:#0f172a!important;
}

/* Title banner */
#titlebox{
  background:linear-gradient(135deg,#dbeafe,#f5f3ff);
  border:1px solid var(--border);
  border-radius:20px;
  padding:24px;
  margin-bottom:20px;
  box-shadow:0 10px 25px rgba(15,23,42,0.08);
}

#titlebox h1{
  color:#020617!important;
  font-size:34px!important;
  font-weight:900!important;
  margin-top:12px!important;
  margin-bottom:0!important;
}

/* Top badges */
.badge{
  display:inline-block;
  background:#ffffff!important;
  color:#0f172a!important;
  border:1px solid #cbd5e1!important;
  border-radius:999px!important;
  padding:7px 14px!important;
  margin-right:8px!important;
  font-size:12px!important;
  font-weight:800!important;
}

/* Gradio blocks and boxes */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .gr-box{
  background:#ffffff!important;
  color:#0f172a!important;
  border-color:#cbd5e1!important;
  border-radius:14px!important;
}

/* Input fields */
.gradio-container input,
.gradio-container textarea,
.gradio-container select{
  background:#ffffff!important;
  color:#0f172a!important;
  border:1px solid #cbd5e1!important;
  border-radius:10px!important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder{
  color:#64748b!important;
}

/* File upload area */
.gradio-container [data-testid="file-upload"],
.gradio-container [data-testid="file-upload"] *,
.gr-file,
.gr-file *{
  background:#ffffff!important;
  color:#0f172a!important;
  border-color:#cbd5e1!important;
}

.gr-file .wrap,
.gr-file .file-preview,
.gr-file .file-dropzone,
.gr-file .upload-container{
  background:#f8fafc!important;
  color:#0f172a!important;
  border:1px dashed #94a3b8!important;
  border-radius:14px!important;
}

/* File icons */
.gr-file svg{
  color:#2563eb!important;
  stroke:#2563eb!important;
}

/* Sliders */
input[type="range"]{
  accent-color:#2563eb!important;
}

.gr-slider,
.gr-slider *{
  color:#0f172a!important;
}

/* Checkbox */
.gr-checkbox,
.gr-checkbox label,
.gr-checkbox span{
  color:#0f172a!important;
}

/* Primary button */
button.primary,
.gr-button-primary{
  background:#2563eb!important;
  color:#ffffff!important;
  font-weight:900!important;
  border:none!important;
  border-radius:12px!important;
  padding:10px 16px!important;
}

button.primary *,
.gr-button-primary *{
  color:#ffffff!important;
}

button.primary:hover,
.gr-button-primary:hover{
  background:#1d4ed8!important;
}

/* General buttons */
button{
  border-radius:10px!important;
  font-weight:700!important;
}

/* Cards */
.card{
  background:#ffffff!important;
  border:1px solid #cbd5e1!important;
  border-radius:18px!important;
  padding:18px!important;
  margin-top:12px!important;
  box-shadow:0 8px 20px rgba(15,23,42,0.06)!important;
}

.card-title{
  color:#020617!important;
  font-size:16px!important;
  font-weight:900!important;
  margin-bottom:12px!important;
}

.summary{
  color:#0f172a!important;
  font-size:14px!important;
  line-height:1.7!important;
}

.muted{
  color:#64748b!important;
  font-size:12px!important;
  margin-top:10px!important;
}

/* Tabs */
.gradio-container .tabs,
.gradio-container .tab-nav{
  background:#ffffff!important;
  color:#0f172a!important;
  border-color:#cbd5e1!important;
}

.gradio-container button[role="tab"]{
  background:#ffffff!important;
  color:#475569!important;
  font-weight:800!important;
  border-radius:10px 10px 0 0!important;
}

.gradio-container button[role="tab"][aria-selected="true"]{
  background:#eff6ff!important;
  color:#0f172a!important;
  border-bottom:3px solid #2563eb!important;
}

/* Tables */
.table-wrap{
  overflow-x:auto!important;
}

table{
  width:100%!important;
  border-collapse:collapse!important;
  background:#ffffff!important;
  color:#0f172a!important;
  border:1px solid #cbd5e1!important;
  border-radius:14px!important;
  overflow:hidden!important;
}

thead th{
  background:#dbeafe!important;
  color:#020617!important;
  font-weight:900!important;
  padding:12px!important;
  border-bottom:1px solid #cbd5e1!important;
  white-space:nowrap!important;
}

tbody td{
  color:#0f172a!important;
  padding:11px 12px!important;
  border-bottom:1px solid #e2e8f0!important;
  vertical-align:top!important;
}

tbody tr:nth-child(odd) td{
  background:#f8fafc!important;
}

tbody tr:nth-child(even) td{
  background:#ffffff!important;
}

tbody tr:hover td{
  background:#e0f2fe!important;
}

/* Download output */
.gradio-container [data-testid="file-download"],
.gradio-container [data-testid="file-download"] *,
.gradio-container .download,
.gradio-container .download *{
  background:#ffffff!important;
  color:#0f172a!important;
  border-color:#cbd5e1!important;
}

/* Severity badges */
.sev{
  display:inline-block!important;
  padding:5px 11px!important;
  border-radius:999px!important;
  font-weight:900!important;
  font-size:12px!important;
  border:1px solid transparent!important;
  white-space:nowrap!important;
}

.sev-critical{
  background:#fee2e2!important;
  color:#991b1b!important;
  border-color:#fecaca!important;
}

.sev-high{
  background:#ffedd5!important;
  color:#9a3412!important;
  border-color:#fed7aa!important;
}

.sev-medium{
  background:#dbeafe!important;
  color:#1e40af!important;
  border-color:#bfdbfe!important;
}

.sev-low{
  background:#dcfce7!important;
  color:#166534!important;
  border-color:#bbf7d0!important;
}
"""


# =========================
# 6. Detection function
# =========================

def run_detection(flow_file, model_bundle_file, threshold, min_flows, dns_allowlist, dns_ip):
    try:
        if flow_file is None:
            return (
                summary_html("Please upload a flow CSV file to begin detection."),
                "",
                "",
                None
            )

        if model_bundle_file is None:
            return (
                summary_html("Please upload your trained model bundle file: c2_rf_model_bundle.joblib"),
                "",
                "",
                None
            )

        # Load model bundle
        bundle = joblib.load(model_bundle_file.name)

        if not isinstance(bundle, dict):
            return (
                summary_html("Model bundle error: uploaded file is not a valid dictionary bundle."),
                "",
                "",
                None
            )

        if "model" not in bundle:
            return (
                summary_html("Model bundle error: missing key `model`. Your joblib file must contain bundle['model']."),
                "",
                "",
                None
            )

        if "feature_cols" not in bundle:
            return (
                summary_html("Model bundle error: missing key `feature_cols`. Your joblib file must contain bundle['feature_cols']."),
                "",
                "",
                None
            )

        model = bundle["model"]
        feature_cols = bundle["feature_cols"]

        # Read uploaded traffic file
        df = pd.read_csv(flow_file.name)

        # Standardize schema
        flows = to_standard_schema(df)

        # Build pair-level features
        feats = build_pair_features(
            flows,
            min_flows_per_pair=int(min_flows)
        )

        if feats.empty:
            return (
                summary_html(
                    "No eligible communication pairs were found. "
                    "Try lowering MIN_FLOWS_PER_PAIR or upload a CSV with more repeated traffic."
                ),
                "",
                "",
                None
            )

        # DNS allowlist
        feats["allowlisted"] = False

        if dns_allowlist:
            feats["allowlisted"] = (
                (feats["dst_ip"].astype(str) == str(dns_ip)) &
                (feats["dst_port"].astype(int) == 53)
            )

        # Prepare model input
        X = (
            feats
            .reindex(columns=feature_cols, fill_value=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        # Predict probability
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            preds = model.predict(X)
            probs = np.array(preds, dtype=float)

        feats["rf_prob"] = probs
        feats["pred_raw"] = (feats["rf_prob"] >= float(threshold)).astype(int)

        # Suppress allowlisted DNS
        feats["final_pred"] = np.where(
            feats["allowlisted"],
            0,
            feats["pred_raw"]
        )

        feats["Severity"] = feats["rf_prob"].apply(severity_from_prob)
        feats["Reasons"] = feats.apply(build_reasons, axis=1)

        cols_out = [
            "Severity",
            "src_ip",
            "dst_ip",
            "dst_port",
            "protocol",
            "rf_prob",
            "pred_raw",
            "allowlisted",
            "final_pred",
            "flow_count",
            "median_iat",
            "cv_iat",
            "near_median_frac",
            "Reasons"
        ]

        flagged = feats[feats["final_pred"] == 1].sort_values(
            "rf_prob",
            ascending=False
        )

        overall = feats.sort_values(
            "rf_prob",
            ascending=False
        )

        # Create output CSV
        os.makedirs("outputs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"outputs/flagged_pairs_{ts}.csv"

        flagged[cols_out].to_csv(out_path, index=False)

        # Severity counts
        severity_counts = flagged["Severity"].value_counts().to_dict()

        critical_count = severity_counts.get("Critical", 0)
        high_count = severity_counts.get("High", 0)
        medium_count = severity_counts.get("Medium", 0)
        low_count = severity_counts.get("Low", 0)

        summary = (
            f"Input rows loaded: {len(df)}\n"
            f"Standardized flow rows: {len(flows)}\n"
            f"Communication pairs extracted: {len(feats)}\n"
            f"Flagged beaconing pairs: {len(flagged)}\n"
            f"Detection threshold: {threshold}\n"
            f"Minimum flows per pair: {int(min_flows)}\n"
            f"Allowlisted DNS pairs suppressed: {int(feats['allowlisted'].sum())}\n\n"
            f"Severity breakdown:\n"
            f"Critical: {critical_count}\n"
            f"High: {high_count}\n"
            f"Medium: {medium_count}\n"
            f"Low: {low_count}\n\n"
            f"Flagged results CSV is ready to download."
        )

        return (
            summary_html(summary),
            df_to_html(flagged[cols_out], "Top Flagged Beaconing Pairs", 50),
            df_to_html(overall[cols_out], "Top Pairs Overall by RF Probability", 50),
            out_path
        )

    except Exception as e:
        error_msg = (
            f"Error: {type(e).__name__}: {e}\n\n"
            f"Check that your CSV has the required columns and that your model bundle contains "
            f"`model` and `feature_cols`."
        )

        return (
            summary_html(error_msg),
            "",
            "",
            None
        )


# =========================
# 7. Gradio Dashboard Layout
# =========================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate"
    ),
    css=css,
    title="C2 Beaconing Dashboard"
) as demo:

    gr.HTML("""
    <div id="titlebox">
      <div>
        <span class="badge">Pair-level Detection</span>
        <span class="badge">Reason Codes</span>
        <span class="badge">Severity Scoring</span>
        <span class="badge">CSV Export</span>
      </div>
      <h1>C2 Beaconing Dashboard</h1>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=330):
            gr.Markdown("### Upload Files")

            flow_file = gr.File(
                label="Upload Flow CSV",
                file_types=[".csv"]
            )

            model_bundle_file = gr.File(
                label="Upload Model Bundle",
                file_types=[".joblib", ".pkl"]
            )

            gr.Markdown("### Detection Settings")

            threshold = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.01,
                label="Detection Threshold"
            )

            min_flows = gr.Slider(
                minimum=3,
                maximum=50,
                value=10,
                step=1,
                label="Minimum Flows Per Pair"
            )

            dns_allowlist = gr.Checkbox(
                value=True,
                label="Enable DNS Allowlist"
            )

            dns_ip = gr.Textbox(
                value="147.32.80.9",
                label="DNS Server IP"
            )

            run_btn = gr.Button(
                "Run Detection",
                variant="primary"
            )

            download_out = gr.File(
                label="Download Flagged Results CSV"
            )

        with gr.Column(scale=2):
            summary_out = gr.HTML()

            with gr.Tabs():
                with gr.Tab("Top Flagged Pairs"):
                    flagged_out = gr.HTML()

                with gr.Tab("Top Pairs Overall"):
                    overall_out = gr.HTML()

    run_btn.click(
        fn=run_detection,
        inputs=[
            flow_file,
            model_bundle_file,
            threshold,
            min_flows,
            dns_allowlist,
            dns_ip
        ],
        outputs=[
            summary_out,
            flagged_out,
            overall_out,
            download_out
        ]
    )


demo.launch(share=True)
