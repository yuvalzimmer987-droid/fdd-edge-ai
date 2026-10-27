import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from detector import Detector
from signatures import FAULTS, SIGNATURES_FILE, signature_vectors, load_signatures, match
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

# Operator dashboard: replays one plant file hour by hour, as if the data were
# arriving live, and shows what the edge detector (detector.py) reports.
#
# Run (from the project folder):  streamlit run src/dashboard.py
# Needs: train_model.py and subsystems.py --hybrid run first.
# Optional: data/hybrid_tradeoff.csv and data/edge_results.csv (from
# subsystems.py --sweep and export_model.py) fill the performance tiles, and
# models/fault_signatures.json (fault_signatures.py) the possible-faults panel.

START = pd.Timestamp("2021-01-01")       # the files hold one year of 1-minute data

# Online demo (Streamlit Community Cloud): without the full data folder, read the
# small pack made by make_demo_data.py (test days, steady operation only).
DEMO_DIR = "demo_data"
DEMO = not os.path.exists(f"{DATA_DIR}/ChillerPlant.csv") and os.path.exists(f"{DEMO_DIR}/file_catalog.csv")
RESULTS_DIR = DEMO_DIR if DEMO else "data"

# --- Look: dark control room, same palette as the presentation ---
BG, PANEL, PANEL2, BORDER = "#0E1621", "#16212F", "#1C2A3C", "#26344A"
TEXT, MUTED, TEAL, PIPE = "#E8EDF3", "#8796AB", "#3CC8BA", "#2A7F77"
BLUE, ORANGE = "#3987e5", "#d95926"      # series: main model, pump detector
GRID = "rgba(135,150,171,0.15)"
# Status colors are only used for status, always with an icon and a word
STATUS = {"Normal": ("#0ca30c", "✓"), "Minor": ("#fab219", "!"),
          "Major": ("#ec835a", "▲"), "Critical": ("#d03b3b", "⚠")}
SEVERITIES = ["Minor", "Major", "Critical"]
LOG_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
DETECTORS = {"main": ("Main model", BLUE), "pumps": ("Pump detector", ORANGE)}

# The parts shown follow the two detectors: the main model watches the chiller
# and the cooling tower together (they are coupled through the condenser loop,
# and separate models detected less), the pump detector watches the pumping loop.
PARTS = {"main": "Chiller & cooling tower", "pumps": "Pumps & chilled-water loop"}
SHORT = {"main": "Chiller & tower", "pumps": "Pumps & loop"}
SENSOR_NAMES = {
    "CHL_SW_TEMP_1": "Chiller supply water temp", "CHL_RW_TEMP_1": "Chiller return water temp",
    "CHL_SWCD_TEMP_1": "Condenser supply water temp", "CHL_RWCD_TEMP_1": "Condenser return water temp",
    "CWL_SEC_SW_TEMP": "Secondary loop supply temp", "CWL_SEC_RW_TEMP": "Secondary loop return temp",
    "CT_SW_TEMP_1": "Cooling tower supply water temp", "CT_RW_TEMP_1": "Cooling tower return water temp",
    "OA_TEMP": "Outdoor air temp", "OA_TEMP_WB": "Outdoor wet-bulb temp",
    "CHL_POW_1": "Chiller power", "CT_POW_1": "Cooling tower fan power",
    "CHL_CW_FLOW_1": "Chiller chilled water flow", "CWL_SEC_DP": "Secondary loop diff. pressure",
    "CWL_SEC_DPSPT": "Diff. pressure setpoint", "CWL_SEC_PM_SPD_1": "Pump 1 speed",
    "CWL_SEC_PM_SPD_2": "Pump 2 speed", "CWL_SEC_PM_POW_1": "Pump 1 power",
    "CWL_SEC_PM_POW_2": "Pump 2 power", "CWL_SEC_CW_FLOW": "Secondary chilled water flow",
    "CWL_PRI_CW_FLOW": "Primary chilled water flow", "CWL_SEC_LOAD": "Cooling load",
}


def unit(sensor):
    """Units we are sure of: temperatures in °F, powers in kW. The units of flows,
    pressures, speeds and load are not given in the files, so they are left out."""
    if "_TEMP" in sensor:
        return "°F"
    if "_POW" in sensor:
        return "kW"
    return ""


def axis_label(sensor):
    u = unit(sensor)
    return f"{SENSOR_NAMES.get(sensor, sensor)} ({u})" if u else f"{SENSOR_NAMES.get(sensor, sensor)} ({sensor})"


def severity(ratio):
    """How far above its threshold: < 2x Minor, < 5x Major, otherwise Critical."""
    if ratio <= 1:
        return "Normal"
    return "Minor" if ratio < 2 else "Major" if ratio < 5 else "Critical"


def when(hour):
    return START + pd.Timedelta(hours=int(hour))


def fmt(hour, pattern="%d %b, %H:00"):
    return when(hour).strftime(pattern)


def chip(level, text=None):
    color, icon = STATUS[level]
    return (f"<span class='chip' style='color:{color};border-color:{color}66;background:{color}1a'>"
            f"{icon} {text or level}</span>")


# --- Data (cached) ---
@st.cache_resource
def get_detector():
    return Detector.from_saved()


@st.cache_data
def get_catalog():
    catalog = pd.read_csv(f"{RESULTS_DIR}/file_catalog.csv", dtype={"severity": str})
    names = {}
    for _, row in catalog.iterrows():
        if row["label"] == "normal":
            names[row["filename"]] = "No fault (fault-free file)"
        else:
            sev = row["severity"] if isinstance(row["severity"], str) else ""
            names[row["filename"]] = f"{row['fault_type'].replace('_', ' ')} {sev}".strip()
    return names


@st.cache_resource
def get_signatures():
    return load_signatures() if os.path.exists(SIGNATURES_FILE) else None


@st.cache_data
def get_performance(budget):
    """Detection and false alarms on test days, and the ONNX numbers, if the files exist."""
    perf = {}
    if os.path.exists(f"{RESULTS_DIR}/hybrid_tradeoff.csv"):
        t = pd.read_csv(f"{RESULTS_DIR}/hybrid_tradeoff.csv")
        row = t.loc[(t["val_budget_pct"] - budget).abs().idxmin()]
        perf["detection"] = row["detection_all_test_pct"]
        perf["false_alarms"] = row["false_alarm_test_pct"]
    if os.path.exists(f"{RESULTS_DIR}/edge_results.csv"):
        e = pd.read_csv(f"{RESULTS_DIR}/edge_results.csv", index_col=0)
        if "ONNX fp32" in e.index:
            perf["size_kb"] = e.at["ONNX fp32", "size_kb"]
            perf["latency_us"] = e.at["ONNX fp32", "latency_median_us"]
    return perf


@st.cache_data(show_spinner="Running the detector on the whole file...")
def analyse(filename, test_days_only):
    detector = get_detector()
    if DEMO:
        frame = pd.read_parquet(f"{DEMO_DIR}/{filename.replace('.csv', '.parquet')}")
        idx = frame.index.to_numpy()
    else:
        df = load_plant_file(f"{DATA_DIR}/{filename}", detector.load_columns())
        keep, _, _, _ = steady_state_mask(df)
        if test_days_only:
            keep = keep & (day_split(len(df)) == "test")
        idx = np.flatnonzero(keep)
        frame = df.iloc[idx]
    signed = detector.signed_z_scores(frame, idx)
    z = {n: v.abs() for n, v in signed.items()}
    hours = detector.hourly(frame, idx, z)
    if hours.empty:
        return frame, hours, None, None, None
    # How much each hour's pattern of sensor changes looks like each known fault
    sigs = get_signatures()
    similarity = match(signature_vectors(signed), sigs) if sigs else None
    # Every sensor's shift in units of its detector's threshold (> 1 = above threshold).
    # A few sensors are in both detectors: keep the larger value.
    per_detector = {n: z[n] / detector.thresholds[n] for n in z}
    ratio = pd.concat(per_detector.values(), axis=1)
    ratio = ratio.T.groupby(level=0).max().T
    # Per hour: each detector's score / threshold, the part furthest above its
    # threshold, its most-shifted sensor and the severity
    part_ratio = pd.DataFrame({n: hours[f"score_{n}"] / detector.thresholds[n] for n in PARTS})
    hours["part"] = part_ratio.idxmax(axis=1)
    hours["ratio"] = part_ratio.max(axis=1)
    hours["top_sensor"] = [hours.at[h, f"sensor_{p}"] for h, p in hours["part"].items()]
    hours["severity"] = hours["ratio"].map(severity)
    return frame, hours, ratio, part_ratio, similarity


def events(hours, max_gap=3):
    """Group alarm hours into events: alarms less than max_gap analysed hours apart."""
    alarm = hours[hours["alarm"]]
    out, current = [], None
    for h, row in alarm.iterrows():
        alarmed = {p for p in PARTS if row[f"alarm_{p}"]}     # every part above its threshold
        if current is not None and h - current["end"] <= max_gap:
            current["end"] = h
            current["hours"] += 1
            current["parts"] |= alarmed
            if row["ratio"] > current["ratio"]:
                current.update(ratio=row["ratio"], part=row["part"], sensor=row["top_sensor"])
        else:
            current = dict(start=h, end=h, hours=1, ratio=row["ratio"], parts=alarmed,
                           part=row["part"], sensor=row["top_sensor"])
            out.append(current)
    return out


# --- Page setup ---
st.set_page_config(page_title="Chiller Plant FDD", page_icon="❄", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp {{ background: radial-gradient(1200px 500px at 70% -10%, #13293a 0%, {BG} 60%); }}
h1, h2, h3, h4, .kpi-value, .brand {{ font-family: 'Space Grotesk', sans-serif !important; }}
.block-container {{ padding-top: 1.6rem; }}
.panel {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:14px; padding:16px 18px; height:100%; }}
.panel-title {{ font-family:'Space Grotesk',sans-serif; font-size:1.02rem; font-weight:600; color:{TEXT}; margin-bottom:10px; }}
.panel-title small {{ font-family:'JetBrains Mono',monospace; color:{MUTED}; font-weight:400; font-size:0.72rem; margin-left:8px; }}
.mono {{ font-family:'JetBrains Mono',monospace; }}
.muted {{ color:{MUTED}; }}
.chip {{ display:inline-block; border:1px solid; border-radius:999px; padding:1px 10px; font-size:0.78rem; font-weight:600; white-space:nowrap; }}
.kpi {{ display:flex; gap:14px; align-items:center; }}
.kpi-icon {{ width:46px; height:46px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; flex-shrink:0; }}
.kpi-label {{ color:{MUTED}; font-size:0.8rem; }}
.kpi-value {{ color:{TEXT}; font-size:1.55rem; font-weight:600; line-height:1.2; }}
.kpi-sub {{ color:{MUTED}; font-size:0.78rem; }}
table.log {{ width:100%; border-collapse:collapse; font-size:0.84rem; }}
table.log th {{ text-align:left; color:{MUTED}; font-weight:500; font-size:0.74rem; padding:6px 8px; border-bottom:1px solid {BORDER}; }}
table.log td {{ padding:7px 8px; border-bottom:1px solid {BORDER}55; color:{TEXT}; }}
table.log td.mono {{ white-space:nowrap; }}
table.log tr.open td {{ background:{TEAL}0d; }}
.pulse {{ display:flex; gap:3px; flex-wrap:nowrap; }}
.pulse div {{ flex:1; height:26px; border-radius:3px; }}
@keyframes flow {{ to {{ stroke-dashoffset: -24; }} }}
.pipe {{ stroke-dasharray: 8 6; animation: flow 1.4s linear infinite; }}
.pipe-back {{ stroke-dasharray: 8 6; animation: flow 1.4s linear infinite reverse; }}
@keyframes glow {{ 0%,100% {{ opacity:0.25; }} 50% {{ opacity:0.7; }} }}
.halo {{ animation: glow 1.6s ease-in-out infinite; }}
</style>""", unsafe_allow_html=True)

detector = get_detector()
catalog = get_catalog()
files = list(catalog)
default = next(i for i, f in enumerate(files) if f != "ChillerPlant.csv")

with st.sidebar:
    st.markdown(f"<div class='brand' style='font-size:1.35rem;font-weight:600'>❄ Plant FDD "
                f"<span style='color:{TEAL}'>· edge</span></div>"
                f"<div class='muted' style='font-size:0.8rem;margin-bottom:18px'>Replay of the LBNL chiller plant</div>",
                unsafe_allow_html=True)
    filename = st.selectbox("Plant file", files, index=default, format_func=catalog.get)
    test_days_only = st.toggle("Test days only", value=True, disabled=DEMO,
                               help="Days the models never saw in training or threshold setting."
                                    + (" The online demo includes test days only." if DEMO else ""))
    step = st.select_slider("Hours per step", [1, 3, 6, 24], value=1)
    delay = st.select_slider("Seconds between steps", [0.2, 0.5, 1.0, 2.0], value=0.5)
    st.markdown(f"<div class='muted' style='font-size:0.78rem;margin-top:14px'>"
                f"Severity = how far a sensor is beyond its alarm limit: "
                f"{chip('Minor', 'Minor &lt;2×')} {chip('Major', 'Major &lt;5×')} {chip('Critical', 'Critical ≥5×')}"
                f"<br><br>Alarm limits are set so that only {detector.budget:g}% of normal hours would raise an alarm.</div>",
                unsafe_allow_html=True)

frame, hours, ratio, part_ratio, similarity = analyse(filename, test_days_only)
if hours.empty:
    st.warning("No steady-state hours in this selection.")
    st.stop()
n = len(hours)

# Replay position (changed only through callbacks, before the slider is drawn)
if st.session_state.get("file") != (filename, test_days_only):
    st.session_state.update(file=(filename, test_days_only), pos=0, playing=False)


def move(delta):
    st.session_state.pos = int(np.clip(st.session_state.pos + delta, 0, n - 1))


def toggle_play():
    st.session_state.playing = not st.session_state.playing


if st.session_state.playing:
    move(step)
    if st.session_state.pos >= n - 1:
        st.session_state.playing = False

pos = st.session_state.pos
hour = hours.index[pos]
now = hours.loc[hour]
so_far = hours.loc[:hour]
plant_level = now["severity"]

# --- Header: title, replay clock and controls ---
head, clock = st.columns([3, 2])
head.markdown(f"<h1 style='margin:0;font-size:2rem'>Chiller Plant FDD</h1>"
              f"<div class='muted'>AI fault detection on the edge · replaying "
              f"<b style='color:{TEXT}'>{catalog[filename]}</b></div>", unsafe_allow_html=True)
clock.markdown(f"<div style='text-align:right'><span class='mono muted' style='font-size:0.78rem'>REPLAY TIME</span><br>"
               f"<span class='brand' style='font-size:1.6rem;color:{TEXT}'>{fmt(hour, '%a %d %b · %H:00')}</span><br>"
               f"<span class='mono muted' style='font-size:0.78rem'>hour {pos + 1} of {n} analysed</span></div>",
               unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1, 1, 1, 7])
c1.button("◀ Back", on_click=move, args=(-step,), width="stretch")
c2.button("❚❚ Pause" if st.session_state.playing else "▶ Play", on_click=toggle_play,
          width="stretch", type="primary")
c3.button("Next ▶", on_click=move, args=(step,), width="stretch")
c4.slider("Hour", 0, n - 1, key="pos", label_visibility="collapsed")

# --- What's happening now: one plain sentence ---
def hour_reading(h, sensor, part):
    """Mean measured and expected value of a sensor over one analysed hour."""
    rows = frame[(frame.index >= h * 60) & (frame.index < (h + 1) * 60)]
    return rows[sensor].mean(), detector.expected(rows, part)[sensor].mean()


def difference_words(sensor, measured, expected):
    """'4.2 °F colder', '12% higher' - the gap between what a sensor reads and what it should."""
    gap = measured - expected
    if "_TEMP" in sensor:
        return f"{abs(gap):.1f} °F {'warmer' if gap > 0 else 'colder'}"
    if unit(sensor):
        return f"{abs(gap):.1f} {unit(sensor)} {'higher' if gap > 0 else 'lower'}"
    return f"{abs(gap / expected) * 100:.0f}% {'higher' if gap > 0 else 'lower'}"


sigs = get_signatures()
best = similarity.loc[hour].idxmax() if similarity is not None and now["alarm"] else None
current_event = next((e for e in events(so_far)[::-1] if hour - e["end"] <= 3), None) if now["alarm"] else None
measured, expected_now = hour_reading(hour, now["top_sensor"], now["part"])
color, icon = STATUS[plant_level]
if now["alarm"]:
    duration = current_event["hours"] if current_event else 1
    line1 = (f"<b>{PARTS[now['part']]}:</b> the {SENSOR_NAMES.get(now['top_sensor'], now['top_sensor']).lower()} "
             f"reads <b>{difference_words(now['top_sensor'], measured, expected_now)}</b> than it should for the "
             f"current weather and load, at {now['ratio']:.1f}× its alarm limit. "
             f"Alarm for {duration} analysed hour{'s' if duration > 1 else ''} so far.")
    line2 = (f"Most similar known fault: <b>{FAULTS[best]['name']}</b> - see the list of possible faults below."
             if best else "")
else:
    line1 = "<b>All monitored parts are running as expected</b> for the current weather and load."
    line2 = "No sensor is beyond its alarm limit this hour."
st.markdown(f"<div class='panel' style='border-left:6px solid {color};margin-bottom:14px'>"
            f"<div class='kpi-label'>WHAT IS HAPPENING NOW</div>"
            f"<div style='font-size:1.15rem;line-height:1.5;color:{TEXT};margin-top:4px'>{icon} {line1}</div>"
            f"<div class='muted' style='margin-top:4px'>{line2}</div></div>", unsafe_allow_html=True)

# --- KPI tiles ---
perf = get_performance(detector.budget)
alarm_parts = [p for p in part_ratio.columns if part_ratio.at[hour, p] > 1]


def kpi(column, icon, color, label, value, sub):
    column.markdown(
        f"<div class='panel kpi'><div class='kpi-icon' style='background:{color}22;color:{color}'>{icon}</div>"
        f"<div><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div></div></div>", unsafe_allow_html=True)


k1, k2, k3, k4 = st.columns(4)
color, icon = STATUS[plant_level]
kpi(k1, icon, color, "Plant status", "Normal" if plant_level == "Normal" else f"{plant_level} alarm",
    "all parts below threshold" if not alarm_parts else
    f"{len(alarm_parts)} part{'s' if len(alarm_parts) > 1 else ''} in alarm")
kpi(k2, "◷", TEAL, "Hours with an alarm so far", f"{int(so_far['alarm'].sum())} / {len(so_far)}",
    f"{so_far['alarm'].mean() * 100:.1f}% of the hours replayed")
kpi(k3, "◎", TEAL, "How reliable is the detector?",
    f"{perf['detection']:.0f}% found" if "detection" in perf else "-",
    f"of fault hours on unseen days · {perf['false_alarms']:.1f}% false alarms" if "detection" in perf
    else "run subsystems.py --hybrid --sweep")
kpi(k4, "⚡", TEAL, "Runs on a small edge device",
    f"{perf['latency_us']:.0f} µs / min" if "latency_us" in perf else "1-hour windows",
    f"of data · {perf['size_kb']:.0f} KB for both models" if "size_kb" in perf
    else "run export_model.py for edge numbers")

st.write("")

# --- Plant overview: the cooling chain with a status badge per monitored part ---
def node(x, label, part, shape, w=140):
    """One piece of equipment: an outline shape, a label and (if monitored) a status badge."""
    level = severity(part_ratio.at[hour, part]) if part in part_ratio.columns else None
    color = STATUS[level][0] if level else MUTED
    stroke = color if level and level != "Normal" else BORDER
    g = [f"<g transform='translate({x},0)'>"]
    if level and level != "Normal":
        g.append(f"<rect class='halo' x='{-w / 2 - 4}' y='18' width='{w + 8}' height='118' rx='18' fill='{color}' opacity='0.35'/>")
    g.append(f"<rect x='{-w / 2}' y='22' width='{w}' height='110' rx='16' fill='{PANEL2}' stroke='{stroke}' stroke-width='2'/>")
    g.append(shape)
    if level:
        icon = STATUS[level][1]
        g.append(f"<circle cx='{w / 2 - 14}' cy='34' r='15' fill='{color}'/>"
                 f"<text x='{w / 2 - 14}' y='40' text-anchor='middle' font-size='16' font-weight='700' fill='{BG}'>{icon}</text>")
        word = level if level != "Normal" else "Healthy"
        g.append(f"<text x='0' y='174' text-anchor='middle' font-size='14' fill='{color}' "
                 f"font-family='Inter,sans-serif'>{word}</text>")
    else:
        g.append(f"<text x='0' y='174' text-anchor='middle' font-size='13' fill='{MUTED}' "
                 f"font-family='Inter,sans-serif'>not monitored</text>")
    g.append(f"<text x='0' y='156' text-anchor='middle' font-size='15' font-weight='600' fill='{TEXT}' "
             f"font-family='Space Grotesk,sans-serif'>{label}</text></g>")
    return "".join(g)


servers = (f"<g stroke='{MUTED}' stroke-width='2' fill='none'>" +
           "".join(f"<rect x='{-36 + i * 26}' y='46' width='20' height='62' rx='3'/>"
                   f"<line x1='{-32 + i * 26}' y1='58' x2='{-20 + i * 26}' y2='58'/>"
                   f"<line x1='{-32 + i * 26}' y1='68' x2='{-20 + i * 26}' y2='68'/>" for i in range(3)) + "</g>")
pump = (f"<g stroke='{TEXT}' stroke-width='2' fill='none'><circle cx='-14' cy='78' r='22'/>"
        f"<path d='M-24 66 L-2 78 L-24 90 Z'/><circle cx='30' cy='78' r='14'/><path d='M23 70 L37 78 L23 86 Z'/></g>")
chiller = (f"<g stroke='{TEXT}' stroke-width='2' fill='none'><rect x='-44' y='52' width='88' height='52' rx='6'/>"
           f"<line x1='-30' y1='64' x2='30' y2='64'/><line x1='-30' y1='76' x2='30' y2='76'/>"
           f"<line x1='-30' y1='88' x2='30' y2='88'/><circle cx='-30' cy='45' r='6'/><circle cx='30' cy='45' r='6'/></g>")
tower = (f"<g stroke='{TEXT}' stroke-width='2' fill='none'><path d='M-36 108 L-26 52 L26 52 L36 108 Z'/>"
         f"<ellipse cx='0' cy='52' rx='26' ry='7'/><path d='M-10 40 q5 -8 10 0 q5 8 10 0'/></g>")
air = (f"<g stroke='{MUTED}' stroke-width='2' fill='none'><path d='M-40 62 h56 q14 0 14 -12 q0 -12 -14 -10'/>"
       f"<path d='M-40 78 h70 q14 0 14 12 q0 12 -14 10'/><path d='M-40 94 h40'/></g>")

# Chiller and cooling tower share one box (one detector), joined by the condenser loop
plant_box = (f"<g transform='translate(-95,0)'>{chiller}</g><g transform='translate(95,0)'>{tower}</g>"
             f"<line class='pipe' x1='-45' y1='62' x2='55' y2='62' stroke='#c98500' stroke-width='3' opacity='0.8'/>"
             f"<line class='pipe' x1='-45' y1='96' x2='55' y2='96' stroke='#c98500' stroke-width='3' opacity='0.45'/>")
xs = {"servers": 110, "loop": 340, "main": 685, "air": 1030}
widths = {"servers": 140, "loop": 140, "main": 370, "air": 140}
pipes = []
for a, b, cls, color in [("servers", "loop", "pipe-back", "#3987e5"), ("loop", "main", "pipe-back", "#3987e5"),
                         ("main", "air", "pipe", MUTED)]:
    x1, x2 = xs[a] + widths[a] / 2, xs[b] - widths[b] / 2
    pipes.append(f"<line class='{cls}' x1='{x1}' y1='62' x2='{x2}' y2='62' stroke='{color}' stroke-width='3' opacity='0.8'/>"
                 f"<line class='{cls}' x1='{x1}' y1='96' x2='{x2}' y2='96' stroke='{color}' stroke-width='3' opacity='0.45'/>")
plant_svg = (
    "<svg viewBox='0 0 1140 190' width='100%' role='img' "
    "aria-label='Cooling chain from the server rooms through the pumps, chiller and cooling tower to outdoor air, with the status of each monitored part'>"
    + "".join(pipes)
    + node(xs["servers"], "Data hall load", None, servers)
    + node(xs["loop"], "Pumps & loop", "pumps", pump)
    + node(xs["main"], PARTS["main"], "main", plant_box, w=widths["main"])
    + node(xs["air"], "Outdoor air", None, air)
    + "</svg>")
st.markdown(f"<div class='panel'><div class='panel-title'>Plant overview<small>chilled water ← · → condenser water · one status per detector</small></div>"
            f"{plant_svg}</div>", unsafe_allow_html=True)

# --- Possible faults vs. what the plant is doing now ---
if sigs is None:
    st.info("Possible faults: run python src/fault_signatures.py to learn the fault signatures.")
else:
    order = list(FAULTS)
    if now["alarm"]:
        order = list(similarity.loc[hour].sort_values(ascending=False).index)
    rows = []
    for k in order:
        f = FAULTS[k]
        if now["alarm"]:
            m = max(0.0, float(similarity.at[hour, k]))
            top_one = k == order[0]
            bar_color = TEAL if top_one else "#3A4A61"
            cell = (f"<div style='display:flex;align-items:center;gap:8px'><div style='flex:1;background:{PANEL2};"
                    f"border-radius:4px;height:10px'><div style='width:{m * 100:.0f}%;background:{bar_color};"
                    f"height:10px;border-radius:4px'></div></div><span class='mono' style='width:38px'>{m * 100:.0f}%</span></div>"
                    + (f"<span class='chip' style='color:{TEAL};border-color:{TEAL}66;margin-top:4px'>Most similar</span>"
                       if top_one else ""))
        else:
            cell = "<span class='muted'>-</span>"
        rows.append(f"<tr><td>{f['where']}</td><td><b>{f['name']}</b><br><span class='muted' style='font-size:0.78rem'>"
                    f"{f['what']}</span></td><td class='muted' style='font-size:0.8rem'>{f['looks']}</td>"
                    f"<td style='width:190px'>{cell}</td></tr>")
    acc = sigs.get("accuracy", {})
    note = (f"Each fault moves the sensors in its own pattern. This hour's pattern is compared with the typical "
            f"pattern of each fault, learned from labelled examples. On days the models never saw, the most similar "
            f"fault was the right one in {acc.get('top1_pct', 0):.0f}% of alarm hours, and among the top three in "
            f"{acc.get('top3_pct', 0):.0f}%. A fault that is not in this list cannot be recognised.")
    status_words = ("Ranked by how closely this hour matches each fault" if now["alarm"] else
                    "No alarm this hour, so the plant matches normal operation; matching runs when an alarm is raised")
    header = ("<tr><th>Where</th><th>Possible fault</th><th>What it would look like</th>"
              "<th>Match with this hour</th></tr>")
    shown = rows[:5] if now["alarm"] else rows        # in an alarm: the five closest first
    st.markdown(f"<div class='panel' style='margin-top:14px'><div class='panel-title'>Possible faults vs. what the "
                f"plant is doing now<small>{status_words}</small></div>"
                f"<table class='log'>{header}{''.join(shown)}</table>"
                f"<div class='muted' style='font-size:0.75rem;margin-top:10px'>{note}</div></div>",
                unsafe_allow_html=True)
    if len(rows) > len(shown):
        with st.expander(f"{len(rows) - len(shown)} less similar faults"):
            st.markdown(f"<table class='log'>{header}{''.join(rows[len(shown):])}</table>", unsafe_allow_html=True)

# --- Plant pulse: the last 72 analysed hours, one cell per hour ---
recent = hours.iloc[max(0, pos - 71):pos + 1]
cells = []
for h, row in recent.iterrows():
    level = row["severity"]
    fill = "#1d4f4b" if level == "Normal" else STATUS[level][0]
    outline = f"outline:2px solid {TEXT};outline-offset:1px;" if h == hour else ""
    tip = f"{fmt(h)} - {level}" + ("" if level == "Normal" else f" - {SENSOR_NAMES.get(row['top_sensor'], row['top_sensor'])}")
    cells.append(f"<div title='{tip}' style='background:{fill};{outline}'></div>")
cells += ["<div style='background:transparent'></div>"] * (72 - len(recent))
st.markdown(f"<div class='panel' style='margin-top:14px'><div class='panel-title'>Plant pulse"
            f"<small>last {len(recent)} analysed hours · hover a cell</small></div>"
            f"<div class='pulse'>{''.join(cells)}</div>"
            f"<div class='muted mono' style='display:flex;justify-content:space-between;font-size:0.72rem;margin-top:6px'>"
            f"<span>{fmt(recent.index[0])}</span><span>now</span></div></div>", unsafe_allow_html=True)
st.write("")

# --- Fault events and trends ---
left, right = st.columns([3, 2])
with left:
    ev = events(so_far)[::-1][:8]
    rows = []
    for e in ev:
        is_open = e["end"] == hour or (now["alarm"] and e is ev[0] and hour - e["end"] <= 3)
        status = (f"<span class='chip' style='color:{TEAL};border-color:{TEAL}66'>● Open</span>" if is_open
                  else "<span class='muted'>Closed</span>")
        rows.append(f"<tr class='{'open' if is_open else ''}'><td class='mono'>{fmt(e['start'])}</td>"
                    f"<td>{chip(severity(e['ratio']))}</td><td style='white-space:nowrap'>{'<br>'.join(SHORT[p] for p in PARTS if p in e['parts'])}</td>"
                    f"<td>{SENSOR_NAMES.get(e['sensor'], e['sensor'])}<br><span class='muted mono' style='font-size:0.72rem'>{e['sensor']}</span></td>"
                    f"<td class='mono'>{e['hours']} h</td><td class='mono'>{e['ratio']:.1f}×</td><td>{status}</td></tr>")
    body = "".join(rows) if rows else f"<tr><td colspan='7' class='muted'>No alarms so far.</td></tr>"
    st.markdown(f"<div class='panel'><div class='panel-title'>Fault events<small>alarm hours grouped into events</small></div>"
                f"<table class='log'><tr><th>Started</th><th>Severity</th><th>Parts in alarm</th><th>Where it shows (peak hour)</th>"
                f"<th>Duration</th><th>Peak (× limit)</th><th>Status</th></tr>{body}</table></div>", unsafe_allow_html=True)


def base_layout(fig, height, **kw):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="Inter, sans-serif", size=12),
                      legend=dict(orientation="h", y=1.15, x=0, font=dict(size=11)), **kw)
    return fig


with right:
    with st.container(border=True):
        st.markdown("<div class='panel-title'>Fault trend<small>alarm hours per day, by severity</small></div>",
                    unsafe_allow_html=True)
        alarms = so_far[so_far["alarm"]]
        day = so_far.index // 24
        days = pd.Index(sorted(set(day)))[-10:]
        fig = go.Figure()
        for level in SEVERITIES:
            counts = alarms[alarms["severity"] == level].groupby(alarms[alarms["severity"] == level].index // 24).size()
            fig.add_trace(go.Bar(x=[fmt(d * 24, "%d %b") for d in days], y=counts.reindex(days, fill_value=0),
                                 name=level, marker=dict(color=STATUS[level][0], cornerradius=3,
                                                         line=dict(color=PANEL, width=1)),
                                 hovertemplate="%{x}: %{y} h " + level + "<extra></extra>"))
        base_layout(fig, 210, barmode="stack", bargap=0.35,
                    yaxis=dict(gridcolor=GRID, title="hours", dtick=6), xaxis=dict(type="category"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown("<div class='panel-title'>Alarm hours by part<small>an hour counts for every part "
                    "above its threshold</small></div>", unsafe_allow_html=True)
        by_part = pd.Series({p: int(alarms[f"alarm_{p}"].sum()) for p in PARTS})
        fig = go.Figure(go.Bar(x=by_part.to_numpy(), y=[PARTS[p] for p in by_part.index], orientation="h",
                               marker=dict(color=TEAL, cornerradius=3), text=by_part.to_numpy(),
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x} alarm hours<extra></extra>"))
        base_layout(fig, 140, xaxis=dict(gridcolor=GRID, visible=False),
                    yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# --- Fault details for the current hour ---
st.write("")
suspect = now["top_sensor"]
owner = now["part"]
title = (f"{chip(plant_level)} &nbsp;{SENSOR_NAMES.get(suspect, suspect)} is off what it should read - {PARTS[now['part']]}"
         if now["alarm"] else f"{chip('Normal')} &nbsp;No alarm this hour - the sensor closest to its limit: "
                              f"{SENSOR_NAMES.get(suspect, suspect)}")
st.markdown(f"<div class='panel-title' style='font-size:1.15rem'>This hour in detail</div>"
            f"<div style='margin-bottom:6px'>{title}</div>", unsafe_allow_html=True)

detail, isolation = st.columns([3, 1])
with detail:
    st.markdown("<div class='panel-title'>What the sensor reads vs. what it should read</div>",
                unsafe_allow_html=True)
    idx = frame.index.to_numpy()
    window = frame[(idx >= (hour - 23) * 60) & (idx < (hour + 1) * 60)]
    expected = detector.expected(window, owner)[suspect]
    t = [START + pd.Timedelta(minutes=int(i)) for i in window.index]
    gaps = np.diff(window.index.to_numpy(), prepend=window.index[0]) > 1
    fig = go.Figure()
    for h in hours.index[(hours.index >= hour - 23) & (hours.index <= hour)]:
        if hours.at[h, "alarm"]:
            fig.add_vrect(x0=when(h), x1=when(h + 1), fillcolor=STATUS[hours.at[h, "severity"]][0],
                          opacity=0.13, line_width=0)
    fig.add_trace(go.Scatter(x=t, y=window[suspect].where(~gaps), name="Measured", mode="lines",
                             line=dict(color=DETECTORS[owner][1], width=2)))
    fig.add_trace(go.Scatter(x=t, y=expected.where(~gaps), name="What it should read (model)", mode="lines",
                             line=dict(color=MUTED, width=2, dash="dash")))
    for level in SEVERITIES:              # legend entries for the shaded hours
        fig.add_trace(go.Bar(x=[None], y=[None], name=f"{level} alarm",
                             marker=dict(color=STATUS[level][0], opacity=0.5)))
    base_layout(fig, 300, hovermode="x unified", yaxis=dict(title=axis_label(suspect), gridcolor=GRID),
                xaxis=dict(showgrid=False))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("x: time of day (one point per minute) · y: the sensor's reading. "
               "Last 24 hours, steady chiller operation only (no line = chiller off or starting up). "
               "Shaded = alarm hour, colored by severity. No shading = no alarm, or fewer than "
               "30 steady minutes in that hour, which is not scored.")
    latest = window[window.index >= hour * 60]
    dev = (latest[suspect] - detector.expected(latest, owner)[suspect]).mean()
    m1, m2, m3, m4 = st.columns(4)
    u = f" {unit(suspect)}" if unit(suspect) else ""
    m1.metric("Reads (hour mean)", f"{latest[suspect].mean():.2f}{u}")
    m2.metric("Should read (hour mean)", f"{(latest[suspect] - dev).mean():.2f}{u}")
    m3.metric("Difference", f"{dev:+.2f}{u}")
    m4.metric("How unusual (× alarm limit)", f"{ratio.at[hour, suspect]:.1f}×")

    engineers = st.expander("Details for engineers: all sensors and both detector scores")
    tab_sensors, tab_scores = engineers.tabs(["Contributing sensors", "Detector scores"])
    with tab_sensors:
        top = ratio.loc[hour].nlargest(8)[::-1]
        fig = go.Figure(go.Bar(
            x=top.to_numpy(), y=[SENSOR_NAMES.get(s, s) for s in top.index], orientation="h",
            marker=dict(color=[TEAL if v > 1 else "#3A4A61" for v in top.to_numpy()], cornerradius=3),
            customdata=[" & ".join(DETECTORS[n][0] for n in PARTS if s in detector.columns[n])
                        for s in top.index],
            hovertemplate="%{y} (%{customdata}): %{x:.2f}× threshold<extra></extra>"))
        fig.add_vline(x=1, line=dict(color=MUTED, dash="dash", width=1))
        base_layout(fig, 320, showlegend=False,
                    xaxis=dict(type="log", title="shift ÷ threshold (dashed line = threshold)", gridcolor=GRID,
                               tickvals=LOG_TICKS, ticktext=[f"{v:g}" for v in LOG_TICKS],
                               range=[-1, np.log10(max(top.max(), 2) * 1.5)]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Teal = above its threshold. The sensors that move most show where the fault appears, "
                   "which is not always the faulty component itself.")

    with tab_scores:
        hrs = so_far.index.to_numpy()
        jump = np.r_[False, np.diff(hrs) > 1]
        xs_, ids = [], []
        for p, (h, j) in enumerate(zip(hrs, jump)):
            if j:
                xs_.append(p - 0.5)
                ids.append(None)
            xs_.append(p)
            ids.append(h)
        labels = [fmt(h) if h is not None else "" for h in ids]
        fig = go.Figure()
        for d, (label, color) in DETECTORS.items():
            r = so_far[f"score_{d}"] / detector.thresholds[d]
            fig.add_trace(go.Scatter(x=xs_, y=[r[h] if h is not None else None for h in ids], name=label,
                                     mode="lines", line=dict(color=color, width=2), customdata=labels,
                                     hovertemplate="%{customdata}<br>%{y:.2f}× threshold<extra>" + label + "</extra>"))
        fig.add_hline(y=1, line=dict(color=MUTED, dash="dash", width=1),
                      annotation_text="threshold", annotation_position="top left")
        day_start = np.flatnonzero(np.r_[True, np.diff(hrs // 24) > 0])
        day_start = day_start[::max(1, len(day_start) // 8)]
        base_layout(fig, 300, hovermode="closest",
                    yaxis=dict(type="log", title="score ÷ threshold", gridcolor=GRID,
                               tickvals=LOG_TICKS, ticktext=[f"{v:g}" for v in LOG_TICKS]),
                    xaxis=dict(tickvals=day_start, ticktext=[fmt(hrs[i], "%d %b") for i in day_start],
                               gridcolor=GRID, range=[-0.5, max(len(hrs) - 0.5, 24)]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("One step per analysed hour; the gaps are hours when the chiller was off or days not shown.")

# --- Fault isolation: which detector, which part, which sensor ---
def field(label, value, first=False):
    return (f"<div style='margin-top:{0 if first else 12}px;font-size:0.85rem'>"
            f"<span class='muted'>{label}</span><br>{value}</div>")


with isolation:
    st.markdown(
        f"<div class='panel'><div class='panel-title'>Where the fault shows</div>"
        + field("Status", chip(plant_level), first=True)
        + field("Part", PARTS[now["part"]])
        + field("Sensor", f"{SENSOR_NAMES.get(suspect, suspect)}<br>"
                          f"<span class='mono muted' style='font-size:0.78rem'>{suspect}</span>")
        + field("How unusual", f"<span class='mono'>{now['ratio']:.1f}×</span> the alarm limit")
        + (field("Most similar known fault", f"<b>{FAULTS[best]['name']}</b>") if best else "")
        + field("Detected by", f"{DETECTORS[owner][0]}<br><span class='muted' style='font-size:0.78rem'>"
                               f"autoencoder, hourly score</span>")
        + f"<div class='muted' style='margin-top:14px;font-size:0.75rem;border-top:1px solid {BORDER};"
          f"padding-top:10px'>Shows where the fault appears in the data - not necessarily "
          f"which component failed.</div></div>", unsafe_allow_html=True)

if st.session_state.playing:
    time.sleep(delay)
    st.rerun()
