import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from detector import Detector, LABELS
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

# Operator dashboard: replays one plant file hour by hour, as if the data were
# arriving live, and shows what the edge detector (detector.py) reports.
#
# Run (from the project folder):  streamlit run src/dashboard.py
# Needs: train_model.py and subsystems.py --hybrid run first.

START = pd.Timestamp("2021-01-01")       # the files hold one year of 1-minute data
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"   # series colors (main, pumps, expected)
GOOD, CRITICAL = "#0ca30c", "#d03b3b"                  # status colors, always with icon + label
COLORS = {"main": BLUE, "pumps": ORANGE}
LOG_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
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


def when(hour):
    return START + pd.Timedelta(hours=int(hour))


def sensor_label(code):
    return f"{SENSOR_NAMES.get(code, code)} ({code})"


@st.cache_resource
def get_detector():
    return Detector.from_saved()


@st.cache_data
def get_catalog():
    catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
    names = {}
    for _, row in catalog.iterrows():
        if row["label"] == "normal":
            names[row["filename"]] = "No fault (fault-free file)"
        else:
            severity = row["severity"] if isinstance(row["severity"], str) else ""
            names[row["filename"]] = f"{row['fault_type'].replace('_', ' ')} {severity}".strip()
    return names


@st.cache_data(show_spinner="Running the detector on the whole file...")
def analyse(filename, test_days_only):
    detector = get_detector()
    df = load_plant_file(f"{DATA_DIR}/{filename}", detector.load_columns())
    keep, _, _, _ = steady_state_mask(df)
    if test_days_only:
        keep = keep & (day_split(len(df)) == "test")
    idx = np.flatnonzero(keep)
    frame = df.iloc[idx]
    z = detector.z_scores(frame, idx)
    hours = detector.hourly(frame, idx, z)
    return frame, z, hours


# --- Sidebar: what to replay ---
st.set_page_config(page_title="Chiller plant FDD", layout="wide")
detector = get_detector()
catalog = get_catalog()
files = list(catalog)
default = next(i for i, f in enumerate(files) if f != "ChillerPlant.csv")

with st.sidebar:
    st.header("Replay")
    filename = st.selectbox("Plant file", files, index=default, format_func=catalog.get)
    test_days_only = st.toggle("Test days only", value=True,
                               help="Days the models never saw in training or threshold setting.")
    step = st.select_slider("Hours per step", [1, 3, 6, 24], value=3)
    delay = st.select_slider("Seconds between steps", [0.2, 0.5, 1.0, 2.0], value=0.5)
    st.caption(f"Detector: main model + pump detector, thresholds for a "
               f"{detector.budget:g}% false-alarm budget on normal hours.")

frame, z, hours = analyse(filename, test_days_only)
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

# --- Header and controls ---
st.title("Chiller plant fault detection")
st.caption(f"Replaying **{catalog[filename]}** - {n} steady-state hours analysed")
c1, c2, c3, c4 = st.columns([1, 1, 1, 6])
c1.button("◀ Back", on_click=move, args=(-step,), width="stretch")
c2.button("Pause" if st.session_state.playing else "▶ Play", on_click=toggle_play,
          width="stretch")
c3.button("Next ▶", on_click=move, args=(step,), width="stretch")
c4.slider("Hour", 0, n - 1, key="pos", label_visibility="collapsed")

pos = st.session_state.pos
hour = hours.index[pos]
now = hours.loc[hour]
names = list(z)

# --- Status tiles ---
def tile(column, title, alarm, detail):
    color, icon, word = (CRITICAL, "⚠", "ALARM") if alarm else (GOOD, "✓", "Normal")
    column.markdown(
        f"<div style='border-left:6px solid {color};background:{color}1f;border-radius:8px;"
        f"padding:12px 16px'><div style='font-size:0.85rem;opacity:0.8'>{title}</div>"
        f"<div style='font-size:1.5rem;font-weight:600'>{icon} {word}</div>"
        f"<div style='font-size:0.85rem;opacity:0.8'>{detail}</div></div>",
        unsafe_allow_html=True)


st.subheader(when(hour).strftime("%d %b, %H:00"))
cols = st.columns(len(names) + 2)
tile(cols[0], "Plant", now["alarm"],
     f"Fault in the {now['subsystem']}" if now["alarm"] else "All detectors below threshold")
for col, name in zip(cols[1:], names):
    ratio = now[f"score_{name}"] / detector.thresholds[name]
    tile(col, LABELS[name].capitalize(), now[f"alarm_{name}"], f"Score {ratio:.2f}× threshold")
with cols[-1]:
    worst = max(names, key=lambda m: now[f"score_{m}"] / detector.thresholds[m])
    suspect = now["suspect_sensor"] if now["alarm"] else now[f"sensor_{worst}"]
    st.markdown(
        f"<div style='border-left:6px solid {GRAY};border-radius:8px;padding:12px 16px'>"
        f"<div style='font-size:0.85rem;opacity:0.8'>{'Suspect sensor' if now['alarm'] else 'Most-shifted sensor'}</div>"
        f"<div style='font-size:1.1rem;font-weight:600'>{SENSOR_NAMES.get(suspect, suspect)}</div>"
        f"<div style='font-size:0.85rem;opacity:0.8'>{suspect}</div></div>",
        unsafe_allow_html=True)

# --- Timeline: score / threshold of each detector, up to the current hour ---
st.markdown("#### Detector scores over time")
full = pd.RangeIndex(hours.index[0], hour + 1)          # gaps (plant off) break the lines
past = hours.loc[:hour].reindex(full)
x = [when(h) for h in full]
fig = go.Figure()
for name in names:
    fig.add_trace(go.Scatter(
        x=x, y=past[f"score_{name}"] / detector.thresholds[name], name=LABELS[name].capitalize(),
        mode="lines", line=dict(color=COLORS[name], width=2), connectgaps=False,
        hovertemplate="%{y:.2f}× threshold<extra>" + LABELS[name] + "</extra>"))
alarms = past[past["alarm"] == True]
fig.add_trace(go.Scatter(
    x=[when(h) for h in alarms.index], y=[0.15] * len(alarms), name="Alarm hour",
    mode="markers", marker=dict(symbol="line-ns-open", size=10, color=CRITICAL, line=dict(width=2)),
    customdata=alarms["suspect_sensor"],
    hovertemplate="Alarm - suspect %{customdata}<extra></extra>"))
fig.add_hline(y=1, line=dict(color=GRAY, dash="dash", width=1),
              annotation_text="threshold", annotation_position="top left")
fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                  yaxis=dict(type="log", title="score ÷ threshold", gridcolor="rgba(128,128,128,0.15)",
                             tickvals=LOG_TICKS, ticktext=[f"{v:g}" for v in LOG_TICKS]),
                  xaxis=dict(showgrid=False), legend=dict(orientation="h", y=1.12, x=0))
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)

# --- Why this hour: the most-shifted sensors, both detectors on one scale ---
with left:
    st.markdown("#### Most-shifted sensors this hour")
    rows = []
    for name in names:
        for sensor, value in z[name].loc[hour].items():
            rows.append((name, sensor, value / detector.thresholds[name]))
    top = pd.DataFrame(rows, columns=["detector", "sensor", "ratio"]).nlargest(8, "ratio")[::-1]
    fig = go.Figure()
    for name in names:
        part = top[top["detector"] == name]
        fig.add_trace(go.Bar(
            x=part["ratio"], y=[SENSOR_NAMES.get(s, s) for s in part["sensor"]], orientation="h",
            name=LABELS[name].capitalize(), marker=dict(color=COLORS[name], cornerradius=4),
            hovertemplate="%{y}: %{x:.2f}× threshold<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=GRAY, dash="dash", width=1),
                  annotation_text="threshold", annotation_position="bottom right")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), barmode="overlay",
                      yaxis=dict(categoryorder="array", categoryarray=[SENSOR_NAMES.get(s, s) for s in top["sensor"]]),
                      xaxis=dict(title="shift ÷ detector threshold", gridcolor="rgba(128,128,128,0.15)"),
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, width="stretch")

# --- The suspect sensor: measured vs what the model expects, last 24 hours ---
with right:
    st.markdown(f"#### {SENSOR_NAMES.get(suspect, suspect)}: measured vs expected")
    owner = next(m for m in names if suspect in detector.columns[m])
    rows_idx = frame.index.to_numpy()
    window = frame[(rows_idx >= (hour - 23) * 60) & (rows_idx < (hour + 1) * 60)]
    expected = detector.expected(window, owner)[suspect]
    t = [START + pd.Timedelta(minutes=int(i)) for i in window.index]
    gaps = np.diff(window.index.to_numpy(), prepend=window.index[0]) > 1   # break lines when off
    measured = window[suspect].where(~gaps)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=measured, name="Measured", mode="lines",
                             line=dict(color=COLORS[owner], width=2)))
    fig.add_trace(go.Scatter(x=t, y=expected.where(~gaps), name="Expected by the model", mode="lines",
                             line=dict(color=GRAY, width=2, dash="dash")))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis=dict(title=suspect, gridcolor="rgba(128,128,128,0.15)"),
                      xaxis=dict(showgrid=False), legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Last 24 hours of steady-state data. A gap that stays between the two lines "
               "for hours is what raises the alarm.")

# --- Summary of the replay so far, and the alarm log ---
so_far = hours.loc[:hour]
m1, m2, m3 = st.columns(3)
m1.metric("Hours replayed", len(so_far))
m2.metric("Hours with an alarm", f"{so_far['alarm'].mean() * 100:.1f}%")
m3.metric("First alarm", when(so_far.index[so_far["alarm"]][0]).strftime("%d %b, %H:00")
          if so_far["alarm"].any() else "-")

with st.expander("Alarm log (table view)"):
    log = so_far[so_far["alarm"]].copy()
    log.insert(0, "time", [when(h).strftime("%d %b, %H:00") for h in log.index])
    for name in names:
        log[f"{name} score ÷ threshold"] = (log[f"score_{name}"] / detector.thresholds[name]).round(2)
    st.dataframe(log[["time", "subsystem", "suspect_sensor"]
                     + [f"{n} score ÷ threshold" for n in names]][::-1],
                 hide_index=True, width="stretch")

if st.session_state.playing:
    time.sleep(delay)
    st.rerun()
