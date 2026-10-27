import json
import numpy as np
import pandas as pd

# Fault signatures: which faults could explain what the plant is doing now?
#
# Each fault moves the sensors in its own pattern - which sensors drift and in
# which direction. An alarm hour's pattern is the vector of signed hourly
# z-scores of all sensors of both detectors. Each fault class gets a typical
# pattern (the mean direction of its alarm hours on validation days, made by
# fault_signatures.py), and an hour is compared with every class by cosine
# similarity: 1 = the same pattern, 0 = unrelated, below 0 = the opposite.
#
# Sensor biases are split by sign (reads high / reads low): the two directions
# move the sensors in opposite ways, and averaging them would cancel out.

SIGNATURES_FILE = "models/fault_signatures.json"

# Plain-language description of each fault class, for the dashboard
FAULTS = {
    # Leaking and stuck bypass valves move the sensors the same way (the --demo
    # check confused them in half the hours), so they are one class here.
    "bypass_valve": dict(
        name="Bypass valve leaking or stuck", where="Chilled-water loop",
        what="Part of the chilled water skips the load and goes straight back to the chiller.",
        looks="Loop temperatures and flows shift, and the chiller and tower follow."),
    "secondary_chilled_water_pressure_bias_pos": dict(
        name="Loop pressure sensor reads high", where="Chilled-water loop",
        what="The pressure sensor reports more pressure than there is, so the pumps chase a wrong value.",
        looks="Pump speed and power drop while the pressure reading itself looks normal."),
    "secondary_chilled_water_pressure_bias_neg": dict(
        name="Loop pressure sensor reads low", where="Chilled-water loop",
        what="The pressure sensor reports less pressure than there is, so the pumps chase a wrong value.",
        looks="Pump speed and power rise while the pressure reading itself looks normal."),
    "chiller_bias_pos": dict(
        name="Chiller supply temperature sensor reads high", where="Chiller",
        what="The sensor reports warmer water than the chiller really supplies.",
        looks="One temperature drifts away from what all the other sensors imply."),
    "chiller_bias_neg": dict(
        name="Chiller supply temperature sensor reads low", where="Chiller",
        what="The sensor reports colder water than the chiller really supplies.",
        looks="One temperature drifts away from what all the other sensors imply."),
    "chiller_fouling": dict(
        name="Chiller fouling", where="Chiller",
        what="Dirty heat-exchanger tubes make the chiller less efficient.",
        looks="Small, steady changes in chiller temperatures and power."),
    "coolingtower_bias_pos": dict(
        name="Cooling tower supply temperature sensor reads high", where="Cooling tower",
        what="The sensor reports warmer water than the tower really supplies.",
        looks="One temperature drifts away from what all the other sensors imply."),
    "coolingtower_bias_neg": dict(
        name="Cooling tower supply temperature sensor reads low", where="Cooling tower",
        what="The sensor reports colder water than the tower really supplies.",
        looks="One temperature drifts away from what all the other sensors imply."),
    "coolingtower_fouling": dict(
        name="Cooling tower fouling", where="Cooling tower",
        what="Dirt in the tower reduces how well it cools the water.",
        looks="Warmer tower and condenser water for the same weather."),
    "coolingtower_PI": dict(
        name="Cooling tower control badly tuned", where="Cooling tower",
        what="The tower fan controller over- or under-reacts.",
        looks="Tower water temperature misses its setpoint."),
}


def fault_class(fault_type, severity):
    """Class of a fault file: its type, and for sensor biases the sign of the bias."""
    if fault_type.startswith("bypass_"):
        return "bypass_valve"
    if fault_type.endswith("_bias"):
        return f"{fault_type}_{'neg' if str(severity).startswith('-') else 'pos'}"
    return fault_type


def signature_vectors(signed_z):
    """One row per hour: the signed z of every sensor of every detector, as a unit vector."""
    vec = pd.concat({name: z for name, z in signed_z.items()}, axis=1)
    vec.columns = [f"{name}:{col}" for name, col in vec.columns]
    norm = np.sqrt((vec ** 2).sum(axis=1)).replace(0, np.nan)
    return vec.div(norm, axis=0)


def load_signatures(path=SIGNATURES_FILE):
    with open(path) as f:
        return json.load(f)


def match(vectors, signatures):
    """Cosine similarity of each hour's pattern with each fault class (hours x classes)."""
    features = signatures["features"]
    centroids = pd.DataFrame({k: v["centroid"] for k, v in signatures["classes"].items()},
                             index=features)
    return vectors[features].fillna(0) @ centroids
