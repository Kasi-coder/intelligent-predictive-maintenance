"""Maintenance action recommendation rules."""

def recommend_maintenance(probability: float, sensor: dict) -> dict:
    actions, inspections = [], []
    temp_gap = sensor.get("process_temperature", 0) - sensor.get("air_temperature", 0)
    torque = sensor.get("torque", 0)
    rpm = sensor.get("rotational_speed", 0)
    wear = sensor.get("tool_wear", 0)

    if probability >= 0.80:
        risk = "CRITICAL"
        actions.append("Stop or isolate the equipment and perform an immediate inspection.")
    elif probability >= 0.50:
        risk = "HIGH"
        actions.append("Schedule preventive maintenance at the earliest maintenance window.")
    elif probability >= 0.25:
        risk = "MEDIUM"
        actions.append("Increase monitoring frequency and plan a maintenance inspection.")
    else:
        risk = "LOW"
        actions.append("Continue normal operation with routine monitoring.")

    if wear >= 180:
        inspections.append("Inspect tool condition and consider tool replacement.")
    if temp_gap < 8.6 and rpm < 1380:
        inspections.append("Inspect cooling/heat-dissipation system and airflow.")
    power = torque * rpm * 2 * 3.141592653589793 / 60
    if power < 3500 or power > 9000:
        inspections.append("Inspect motor/load conditions and power transmission.")
    if wear * torque > 11000:
        inspections.append("Inspect rotating components for overstrain and excessive load.")
    if not inspections:
        inspections.append("No specific component alarm; perform standard preventive checks.")
    return {"risk_level": risk, "recommended_action": actions[0], "inspection_items": inspections}
