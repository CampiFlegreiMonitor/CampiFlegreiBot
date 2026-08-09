#!/usr/bin/env python3
"""
Campi Flegrei Seismic Monitor
Automated multi-indicator hazard assessment with Telegram push notifications.

Data sources:
  - INGV FDSN Event Web Service (real-time earthquake catalog)
  - INGV RING GNSS data gateway (daily RINEX for RITE station)
  - Computed metrics (b-value, inter-event times, swarm detection)

Date: August 2026
License: MIT
"""

import os
import json
import math
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import requests

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Campi Flegrei bounding box (approximate caldera extent)
CF_LAT_MIN = 40.78
CF_LAT_MAX = 40.88
CF_LON_MIN = 14.06
CF_LON_MAX = 14.20

# INGV FDSN Event Web Service
INGV_FDSN_URL = "https://webservices.ingv.it/fdsnws/event/1/query"
INGV_FDSN_COUNT_URL = "https://webservices.ingv.it/fdsnws/event/1/count"

# INGV RING GNSS data portal (RITE station daily position time series)
# Zenodo hosts the published displacement dataset:
ZENODO_RITE_URL = "https://zenodo.org/records/17132077/files/RITE_vertical_displacement.csv"

# State persistence (for tracking changes between runs)
STATE_FILE = "monitor_state.json"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("campi_flegrei")

# ──────────────────────────────────────────────────────────────────────
# THRESHOLD DEFINITIONS
# ──────────────────────────────────────────────────────────────────────

class AlertLevel(Enum):
    GREEN = "🟢 GREEN"
    AMBER = "🟡 AMBER"
    RED = "🔴 RED"

@dataclass
class Threshold:
    green_max: float
    amber_max: float
    unit: str
    description: str

THRESHOLDS = {
    "uplift_rate_mm_month": Threshold(green_max=30, amber_max=40, unit="mm/month",
        description="GNSS vertical displacement rate at RITE"),
    "m4_plus_annual_rate": Threshold(green_max=12, amber_max=20, unit="events/year",
        description="M4+ earthquakes per rolling 12 months"),
    "b_value_6mo": Threshold(green_max=0.80, amber_max=0.65, unit="(dimensionless, inverted)",
        description="Gutenberg-Richter b-value (lower = more large events)"),
    "microseismic_shallow_rate": Threshold(green_max=1, amber_max=5, unit="events/day",
        description="Earthquakes <1km depth near Accademia dome"),
    "days_since_m4_plus": Threshold(green_max=120, amber_max=240, unit="days",
        description="Days since last M4+ event (Weibull hazard window)"),
}

def classify_indicator(name: str, value: float) -> AlertLevel:
    """Classify an indicator value against thresholds."""
    t = THRESHOLDS[name]
    if name == "b_value_6mo":
        # Inverted: lower is worse
        if value <= t.amber_max:
            return AlertLevel.RED
        elif value <= t.green_max:
            return AlertLevel.AMBER
        return AlertLevel.GREEN
    else:
        if value >= t.amber_max:
            return AlertLevel.RED
        elif value >= t.green_max:
            return AlertLevel.AMBER
        return AlertLevel.GREEN

# ──────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────────────

def fetch_earthquakes(start_time: datetime, end_time: datetime,
                      min_magnitude: float = 0.0) -> List[Dict]:
    """
    Fetch earthquakes from INGV FDSN Event Web Service.
    Returns list of dicts with keys: time, lat, lon, depth, magnitude, event_id.
    """
    params = {
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minlatitude": CF_LAT_MIN,
        "maxlatitude": CF_LAT_MAX,
        "minlongitude": CF_LON_MIN,
        "maxlongitude": CF_LON_MAX,
        "minmagnitude": min_magnitude,
        "format": "json",
        "orderby": "time-asc",
        "limit": 10000,
    }

    try:
        resp = requests.get(INGV_FDSN_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch earthquakes: {e}")
        return []

    events = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None, None])
        mag = props.get("mag") or props.get("magnitude")
        if mag is None:
            continue
        events.append({
            "event_id": props.get("eventId") or props.get("code"),
            "time": props.get("time") or props.get("timeISO"),
            "lat": coords[1],
            "lon": coords[0],
            "depth": coords[2],
            "magnitude": float(mag),
        })

    log.info(f"Fetched {len(events)} events from INGV "
             f"({start_time.date()} to {end_time.date()}, M≥{min_magnitude})")
    return events


def fetch_rite_displacement(days_back: int = 90) -> Optional[List[Tuple[str, float]]]:
    """
    Attempt to fetch RITE station vertical displacement data.
    Falls back to INGV RING portal or Zenodo dataset.
    Returns list of (date_str, displacement_mm) tuples.
    """
    # Try INGV RING daily position time series first
    # The RING portal provides position time series at:
    # http://webring.gm.ingv.it/index.php/data-repository

    # Attempt Zenodo dataset (published RITE vertical displacement)
    try:
        resp = requests.get(ZENODO_RITE_URL, timeout=120)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            data = []
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        date_str = parts[0].strip()
                        disp = float(parts[1].strip())
                        data.append((date_str, disp))
                    except (ValueError, IndexError):
                        continue

            if data:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
                # Filter to recent data
                recent = [(d, v) for d, v in data
                          if datetime.fromisoformat(d.replace("Z", "+00:00")) > cutoff]
                log.info(f"Fetched {len(recent)} RITE displacement points (last {days_back}d)")
                return recent if recent else data[-days_back:]
    except Exception as e:
        log.warning(f"Zenodo RITE fetch failed: {e}")

    # Fallback: try INGV RING FTP-style access
    # http://webring.gm.ingv.it/rinex/YYYY/MM/DD/
    log.warning("Could not fetch RITE displacement data from Zenodo. "
                 "Manual check required at http://webring.gm.ingv.it")
    return None


# ──────────────────────────────────────────────────────────────────────
# METRIC COMPUTATIONS
# ──────────────────────────────────────────────────────────────────────

def compute_b_value(events: List[Dict], min_mag: float = 1.5) -> Optional[float]:
    """
    Compute Gutenberg-Richter b-value using maximum likelihood estimation:
    b = log10(e) / (M_mean - M_min)
    Requires at least 20 events above min_mag for a stable estimate.
    """
    mags = [e["magnitude"] for e in events if e["magnitude"] >= min_mag]
    if len(mags) < 20:
        log.info(f"Insufficient events ({len(mags)}) for b-value calc (need ≥20)")
        return None

    m_mean = sum(mags) / len(mags)
    b = math.log10(math.e) / (m_mean - min_mag)
    log.info(f"b-value = {b:.3f} (n={len(mags)}, M_mean={m_mean:.2f}, M_min={min_mag})")
    return round(b, 3)


def compute_m4_plus_rate(events: List[Dict], window_days: int = 365) -> float:
    """Count M4+ events in trailing window and annualize."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    count = 0
    for e in events:
        try:
            event_time = datetime.fromisoformat(
                e["time"].replace("Z", "+00:00")
            ) if isinstance(e["time"], str) else e["time"]
            if e["magnitude"] >= 4.0 and event_time > cutoff:
                count += 1
        except (ValueError, TypeError):
            continue
    rate = count  # Already annualized (365-day window)
    log.info(f"M4+ rate: {rate} events in last {window_days} days")
    return rate


def compute_uplift_rate(displacement_data: List[Tuple[str, float]],
                        window_days: int = 30) -> Optional[float]:
    """
    Compute recent uplift rate (mm/month) from GNSS displacement time series.
    Uses linear regression over the trailing window_days.
    """
    if not displacement_data or len(displacement_data) < 5:
        return None

    # Parse dates and filter to window
    parsed = []
    for date_str, disp_mm in displacement_data:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            parsed.append((dt, disp_mm))
        except ValueError:
            continue

    if len(parsed) < 5:
        return None

    parsed.sort(key=lambda x: x[0])
    cutoff = parsed[-1][0] - timedelta(days=window_days)
    window = [(dt, disp) for dt, disp in parsed if dt > cutoff]

    if len(window) < 5:
        window = parsed[-min(len(parsed), window_days):]

    if len(window) < 2:
        return None

    # Linear regression: y = a + b*x (x in days, y in mm)
    n = len(window)
    t0 = window[0][0]
    xs = [(dt - t0).total_seconds() / 86400 for dt, _ in window]
    ys = [disp for _, disp in window]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)

    if den == 0:
        return None

    slope_mm_per_day = num / den
    rate_mm_per_month = slope_mm_per_day * 30.4375

    log.info(f"Uplift rate: {rate_mm_per_month:.1f} mm/month "
             f"(slope {slope_mm_per_day:.3f} mm/day over {n} points)")
    return round(rate_mm_per_month, 1)


def compute_shallow_microseismic_rate(events: List[Dict],
                                       depth_threshold_km: float = 1.0) -> float:
    """Count events shallower than threshold in last 7 days, return daily rate."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    count = 0
    for e in events:
        try:
            event_time = datetime.fromisoformat(
                e["time"].replace("Z", "+00:00")
            ) if isinstance(e["time"], str) else e["time"]
            depth = e.get("depth", 999)
            if depth is not None and depth < depth_threshold_km and event_time > cutoff:
                count += 1
        except (ValueError, TypeError):
            continue
    daily_rate = count / 7.0
    log.info(f"Shallow (<{depth_threshold_km}km) microseismic rate: "
             f"{daily_rate:.2f}/day ({count} events in 7 days)")
    return round(daily_rate, 2)


def find_last_m4_plus(events: List[Dict]) -> Tuple[Optional[datetime], Optional[float]]:
    """Find the most recent M4+ event and return its time and magnitude."""
    m4_events = []
    for e in sorted(events, key=lambda x: x["time"], reverse=True):
        if e["magnitude"] >= 4.0:
            try:
                dt = datetime.fromisoformat(
                    e["time"].replace("Z", "+00:00")
                ) if isinstance(e["time"], str) else e["time"]
                m4_events.append((dt, e["magnitude"]))
            except (ValueError, TypeError):
                continue

    if not m4_events:
        return None, None

    m4_events.sort(key=lambda x: x[0], reverse=True)
    return m4_events[0]


def detect_swarm_bursts(events: List[Dict], window_minutes: int = 30,
                        min_events: int = 10) -> List[Dict]:
    """
    Detect burst-like swarms: clusters of ≥ min_events within window_minutes.
    Returns list of burst dicts with start_time, event_count, max_magnitude.
    """
    if len(events) < min_events:
        return []

    parsed = []
    for e in events:
        try:
            dt = datetime.fromisoformat(
                e["time"].replace("Z", "+00:00")
            ) if isinstance(e["time"], str) else e["time"]
            parsed.append({"time": dt, "magnitude": e["magnitude"]})
        except (ValueError, TypeError):
            continue

    parsed.sort(key=lambda x: x["time"])

    bursts = []
    i = 0
    while i < len(parsed):
        window_end = parsed[i]["time"] + timedelta(minutes=window_minutes)
        j = i
        cluster = []
        while j < len(parsed) and parsed[j]["time"] <= window_end:
            cluster.append(parsed[j])
            j += 1

        if len(cluster) >= min_events:
            bursts.append({
                "start_time": cluster[0]["time"].isoformat(),
                "event_count": len(cluster),
                "max_magnitude": max(c["magnitude"] for c in cluster),
                "window_minutes": (cluster[-1]["time"] - cluster[0]["time"]).total_seconds() / 60,
            })
            i = j  # Skip past this burst
        else:
            i += 1

    log.info(f"Detected {len(bursts)} swarm bursts in dataset")
    return bursts


# ──────────────────────────────────────────────────────────────────────
# NOTIFICATION SYSTEM
# ──────────────────────────────────────────────────────────────────────

def send_telegram(message: str, parse_mode: str = "Markdown") -> bool:
    """Send a push notification via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not set — skipping notification")
        log.info(f"Message would be:\n{message}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, data=payload, timeout=30)
        resp.raise_for_status()
        log.info("Telegram notification sent successfully")
        return True
    except Exception as e:
        log.error(f"Failed to send Telegram notification: {e}")
        return False


def format_alert_report(indicators: Dict) -> str:
    """Format a comprehensive status report for Telegram."""
    lines = []
    lines.append("🌋 *Campi Flegrei Seismic Monitor*")
    lines.append(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("*Indicator Status:*")

    overall_alert = AlertLevel.GREEN
    for name, result in indicators.items():
        level = result["level"]
        value = result["value"]
        desc = result["description"]
        lines.append(f"{level.value} `{name}`: {value} — {desc}")

        if level == AlertLevel.RED:
            overall_alert = AlertLevel.RED
        elif level == AlertLevel.AMBER and overall_alert != AlertLevel.RED:
            overall_alert = AlertLevel.AMBER

    lines.append("")
    if overall_alert == AlertLevel.RED:
        lines.append("⚠️ *OVERALL: RED ALERT* — Multiple indicators at critical thresholds. "
                     "Check INGV official assessment: https://www.ingv.it/en/campi-flegrei")
    elif overall_alert == AlertLevel.AMBER:
        lines.append("⚡ *OVERALL: AMBER — Elevated activity detected.* "
                     "Monitor closely for escalation.")
    else:
        lines.append("✅ *OVERALL: GREEN* — All indicators within expected range.")

    return "\n".join(lines)


def format_new_event_alert(event: Dict) -> str:
    """Format a new significant earthquake alert."""
    mag = event["magnitude"]
    depth = event.get("depth", "?")
    time_str = event.get("time", "?")

    if mag >= 5.0:
        emoji = "🚨🚨🚨"
        urgency = "*M5+ EARTHQUAKE DETECTED*"
    elif mag >= 4.0:
        emoji = "🚨"
        urgency = "*M4+ EARTHQUAKE DETECTED*"
    elif mag >= 3.0:
        emoji = "⚠️"
        urgency = "*M3+ Earthquake*"
    else:
        return None  # Don't notify for small events

    msg = (
        f"{emoji} {urgency}\n\n"
        f"*Magnitude:* {mag}\n"
        f"*Depth:* {depth} km\n"
        f"*Time:* {time_str}\n"
        f"*Location:* Campi Flegrei caldera\n\n"
        f"INGV event page: https://terremoti.ingv.it\n"
        f"Official monitoring: https://www.ingv.it/en/campi-flegrei"
    )
    return msg


# ──────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ──────────────────────────────────────────────────────────────────────

def load_state() -> Dict:
    """Load persistent state from disk."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load state file: {e}")
    return {
        "last_m4_plus_time": None,
        "last_run": None,
        "known_event_ids": [],
        "previous_indicators": {},
    }


def save_state(state: Dict):
    """Save persistent state to disk."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        log.error(f"Could not save state file: {e}")


# ──────────────────────────────────────────────────────────────────────
# MAIN MONITORING ROUTINE
# ──────────────────────────────────────────────────────────────────────

def run_monitor():
    """Main monitoring routine — called on schedule."""
    log.info("=" * 60)
    log.info("Campi Flegrei Monitor — starting run")
    log.info("=" * 60)

    state = load_state()
    now = datetime.now(timezone.utc)

    # ─── Fetch data ───────────────────────────────────────────────────
    # Get events from last 365 days for rate calculations
    events_year = fetch_earthquakes(
        start_time=now - timedelta(days=365),
        end_time=now,
        min_magnitude=0.0
    )

    # Get events from last 6 months for b-value
    events_6mo = fetch_earthquakes(
        start_time=now - timedelta(days=183),
        end_time=now,
        min_magnitude=0.0
    )

    # Get events from last 7 days for shallow microseismicity
    events_7d = fetch_earthquakes(
        start_time=now - timedelta(days=7),
        end_time=now,
        min_magnitude=0.0
    )

    # Fetch RITE displacement data
    displacement_data = fetch_rite_displacement(days_back=90)

    # ─── Compute indicators ────────────────────────────────────────────
    indicators = {}

    # 1. Uplift rate
    if displacement_data:
        uplift_rate = compute_uplift_rate(displacement_data, window_days=30)
        if uplift_rate is not None:
            level = classify_indicator("uplift_rate_mm_month", uplift_rate)
            indicators["uplift_rate_mm_month"] = {
                "value": f"{uplift_rate} mm/month",
                "level": level,
                "description": "RITE GNSS vertical displacement rate"
            }
    else:
        indicators["uplift_rate_mm_month"] = {
            "value": "N/A",
            "level": AlertLevel.GREEN,
            "description": "RITE GNSS data unavailable — check manually"
        }

    # 2. M4+ annual rate
    m4_rate = compute_m4_plus_rate(events_year, window_days=365)
    level = classify_indicator("m4_plus_annual_rate", float(m4_rate))
    indicators["m4_plus_annual_rate"] = {
        "value": f"{m4_rate} events/year",
        "level": level,
        "description": "M4+ events in trailing 12 months"
    }

    # 3. b-value (6-month rolling window)
    b_val = compute_b_value(events_6mo, min_mag=1.5)
    if b_val is not None:
        level = classify_indicator("b_value_6mo", b_val)
        indicators["b_value_6mo"] = {
            "value": f"{b_val}",
            "level": level,
            "description": "Gutenberg-Richter b-value (6mo window, lower=worse)"
        }
    else:
        indicators["b_value_6mo"] = {
            "value": "N/A",
            "level": AlertLevel.GREEN,
            "description": "Insufficient events for b-value calculation"
        }

    # 4. Shallow microseismicity rate
    shallow_rate = compute_shallow_microseismic_rate(events_7d, depth_threshold_km=1.0)
    level = classify_indicator("microseismic_shallow_rate", shallow_rate)
    indicators["microseismic_shallow_rate"] = {
        "value": f"{shallow_rate}/day",
        "level": level,
        "description": "Events <1km depth (7-day avg)"
    }

    # 5. Days since last M4+
    last_m4_time, last_m4_mag = find_last_m4_plus(events_year)
    if last_m4_time:
        days_since = (now - last_m4_time).days
        level = classify_indicator("days_since_m4_plus", float(days_since))
        indicators["days_since_m4_plus"] = {
            "value": f"{days_since} days (last: M{last_m4_mag})",
            "level": level,
            "description": "Weibull hazard window (peak ~105 days post-event)"
        }

        # Check for NEW M4+ event since last run
        prev_last_m4 = state.get("last_m4_plus_time")
        if prev_last_m4 and last_m4_time.isoformat() != prev_last_m4:
            # New M4+ event detected!
            for e in reversed(sorted(events_year, key=lambda x: x["time"])):
                if e["magnitude"] >= 4.0:
                    try:
                        e_time = datetime.fromisoformat(
                            e["time"].replace("Z", "+00:00")
                        ) if isinstance(e["time"], str) else e["time"]
                        if e_time.isoformat() != prev_last_m4:
                            alert_msg = format_new_event_alert(e)
                            if alert_msg:
                                send_telegram(alert_msg)
                                break
                    except (ValueError, TypeError):
                        continue

        state["last_m4_plus_time"] = last_m4_time.isoformat()

    # 6. Swarm burst detection (last 7 days)
    bursts = detect_swarm_bursts(events_7d, window_minutes=30, min_events=10)
    if len(bursts) > 0:
        recent_bursts = [b for b in bursts if
                         datetime.fromisoformat(b["start_time"]) >
                         now - timedelta(hours=24)]
        if len(recent_bursts) > 0:
            indicators["swarm_burst_24h"] = {
                "value": f"{len(recent_bursts)} bursts",
                "level": AlertLevel.AMBER,
                "description": "Burst-like swarms in last 24h"
            }

    # ─── Check for threshold crossings ────────────────────────────────
    prev_indicators = state.get("previous_indicators", {})
    threshold_crossed = False
    crossings = []

    for name, result in indicators.items():
        prev_level_str = prev_indicators.get(name, {}).get("level", "GREEN")
        curr_level = result["level"]

        if curr_level.value != prev_level_str:
            threshold_crossed = True
            crossings.append(
                f"{name}: {prev_level_str} → {curr_level.value} "
                f"({result['value']})"
            )

    # ─── Send notifications ────────────────────────────────────────────
    # Always send if any indicator changed level
    if threshold_crossed:
        msg = format_alert_report(indicators)
        msg += "\n\n*🔄 THRESHOLD CROSSINGS:*\n"
        for c in crossings:
            msg += f"• {c}\n"
        send_telegram(msg)

    # Send periodic summary (every run if configured, or on changes)
    elif now.hour in [6, 12, 18, 0]:  # 4x daily summary
        msg = format_alert_report(indicators)
        send_telegram(msg)

    # ─── Update state ──────────────────────────────────────────────────
    state["last_run"] = now.isoformat()
    state["previous_indicators"] = {
        name: {"level": result["level"].value, "value": result["value"]}
        for name, result in indicators.items()
    }

    # Track known event IDs (keep last 10000)
    known_ids = set(state.get("known_event_ids", []))
    new_events = []
    for e in events_7d:
        eid = e.get("event_id")
        if eid and eid not in known_ids:
            new_events.append(e)
            known_ids.add(eid)

    state["known_event_ids"] = list(known_ids)[-10000:]
    save_state(state)

    log.info(f"Run complete. Indicators: {len(indicators)}. "
             f"Threshold crossings: {len(crossings)}. "
             f"New events: {len(new_events)}.")


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_monitor()

