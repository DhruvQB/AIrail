# Wraps /trains/{no} + /trains/between
"""
tools/railradar_tool.py

LangChain tools that wrap the RailRadar REST API.
Each function is decorated with @tool so LangGraph agents can call them.

Endpoints used:
  GET /api/v1/trains/{trainNumber}?dataType=live   → get_live_train_status
  GET /api/v1/trains/between?from=&to=             → search_trains_between
  GET /api/v1/trains/{trainNumber}/schedule        → get_train_schedule
  GET /api/v1/stations/{stationCode}/live          → get_live_station_board
"""

import httpx
from typing import Optional
from langchain_core.tools import tool

from app.config import settings

TIMEOUT = 10.0  # seconds


def _headers() -> dict:
    return {"X-API-Key": settings.RAILRADAR_API_KEY}


def _minutes_to_hhmm(minutes: int) -> str:
    """Convert minutes-from-midnight to HH:MM string."""
    if not minutes:
        return "00:00"
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


def _epoch_to_hhmm(ts: Optional[int]) -> Optional[str]:
    """Convert epoch timestamp to HH:MM string in IST."""
    if not ts:
        return None
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.fromtimestamp(ts, tz=ist).strftime("%H:%M")


def _unwrap(payload: dict) -> dict:
    if isinstance(payload, dict) and "data" in payload and "success" in payload:
        return payload["data"]
    return payload


# ---------------------------------------------------------------------------
# Tool 1 — Live Train Status
# ---------------------------------------------------------------------------

@tool
def get_live_train_status(train_number: str, journey_date: Optional[str] = None, include_full_route: bool = False) -> dict:
    """
    Get the real-time running status of a train.
    You can optionally provide a journey_date in YYYY-MM-DD format.
    If not provided, it defaults to today's date in IST.
    """
    try:
        params = {"dataType": "live"}
        if journey_date:
            params["journeyDate"] = journey_date
        else:
            from datetime import datetime, timezone, timedelta
            ist = timezone(timedelta(hours=5, minutes=30))
            params["journeyDate"] = datetime.now(ist).strftime("%Y-%m-%d")

        response = httpx.get(
            f"{settings.RAILRADAR_BASE_URL}/trains/{train_number}",
            params=params,
            headers=_headers(),
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = _unwrap(response.json())

        current = data.get("currentLocation", {})
        route = data.get("route", [])
        
        # Fetch train name from schedule endpoint since live doesn't provide it
        train_name = "Unknown"
        journey_date = data.get("journeyDate")
        if journey_date:
            try:
                sched_resp = httpx.get(
                    f"{settings.RAILRADAR_BASE_URL}/trains/{train_number}/schedule",
                    params={"journeyDate": journey_date},
                    headers=_headers(),
                    timeout=TIMEOUT,
                )
                if sched_resp.status_code == 200:
                    sched_data = _unwrap(sched_resp.json())
                    train_name = sched_data.get("train", {}).get("name", "Unknown")
            except Exception:
                pass  # Fallback to "Unknown" if the secondary call fails

        last_updated_str = data.get("lastUpdatedAt")
        last_updated_ts = None
        if last_updated_str:
            try:
                from datetime import datetime
                last_updated_ts = datetime.fromisoformat(last_updated_str).timestamp()
            except Exception:
                pass

        # Load station names from local JSON for enrichment
        station_names = {}
        try:
            import json
            import os
            json_path = os.path.join(os.getcwd(), "railwayStationsList.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    stn_data = json.load(f)
                    for s in stn_data.get("stations", []):
                        station_names[s["stnCode"]] = s["stnName"]
        except Exception:
            pass

        processed_route = []
        for stop in route:
            arr_ts = stop.get("actualArrival")
            dep_ts = stop.get("actualDeparture")
            
            raw_exp_arr = arr_ts
            raw_exp_dep = dep_ts

            if last_updated_ts:
                if arr_ts and arr_ts > last_updated_ts + 60:
                    arr_ts = None
                if dep_ts and dep_ts > last_updated_ts + 60:
                    dep_ts = None
                    
            delay_arr = stop.get("delayArrivalMinutes")
            delay_dep = stop.get("delayDepartureMinutes")

            stn_code = stop.get("stationCode")
            processed_stop = {
                **stop,
                "stationName": station_names.get(stn_code, stn_code),
                "scheduledArrival": _epoch_to_hhmm(stop.get("scheduledArrival")),
                "scheduledDeparture": _epoch_to_hhmm(stop.get("scheduledDeparture")),
                "actualArrival": _epoch_to_hhmm(arr_ts),
                "actualDeparture": _epoch_to_hhmm(dep_ts),
                "delayArrivalMinutes": delay_arr if arr_ts is not None else None,
                "delayDepartureMinutes": delay_dep if dep_ts is not None else None,
                "expectedDelayArrivalMinutes": delay_arr if arr_ts is None else None,
                "expectedDelayDepartureMinutes": delay_dep if dep_ts is None else None,
                "_raw_exp_arr": raw_exp_arr,
                "_raw_exp_dep": raw_exp_dep,
            }
            processed_route.append(processed_stop)

        # Determine next stop more reliably
        current_stn_code = current.get("stationCode")
        current_idx = -1
        for i, stop in enumerate(processed_route):
            if stop.get("stationCode") == current_stn_code:
                current_idx = i
                break
        
        next_stop = None
        if current_idx != -1:
            if current.get("status") == "AT_STATION":
                if current_idx + 1 < len(processed_route):
                    next_stop = processed_route[current_idx + 1]
            else:
                # RUNNING: look for the first station that hasn't been reached yet
                for i in range(current_idx, len(processed_route)):
                    if processed_route[i].get("actualArrival") is None:
                        next_stop = processed_route[i]
                        break

        # Fallback to original logic if next_stop is still None
        if not next_stop:
            for i, stop in enumerate(processed_route):
                if stop.get("actualArrival") is None and stop.get("actualDeparture") is None:
                    if current.get("status") == "AT_STATION" and current.get("stationCode") == stop.get("stationCode"):
                        if i + 1 < len(processed_route):
                            next_stop = processed_route[i + 1]
                        break
                    else:
                        next_stop = stop
                        break
            
        if next_stop:
            stn_code = next_stop.get("stationCode")
            current["nextStation"] = {
                "stationCode": stn_code,
                "stationName": station_names.get(stn_code, stn_code),
                "expectedArrival": _epoch_to_hhmm(next_stop.get("_raw_exp_arr")) or next_stop.get("scheduledArrival"),
                "expectedDeparture": _epoch_to_hhmm(next_stop.get("_raw_exp_dep")) or next_stop.get("scheduledDeparture"),
                "distanceToNextStationKm": None,
                "delayArrivalMinutes": next_stop.get("expectedDelayArrivalMinutes") or next_stop.get("delayArrivalMinutes")
            }

        curr_stn = current.get("stationCode")
        if curr_stn:
            current["stationName"] = station_names.get(curr_stn, curr_stn)

        for stop in processed_route:
            stop.pop("_raw_exp_arr", None)
            stop.pop("_raw_exp_dep", None)

        if not include_full_route and processed_route:
            idx = next((i for i, stop in enumerate(processed_route) if stop == next_stop), -1)
            if idx != -1:
                start = max(0, idx - 1)
                end = min(len(processed_route), idx + 3)
                processed_route = processed_route[start:end]
            else:
                processed_route = processed_route[-3:]


        return {
            "trainNumber": data.get("trainNumber", train_number),
            "trainName": train_name,
            "journeyDate": data.get("journeyDate"),
            "lastUpdatedAt": data.get("lastUpdatedAt"),
            "currentLocation": current,
            "route": processed_route,
        }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Train {train_number} not found.", "error_code": "TRAIN_NOT_FOUND"}
        return {"error": f"RailRadar API error: {e.response.status_code}", "error_code": "API_ERROR"}
    except httpx.TimeoutException:
        return {"error": "RailRadar API timed out.", "error_code": "API_TIMEOUT"}
    except Exception as e:
        return {"error": str(e), "error_code": "UNKNOWN_ERROR"}


# ---------------------------------------------------------------------------
# Tool 2 — Search Trains Between Stations
# ---------------------------------------------------------------------------

@tool
def search_trains_between(from_station: str, to_station: str, limit: Optional[int] = 10) -> dict:
    """
    Find all trains that run between two stations.
    """
    try:
        response = httpx.get(
            f"{settings.RAILRADAR_BASE_URL}/trains/between",
            params={"from": from_station, "to": to_station},
            headers=_headers(),
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = _unwrap(response.json())

        trains = data.get("trains", [])
        simplified = []
        for t in trains:
            from_stop = t.get("fromStationSchedule", {})
            to_stop = t.get("toStationSchedule", {})
            
            days_obj = t.get("runningDays", {})
            if "days" in days_obj and isinstance(days_obj["days"], list):
                days_list = days_obj["days"]
            else:
                days_list = [day[:3].capitalize() for day, runs in days_obj.items() if isinstance(runs, bool) and runs]

            from_time = (from_stop.get("day", 1) - 1) * 24 * 60 + from_stop.get("departureMinutes", 0)
            to_time = (to_stop.get("day", 1) - 1) * 24 * 60 + to_stop.get("arrivalMinutes", 0)
            calculated_duration = to_time - from_time if to_time > from_time else t.get("travelTimeMinutes")
            
            from_dist = from_stop.get("distanceFromSourceKm", 0)
            to_dist = to_stop.get("distanceFromSourceKm", 0)
            calculated_distance = round(to_dist - from_dist, 1) if to_dist > from_dist else t.get("distanceKm")

            simplified.append({
                "train_number": t.get("trainNumber"),
                "train_name": t.get("trainName"),
                "departure_time": _minutes_to_hhmm(from_stop.get("departureMinutes", 0)),
                "arrival_time": _minutes_to_hhmm(to_stop.get("arrivalMinutes", 0)),
                "travel_time_minutes": calculated_duration,
                "distance_km": calculated_distance,
                "running_days": days_list,
                "runs_all_days": len(days_list) == 7,
                "train_type": t.get("type"),
            })

        if limit:
            simplified = simplified[:limit]

        return {
            "from_station": from_station,
            "to_station": to_station,
            "total_trains": len(simplified),
            "trains": simplified,
        }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": "No trains found for this route.", "error_code": "NO_TRAINS_FOUND"}
        return {"error": f"RailRadar API error: {e.response.status_code}", "error_code": "API_ERROR"}
    except httpx.TimeoutException:
        return {"error": "RailRadar API timed out.", "error_code": "API_TIMEOUT"}
    except Exception as e:
        return {"error": str(e), "error_code": "UNKNOWN_ERROR"}


# ---------------------------------------------------------------------------
# Tool 3 — Train Schedule
# ---------------------------------------------------------------------------

@tool
def get_train_schedule(train_number: str, journey_date: str, limit_stops: bool = True) -> dict:
    """
    Get the full stop-by-stop schedule of a train for a specific date.
    """
    try:
        response = httpx.get(
            f"{settings.RAILRADAR_BASE_URL}/trains/{train_number}/schedule",
            params={"journeyDate": journey_date},
            headers=_headers(),
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = _unwrap(response.json())

        stops = []
        for stop in data.get("route", []):
            stops.append({
                "sequence": stop.get("sequence"),
                "station_code": stop.get("stationCode"),
                "station_name": stop.get("stationName"),
                "arrival": _minutes_to_hhmm(stop.get("scheduledArrivalMinutes", 0)),
                "departure": _minutes_to_hhmm(stop.get("scheduledDepartureMinutes", 0)),
                "halt_minutes": stop.get("haltDurationMinutes", 0),
                "day": stop.get("day", 1),
                "distance_km": stop.get("distanceFromSourceKm"),
                "platform": stop.get("platform"),
            })

        train_info = data.get("train", {})
        
        if limit_stops and len(stops) > 10:
            stops = stops[:5] + [{"sequence": "...", "station_code": "...", "station_name": f"... {len(stops)-10} stops omitted ...", "arrival": "", "departure": "", "halt_minutes": 0, "day": 0, "distance_km": 0, "platform": ""}] + stops[-5:]

        return {
            "train_number": train_number,
            "train_name": train_info.get("name"),
            "journey_date": journey_date,
            "total_stops": len(stops),
            "stops": stops,
        }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Schedule for train {train_number} not found.", "error_code": "TRAIN_NOT_FOUND"}
        return {"error": f"RailRadar API error: {e.response.status_code}", "error_code": "API_ERROR"}
    except httpx.TimeoutException:
        return {"error": "RailRadar API timed out.", "error_code": "API_TIMEOUT"}
    except Exception as e:
        return {"error": str(e), "error_code": "UNKNOWN_ERROR"}


# ---------------------------------------------------------------------------
# Tool 4 — Live Station Board
# ---------------------------------------------------------------------------

@tool
def get_live_station_board(station_code: str, hours: int = 4, to_station: str = None, limit: int = 10) -> dict:
    """
    Get live arrivals and departures at a station.
    """
    try:
        params = {"hours": hours}
        if to_station:
            params["toStationCode"] = to_station

        response = httpx.get(
            f"{settings.RAILRADAR_BASE_URL}/stations/{station_code}/live",
            params=params,
            headers=_headers(),
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        # Station board usually isn't wrapped in data, but unwrap just in case
        data = _unwrap(response.json())

        trains = data.get("trains", [])

        simplified = []
        for event in trains:
            train_info = event.get("train", {})
            schedule = event.get("schedule", {})
            simplified.append({
                "train_number": train_info.get("number"),
                "train_name": train_info.get("name"),
                "event_type": "ARRIVAL/DEPARTURE", 
                "scheduled_time": _minutes_to_hhmm(schedule.get("scheduledDepartureMinutes", 0)),
                "expected_time": _minutes_to_hhmm(schedule.get("actualDepartureMinutes", 0)),
                "delay_minutes": schedule.get("delayDepartureMinutes", 0),
                "platform": event.get("platform"),
                "status": "AT_STATION",
                "destination": train_info.get("destinationStationCode"),
                "source": train_info.get("sourceStationCode"),
            })

        if limit:
            simplified = simplified[:limit]

        return {
            "station_code": station_code,
            "hours_window": hours,
            "total_events": len(simplified),
            "events": simplified,
        }

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Station {station_code} not found.", "error_code": "STATION_NOT_FOUND"}
        return {"error": f"RailRadar API error: {e.response.status_code}", "error_code": "API_ERROR"}
    except httpx.TimeoutException:
        return {"error": "RailRadar API timed out.", "error_code": "API_TIMEOUT"}
    except Exception as e:
        return {"error": str(e), "error_code": "UNKNOWN_ERROR"}


# ---------------------------------------------------------------------------
# Tool 5 — Station Lookup
# ---------------------------------------------------------------------------

@tool
def lookup_station_by_name(query: str) -> dict:
    """
    Find station codes by searching for station names or city names in the local database.
    Use this when the user provides a station name (e.g., 'Surat') instead of a code (e.g., 'ST').
    """
    try:
        import json
        import os
        
        # Load the station list
        # Path is relative to the root of the backend folder
        json_path = os.path.join(os.getcwd(), "railwayStationsList.json")
        if not os.path.exists(json_path):
             return {"error": "Station list database not found.", "error_code": "DB_NOT_FOUND"}
             
        with open(json_path, "r") as f:
            data = json.load(f)
            
        stations = data.get("stations", [])
        query_lower = query.lower().strip()
        
        # Search for matches in name, city, or code
        exact_matches = []
        partial_matches = []
        
        for s in stations:
            name = s.get("stnName", "").lower()
            city = s.get("stnCity", "").lower()
            code = s.get("stnCode", "").lower()
            
            # Check for exact matches
            if query_lower == name or query_lower == code or query_lower == city:
                exact_matches.append(s)
            # Check for partial matches
            elif query_lower in name or query_lower in city:
                partial_matches.append(s)
                
        # Sort partial matches to put "Junctions" first
        partial_matches.sort(key=lambda x: "junction" not in x.get("stnName", "").lower())
        
        # Combine matches
        matches = exact_matches + partial_matches
        
        # Remove duplicates (if any)
        seen_codes = set()
        unique_matches = []
        for m in matches:
            if m["stnCode"] not in seen_codes:
                unique_matches.append(m)
                seen_codes.add(m["stnCode"])
                
        # Limit matches to top 15 to avoid token overhead
        final_matches = unique_matches[:15]
        
        if not final_matches:
            return {"error": f"No stations found matching '{query}'.", "error_code": "NO_STATION_FOUND"}
            
        return {
            "query": query,
            "total_matches": len(unique_matches),
            "stations": final_matches,
            "note": "Please prefer 'Junction' stations for major cities unless specified otherwise."
        }
        
    except Exception as e:
        return {"error": str(e), "error_code": "UNKNOWN_ERROR"}


# ---------------------------------------------------------------------------
# Export all tools as a list for agent use
# ---------------------------------------------------------------------------

railradar_tools = [
    get_live_train_status,
    search_trains_between,
    get_train_schedule,
    get_live_station_board,
    lookup_station_by_name,
]