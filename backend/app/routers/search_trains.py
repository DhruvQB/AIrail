"""
routers/search_trains.py
GET /api/search-trains?source=ST&destination=BCT&session_id=...

Calls the RailRadar tool directly — bypasses the supervisor/graph
for this dedicated endpoint. Useful for direct feature testing.

Note: RailRadar /trains/between does not accept a date param.
      It returns all trains on the route. The frontend can filter
      by running_days if needed.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import SearchTrainsRequest, SearchTrainsResponse, TrainSummary, ErrorResponse
from app.tools.railradar_tool import search_trains_between

router = APIRouter()


def _minutes_to_duration(minutes: int | None) -> str:
    """Convert travel time in minutes to human readable string like '7h 25m'."""
    if not minutes:
        return "N/A"
    h, m = divmod(minutes, 60)
    return f"{h}h {m}m" if h else f"{m}m"


@router.get(
    "/search-trains",
    summary="Search trains between two stations",
    response_model=SearchTrainsResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def search_trains(params: SearchTrainsRequest = Depends()):
    result = search_trains_between.invoke({
        "from_station": params.source,
        "to_station": params.destination,
        "limit": None
    })

    # Handle tool errors
    if "error" in result:
        status_code = 404 if result.get("error_code") == "NO_TRAINS_FOUND" else 500
        raise HTTPException(status_code=status_code, detail=result["error"])

    trains = [
        TrainSummary(
            train_number=t["train_number"],
            train_name=t["train_name"],
            departure_time=t["departure_time"],
            arrival_time=t["arrival_time"],
            duration=_minutes_to_duration(t.get("travel_time_minutes")),
            days_of_operation=t.get("running_days", []),
            available_classes=[],       # RailRadar /between does not return classes
            distance_km=t.get("distance_km"),
        )
        for t in result.get("trains", [])
    ]

    total = result.get("total_trains", len(trains))

    return SearchTrainsResponse(
        success=True,
        message="OK",
        source=params.source,
        destination=params.destination,
        journey_date=None,
        total_trains_found=total,
        trains=trains,
        natural_language_summary=(
            f"Found {total} train(s) from {params.source} to {params.destination}. "
            + (
                f"Earliest departure is at {trains[0].departure_time}."
                if trains else "No trains found on this route."
            )
        ),
    )