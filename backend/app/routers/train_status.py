"""
routers/train_status.py
GET /api/train-status?train_number=12951

Calls the RailRadar tool directly — bypasses the supervisor/graph
for this dedicated endpoint. Useful for direct feature testing.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import TrainStatusRequest, TrainStatusResponse, ErrorResponse
from app.tools.railradar_tool import get_live_train_status

router = APIRouter()


@router.get(
    "/train-status",
    summary="Get live running status of a train",
    response_model=TrainStatusResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def train_status(params: TrainStatusRequest = Depends()):
    result = get_live_train_status.invoke({
        "train_number": params.train_number,
        "journey_date": params.journey_date,
        "include_full_route": True
    })

    # Handle tool errors
    if "error" in result:
        status_code = 404 if result.get("error_code") == "TRAIN_NOT_FOUND" else 500
        raise HTTPException(status_code=status_code, detail=result["error"])

    return TrainStatusResponse(
        success=True,
        message="OK",
        trainNumber=result.get("trainNumber", params.train_number),
        trainName=result.get("trainName", "Unknown"),
        journeyDate=result.get("journeyDate"),
        lastUpdatedAt=result.get("lastUpdatedAt"),
        currentLocation=result.get("currentLocation"),
        route=result.get("route", []),
        natural_language_summary=(
            f"{result.get('trainName', 'Unknown')} (Train {result.get('trainNumber', params.train_number)}) live status fetched."
        ),
    )