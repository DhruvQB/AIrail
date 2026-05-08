"""
routers/pnr_status.py
GET /api/pnr-status?pnr_number=1234567890&session_id=...
"""
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import PNRRequest, PNRResponse, PassengerStatus, ErrorResponse
from app.tools.pnr_tool import get_pnr_status
 
router = APIRouter()
 
@router.get(
    "/pnr-status",
    summary="Get current status of a PNR booking",
    response_model=PNRResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def pnr_status(params: PNRRequest = Depends()):
    result = get_pnr_status.invoke({"pnr": params.pnr_number})
    
    if "error" in result:
        status_code = 404 if result.get("error_code") in ["INVALID_PNR", "PNR_FLUSHED"] else 500
        raise HTTPException(status_code=status_code, detail=result["error"])
        
    chart_status_str = result.get("chartStatus", "Not Prepared")
    class_type = result.get("journeyClass", "Unknown")
    chart_prepared = "not" not in chart_status_str.lower()
    
    passengers = []
    for p in result.get("passengerList", []):
        current_status = p.get("currentStatusDetails", "")
        booking_status = p.get("bookingStatusDetails", "")
        coach = p.get("currentCoachId") or None
        berth = str(p.get("currentBerthNo")) if p.get("currentBerthNo") else None
        
        if not coach and "CNF" in current_status.upper():
            parts = current_status.replace("-", "/").split("/")
            if len(parts) >= 3:
                coach = parts[1]
                berth = parts[2]
            elif len(parts) >= 2:
                coach = parts[0] if parts[0] != "CNF" else None
                berth = parts[1]
                
        passengers.append(PassengerStatus(
            passenger_index=int(p.get("passengerSerialNumber", 0) or 0),
            booking_status=booking_status,
            current_status=current_status,
            coach=coach,
            berth=berth
        ))
        
    train_name = result.get("trainName", "Unknown")
    
    return PNRResponse(
        success=True,
        message="OK",
        pnr_number=result.get("pnrNumber", params.pnr_number),
        train_number=result.get("trainNumber", "Unknown"),
        train_name=train_name,
        journey_date=result.get("dateOfJourney", "Unknown"),
        source_station=result.get("sourceStation", "Unknown"),
        destination_station=result.get("destinationStation", "Unknown"),
        class_type=class_type,
        chart_prepared=chart_prepared,
        passengers=passengers,
        natural_language_summary=f"PNR {params.pnr_number} status fetched successfully for {train_name}."
    )