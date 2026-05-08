"""
routers/availability.py
GET /api/availability?train_number=12951&source=ST&destination=BCT&journey_date=2025-01-15&class_type=3A
"""
from fastapi import APIRouter, Depends
from app.models.schemas import AvailabilityRequest
 
router = APIRouter()
 
 
@router.get("/availability", summary="Check seat availability for a train on a given date")
async def availability(params: AvailabilityRequest = Depends()):
    return {"status": "not implemented"}