# Tools for PNR status tracking
import time
from langchain_core.tools import tool
from ntes import NTESClient

@tool
def get_pnr_status(pnr: str) -> dict:
    """
    Check the booking status of a 10-digit PNR.
    Automatically handles captcha challenges and returns passenger, train, and chart details.
    """
    client = NTESClient()
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            result = client.pnr_status(pnr)
        except Exception as e:
            return {"error": str(e), "error_code": "API_ERROR"}
            
        if not isinstance(result, dict):
            return {"error": "Invalid API response format", "error_code": "API_ERROR"}
            
        error_msg = result.get("errorMessage")
        if error_msg:
            # Check for captchas
            if "Captcha not matched" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return {"error": "Failed to bypass captcha after multiple attempts.", "error_code": "CAPTCHA_FAILED"}
            
            # Flushed / Invalid
            if "FLUSHED PNR" in error_msg or "NOT YET GENERATED" in error_msg:
                return {"error": "PNR flushed or not yet generated.", "error_code": "PNR_FLUSHED"}
                
            if "PNR No. is not valid" in error_msg:
                return {"error": "Invalid PNR number.", "error_code": "INVALID_PNR"}
                
            # Generic error
            return {"error": error_msg, "error_code": "API_ERROR"}
            
        if "pnrNumber" in result or "trainName" in result or result.get("status") == "successful":
            return result
            
    return {"error": "Unknown error checking PNR.", "error_code": "UNKNOWN_ERROR"}
