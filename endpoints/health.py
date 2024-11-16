from fastapi import APIRouter

router = APIRouter()

@router.get(
    "/health",
    summary="Health Check",
    description="Check the health of the API.",
    tags=["Health"],
)
def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "healthy"}

