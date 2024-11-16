from fastapi import FastAPI
from core.config import settings
from core.docs import custom_openapi
from endpoints.health import router as health_router
from endpoints.translate import router as translation_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "A FastAPI application for translating English text to Amharic using a transformer model."
    ),
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(translation_router)

# Add custom OpenAPI documentation
app.openapi = lambda: custom_openapi(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
