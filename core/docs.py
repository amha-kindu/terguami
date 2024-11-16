from fastapi.openapi.utils import get_openapi
from core.config import settings

def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=f"{settings.PROJECT_NAME}",
        version="1.0.0",
        description=(
            "This API translates English text to Amharic using a transformer model. "
            "Send an English text string to the `/translate` endpoint, and receive the Amharic translation."
        ),
        routes=app.routes,
    )

    openapi_schema["info"]["contact"] = {
        "name": "Amha Kindu",
        "email": "amha.kindu@example.com",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema
