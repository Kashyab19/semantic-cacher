from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# mocking database of tenants - this will be postgres in real life.
VALID_USERS = {
    "sk-gold": {"tenant_id": "tenant_gold", "plan": "pro"},
    "sk-silver": {"tenant_id": "tenant_silver", "plan": "free"},
}


async def get_current_user(api_key: str = Security(api_key_header)):
    """
    1. Checks if header exists.
    2. Checks if key is valid.
    3. Returns the user profile (tenant_id).
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key header aka x-api-key",
        )

    user = VALID_USERS.get(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )

    return user
