

from fastapi import (
    FastAPI,
    Request,
    status,
    Response,
)
from fastapi.exceptions import RequestValidationError
import time
from dog import Dog

app = FastAPI()
dog = Dog()

@app.get("/status/connected")
async def get_battery_status():
    isConnected = dog.check_connection()
    return {"isConnected": isConnected}

@app.get("/status/battery")
async def get_battery_status():
    dog.get_soc()
    return {"battery_soc": dog.get_soc()}

@app.push("/connect")
async def connect(ip_address: str):
    dog = Dog(ip_address=ip_address)
    
    for _ in range(3):
        time.sleep(1)
        if dog.check_connection():
            return Response(
                    status_code=status.HTTP_200_OK,
                    content={
                        "success": True,
                        "message": f"{ip_address} connected",
                    }
                )

    return Response(
        status_code=status.HTTP_408_REQUEST_TIMEOUT,
        content={
            "success": False,
            "message": 'Timeout connecting to dog',
        }
    )