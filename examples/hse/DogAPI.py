

from fastapi import (
    FastAPI,
    Request,
    status,
    Response,
)
import uvicorn
import time
from dog import Dog

app = FastAPI()
dog = Dog()

@app.get("/status/connected")
async def get_battery_status():
    global dog
    isConnected = dog.check_connection()
    return {"isConnected": isConnected}

@app.get("/status/battery")
async def get_battery_status():
    global dog
    return {"battery_soc": dog.get_soc()}

@app.get("/status/current")
async def get_battery_status():
    global dog
    dog.get_current()
    return {"battery_current": dog.get_current()}

@app.get("/status")
async def get_battery_status():
    global dog
    return {
        "battery_current": dog.get_current(),
        "battery_soc": dog.get_soc(),
        "isConnected": dog.check_connection()}


@app.post("/connect")
async def connect(ip_address: str = "192.168.4.199"):
    global dog
    dog = Dog(ip_address=ip_address)
    
    for _ in range(3):
        time.sleep(1)
        if dog.check_connection():
            return Response(status_code=status.HTTP_200_OK)

    return Response(status_code=status.HTTP_408_REQUEST_TIMEOUT)
    

if __name__ == "__main__":
    uvicorn.run("DogAPI:app", host="127.0.0.1", port=8000, reload=True)
