import asyncio
import websockets
import time

async def stream_verdicts(websocket):
    print("Phone connected to the pipeline!")
    try:
        while True:
            # TODO: Replace this simulated list with the real output from your AI model
            for state in ["Engaged", "Engaged", "Distracted", "Sleeping"]:
                await websocket.send(state)
                print(f"Sent: {state}")
                await asyncio.sleep(1.5) # Send a new verdict every 1.5 seconds
    except websockets.exceptions.ConnectionClosed:
        print("Phone disconnected.")

async def main():
    # 0.0.0.0 allows any device on your Wi-Fi to connect to port 8765
    async with websockets.serve(stream_verdicts, "0.0.0.0", 8765):
        print("WebSocket Server running on ws://0.0.0.0:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())