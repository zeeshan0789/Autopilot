from fastapi import FastAPI

app = FastAPI(title="Autopilot Gateway")


@app.get("/health")
def health():
    return {"status": "ok"}
