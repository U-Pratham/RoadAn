"""
RoadSight AI - Launch Web Server
"""

import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print("=" * 60)
    print(f"🚀 Starting RoadSight AI Web Server on {host}:{port}")
    print(f"🌐 Dashboard URL: http://{host}:{port}")
    print("=" * 60)
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
