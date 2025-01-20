#!/bin/bash
# npx ts-node fakeBackend.ts
# npx ts-node fakeBackend.ts --host 0.0.0.0 --port 3000
uvicorn main:app --host 0.0.0.0 --port 4000