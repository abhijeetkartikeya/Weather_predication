# Running Version 2 in Parallel

This document explains how to securely run `version_2` of the weather prediction system alongside the originally existing system (version 1) without causing any conflicts natively.

## How Isolation is Ensured

1. **Docker Container Names**: `version_2` Docker services are named exclusively (e.g. `weather_postgres_v2`, `weather_grafana_v2`), which ensures Docker Engine treats them as entirely separate entities from existing containers.
2. **Ports Mapping**: 
   - `version_2` `weather_postgres_v2` maps explicitly to host port `5434` (instead of the conflicting `5432`). 
   - `version_2` `weather_grafana_v2` maps to host port `3002` (instead of the originally mapping `3000`).
3. **Internal Network Separation**: `version_2` components communicate with each other utilizing a custom scoped bridge network named `weather_net_v2`. This prevents any overlapping internal network resolutions.
4. **Volumes Separation**: Data for postgres and grafana persists to independent Docker volumes named `postgres_data_v2` and `grafana_data_v2`.
5. **Database Naming**: `version_2` uses a differently named internal PostgreSQL database called `weather_ml_v2` configurable in the `.env` settings.
6. **PM2 Independent Job Instances**: `ecosystem.config.js` sets the background running task to the alias `"weather-scheduler-v2"`, guaranteeing PM2 manages it as a fundamentally unique app.

## Commands for Startup & Verification

Assume your existing project `version_1` is currently already actively running, here are the step-by-step actions for initiating `version_2`:

### 1. Boot up the Docker Infrastructure

Start up the database, grafana, and internal ML microservice components from within the `version_2` directory:
```bash
docker-compose up -d --build
```
*Docker will create the new network, load the distinct volumes, and expose the services on ports `5434` and `3002`!*

### 2. Verify Docker Containers 

Run the following command to check the existing status of containers:
```bash
docker ps
```
You should observe both the older containers (from `version_1`) operating seamlessly alongside newly created containers with postfix `_v2`.

### 3. Initiate the Job Scheduler 

Use PM2 to start the job scheduled worker:
```bash
pm2 start ecosystem.config.js
```

### 4. Verify Active Jobs

Print PM2's currently active process tracking list:
```bash
pm2 list
```
You should now notice `"weather-scheduler-v2"` prominently displayed within your active process collection concurrently to `"weather-scheduler"`.
