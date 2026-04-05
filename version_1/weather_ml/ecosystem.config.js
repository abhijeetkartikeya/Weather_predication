module.exports = {
  apps: [
    {
      name: "weather_forecaster",
      script: "src/scheduler.py",
      interpreter: "/Users/kartikeya/Desktop/Weather_predication/venv312/bin/python3",
      cwd: __dirname,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 10000,
      watch: false,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "logs/error.log",
      out_file: "logs/output.log",
      merge_logs: true,
      env: {
        PG_HOST: "localhost",
        PG_DATABASE: "weatherdb",
        PG_USER: "kartikeya",
        PG_PASSWORD: "",
        INFLUXDB_URL: "http://localhost:8086",
        INFLUXDB_TOKEN: "weather-ml-token-secret",
        INFLUXDB_ORG: "weather-ml",
        INFLUXDB_BUCKET: "weather_forecast",
      },
    },
  ],
};
