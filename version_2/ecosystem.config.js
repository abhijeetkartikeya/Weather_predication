module.exports = {
  apps: [
    {
      name: 'weather_scheduler',
      script: 'scheduler.py',
      interpreter: 'venv/bin/python',
      args: '',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      log_date_format: 'YYYY-MM-DD HH:mm Z',
      error_file: 'logs/scheduler-error.log',
      out_file: 'logs/scheduler-out.log',
      merge_logs: true,
      env: {
        NODE_ENV: 'production',
      }
    }
  ]
};
