from apscheduler.schedulers.background import BackgroundScheduler

_scheduler = None

def start_scheduler(cfg, agent):
    global _scheduler
    _scheduler = BackgroundScheduler()
    cron = cfg.get("scheduler",{}).get("dreams_cron","0 3 * * *")
    minute, hour = cron.split()[0], cron.split()[1]
    _scheduler.add_job(lambda: agent.dreams.dream(), "cron", minute=minute, hour=hour)
    _scheduler.start()
