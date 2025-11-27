from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from django.core.management import call_command
from pytz import timezone
import traceback

def run_update_market_data_command():
    print(f"\n{'='*70}")
    print(f"[{datetime.now()}] Executando comando update_market_data...")
    print(f"{'='*70}")
    try:
        call_command('update_market_data')
        print(f"[{datetime.now()}]  update_market_data concluído com sucesso!")
    except Exception as e:
        print(f"[{datetime.now()}]  Erro no update_market_data: {e}")
        traceback.print_exc()

def run_predictions_command():
    print(f"\n{'='*70}")
    print(f"[{datetime.now()}] Executando comando run_and_save_predictions...")
    print(f"{'='*70}")
    try:
        call_command('run_and_save_predictions')
        print(f"[{datetime.now()}]  run_and_save_predictions concluído com sucesso!")
    except Exception as e:
        print(f"[{datetime.now()}]  Erro no run_and_save_predictions: {e}")
        traceback.print_exc()

def run_update_portfolios_data():
    print(f"\n{'='*70}")
    print(f"[{datetime.now()}] Atualizando dados dos portfólios...")
    print(f"{'='*70}")
    try:
        call_command('update_portfolios_data')
        print(f"[{datetime.now()}]  update_portfolios_data concluído com sucesso!")
    except Exception as e:
        print(f"[{datetime.now()}]  Erro no update_portfolios_data: {e}")
        traceback.print_exc()

def start():
    scheduler = BackgroundScheduler(timezone=timezone('America/Sao_Paulo'))

    scheduler.add_job(
        run_update_market_data_command, 
        trigger=CronTrigger(hour=22, minute=0),
        misfire_grace_time=240
    )

    scheduler.add_job(
        run_predictions_command, 
        trigger=CronTrigger(hour=22, minute=30),
        misfire_grace_time=120 
    )

    scheduler.add_job(
        run_update_portfolios_data,
        trigger=CronTrigger(hour=22, minute=15),
        misfire_grace_time=120
    )

    print(f"[{datetime.now()}] Scheduler iniciado...")
    scheduler.start()
