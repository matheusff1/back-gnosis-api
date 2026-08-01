from django.core.management.base import BaseCommand
from api.quant.src.predictionModelling import *

class Command(BaseCommand):
    help = 'Executa as predições e salva os resultados no banco de dados'

    def handle(self, *args, **kwargs):
        print("Iniciando o processo de previsões.")
        try:
            #predictions_process()
            pred_process = PredictionProcessor()
            pred_process.predictions_process()
        except Exception as e:
            print(f"Ocorreu um erro ao executar o modelo de previsão: {e}")