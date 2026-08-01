from django.core.management.base import BaseCommand
from api.market.src.loadData import DataCollector

class Command(BaseCommand):
    help = 'Coleta dados com o DataCollector e insere no banco'

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE('Iniciando coleta de dados.'))

        collector = DataCollector()

        collector.collect_all_data()

        self.stdout.write(self.style.SUCCESS('Coleta de dados finalizada.'))
