from django.core.management.base import BaseCommand
from api.portfolios.src.portfolioDataUpdating import PortfolioDataUpdater

class Command(BaseCommand):
    help = 'Atualiza os dados de portfólios diariamente'

    def handle(self, *args, **kwargs):
        print("Iniciando atualização diária dos portfólios...")
        portfolio_data_updater = PortfolioDataUpdater()
        result = portfolio_data_updater.update_all_portfolios_tracking_data()
        if not result:
            print("A atualização dos portfólios encontrou um erro crítico e foi interrompida.")

        print("Atualização dos portfólios finalizada.")