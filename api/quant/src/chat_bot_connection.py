import requests
from .RiskMeasurements import RiskMeasurements
from api.models import MarketData
import pandas as pd


def get_chat_analysis(prompt= """
                    Você é um assistente financeiro que analisa dados de mercado e retorna insights resumidos e objetivos.
                    Você irá receber dados de mercado e medidas estatísticas, como volatilidade, risco, outros indicadores financeiros e previsões.
                    Você deve gerar uma análise detalhada e objetiva.
                    Dados:
                
                    Por favor, gere um relatório detalhado com:
                    - Principais riscos
                    - Recomendações
                    - Notícias relacionadas ao ativo
                    -Resumo geral e interpretação dos dados"""
                      ,symbol='PETR4.SA'):
    symbol_data = MarketData.objects.filter(symbol=symbol).order_by('date')
    if not symbol_data.exists():
        return {'error': 'No market data found for the given symbol.'}
    df = pd.DataFrame(symbol_data.values('date','close', 'open', 'high', 'low', 'volume'))

    risk_m = RiskMeasurements(df)
    full_analysis = risk_m.full_process()

    prompt = f"{prompt}\n\nDados calculados:\n{full_analysis},\nAtivo:{symbol}"

    url = 'http://localhost:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'model': 'mistral',
        'prompt': prompt,
        'stream': False
    }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data['response']
    else:
        print(f'Erro: {response.status_code} - {response.text}')


