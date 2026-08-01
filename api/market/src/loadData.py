from wsgiref import headers
import yfinance as yf
import pandas as pd
import requests
from decimal import Decimal
import os
import django
from django.conf import settings
from gnosis.settings import TWELVE_DATA_API_KEY, FRED_API_KEY 
from fredapi import Fred
import io
from datetime import datetime, timedelta

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gnosis.settings")
django.setup()

from api.models import Asset, MarketData
from api.market.services import AssetCatalogService

API_KEY = TWELVE_DATA_API_KEY  
FRED_KEY = FRED_API_KEY


SYMBOLS_TD = ['PETR4', 'BRL/USD', 'BTC/USD', 'AAPL', 'XAU/USD', 'BVSP', 'GSPC', 'IXIC']
SYMBOLS_YF = [
    'PETR4.SA', 'BRL=X', 'BTC-USD', 'AAPL', '^BVSP', '^GSPC', '^IXIC', 'BZ=F', 'GC=F',
    'VALE3.SA', 'ITUB4.SA', 'B3SA3.SA', 'WEGE3.SA', 'BBAS3.SA', 'ABEV3.SA', 'RENT3.SA',
    'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JNJ', 'DIS',
    'BABA', 'TSM', 'SAP.DE', 'NESN.SW', '7203.T', 'RDSA.AS', 'BP.L', 'TM',
    'CL=F', 'NG=F', 'SI=F', 'ZC=F', 'ZS=F'
]

SYMBOLS_YF_ATT = [
    # Moedas e criptos
    'BRL=X',       # Dólar/Real
    'BTC-USD',     # Bitcoin em USD

    # Índices
    '^BVSP',       # Ibovespa
    '^GSPC',       # S&P 500
    '^CRB',        # Índice de Commodities
    "^TRCCRB",     # Índice de Commodities Thomson Reuters
    "CRBQ",       # Índice de Commodities CRB
    "GSG",        # Índice de Commodities Goldman Sachs
    '^FVX',        # Treasury 5 anos (EUA)

    # Ações brasileiras
    'VALE3.SA',
    'PETR4.SA',
    'ITUB4.SA',
    'BBDC4.SA',
    'ABEV3.SA',
    'WEGE3.SA', 
    'B3SA3.SA', 
    'ITSA4.SA', 
    'CSAN3.SA', 
    'BRFS',

    # Ações americanas
    'AAPL',
    'NVDA',
    'MSFT',
    'AMZN',
    'GOOGL',
    'META', 
    'TSLA', 
    'UNH', 
    'DHR', 
    'SPGI',

    # Commodities
    'CL=F',        # Petróleo WTI
    'BZ=F',        # Petróleo Brent
    'NG=F',        # Gás Natural
    'ZC=F',        # Milho
    'ZS=F',        # Soja
    'GC=F',        # Ouro
    'DX=F',       # Dólar Index

    # Títulos públicos brasileiros (via fundos/ETFs)
    'BTGIMABFIRF.SA',  # Fundo BTG Pactual Tesouro IPCA Geral
    'NTNS11.SA'        # ETF Investo Teva Tesouro IPCA+ 0–4 anos
]

SYMBOLS_FRED = ["M2SL", "NFCI", "TEDRATE"]
BACEN_IDS = [1178,11]

BACEN_SYMBOLS = {"Swap_DI_5Y":1178, "CDI":11}

PERIOD_YF = 'max'
INTERVAL_YF = '1d'
INTERVAL_TD = '1day'
PERIOD_TD = 5000
URL = 'https://api.twelvedata.com/time_series'
TODAY = pd.Timestamp.now().normalize()


##ATENÇÃO: Rever as funções de coleta e atualização dos dados do BACEN, estão explicitas no código e precisam ser mais modulares e limpas.

class DataCollector:
    def __init__(self, symbols_yf=None, symbols_td=None, symbols_fred=None, url=URL, api_key=API_KEY, fred_key=FRED_KEY,
                 bacen_symbols=None, interval_yf=INTERVAL_YF, period_yf=PERIOD_YF,
                 interval_td=INTERVAL_TD, period_td=PERIOD_TD):
        # Por padrão, os ativos a coletar/atualizar vêm do catálogo (AssetSource
        # no banco). Ainda é possível passar listas explícitas (ex.: testes) ou
        # cair nas constantes legadas deste módulo.
        self.symbols_yf = symbols_yf if symbols_yf is not None else AssetCatalogService.source_symbols('yfinance')
        self.symbols_td = symbols_td if symbols_td is not None else AssetCatalogService.source_symbols('twelvedata')
        self.symbols_fred = symbols_fred if symbols_fred is not None else AssetCatalogService.source_symbols('fred')
        self.bacen_symbols = bacen_symbols if bacen_symbols is not None else AssetCatalogService.bacen_series()
        self.api_key = api_key
        self.period_yf = period_yf
        self.interval_yf = interval_yf
        self.interval_td = interval_td
        self.period_td = period_td
        self.url = url
        self.today_date = pd.Timestamp.now().date()
        self.fred = Fred(api_key=fred_key)

    def collect_all_data(self):
        try:
            self._collect_yfinance_data()
            # self._collect_twelvedata_data()
            self._collect_fred_data()
            self._collect_bacen_data()
        except Exception as e:
            print(f"Erro ao coletar dados: {e}")

    def update_all_data(self):
        try:
            self._update_yfinance_data()
            # self._update_twelvedata_data()
            self._update_fred_data()
            self._update_bacen_data()
        except Exception as e:
            print(f"Erro ao atualizar dados: {e}")

    def _collect_yfinance_data(self):
        for symbol in self.symbols_yf:
            try:
                data = yf.download(
                    tickers=symbol,
                    period=self.period_yf,
                    interval=self.interval_yf,
                    progress=True
                )

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

            
                if not data.empty:
                    data.reset_index(inplace=True)
                    data['Symbol'] = symbol
                    data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)


                    ret = self._process_and_store_data(data, symbol)

                    if ret is not None:
                        print(f"{ret} registros salvos para {symbol}.")
                else:
                    print(f"Dados vazios para {symbol}.")
            except Exception as e:
                print(f"Erro ao coletar {symbol}: {e}")

    def _collect_twelvedata_data(self):
        for symbol in self.symbols_td:
            print(f'Coletando dados de {symbol} via TwelveData...')

            params = {
                'symbol': symbol,
                'interval': self.interval_td,
                'outputsize': self.period_td,
                'apikey': self.api_key,
                'format': 'JSON'
            }

            response = requests.get(self.url, params=params)
            data = response.json()

            if data.get('status') == 'ok' and 'values' in data:
                df = pd.DataFrame(data['values'])
                df['symbol'] = symbol
                df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'datetime'], inplace=True)

                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'datetime': 'Date'
                })

                ret = self._process_and_store_data(df, symbol)

                if ret is not None:
                    print(f"{ret} registros salvos para {symbol}.")


            else:
                print(f"Erro ao buscar dados para {symbol}: {data.get('message', 'Sem mensagem de erro')}")


    def _baixar_periodo_bacen(self, inicio, fim, serie, symbol):
            data_inicial = inicio.strftime("%d/%m/%Y")
            data_final = fim.strftime("%d/%m/%Y")
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=csv&dataInicial={data_inicial}&dataFinal={data_final}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)

            if response.status_code != 200 or not response.text.strip():
                print(f"Erro ao baixar série {serie} ({symbol}): resposta vazia ou inválida. Status {response.status_code}")
                return pd.DataFrame(columns=['Date', 'Symbol', 'Close'])

            text = response.text.strip()

            if text.startswith("<") and "html" in text.lower():
                print(f"Erro ao baixar série {serie} ({symbol}): resposta HTML recebida.")
                print(text[:200])
                return pd.DataFrame(columns=['Date', 'Symbol', 'Close'])

            try:
                df = pd.read_csv(io.StringIO(text), sep=';', on_bad_lines='skip')
            except pd.errors.ParserError as e:
                print(f"Erro ao ler CSV da série {serie} ({symbol}): {e}")
                print(text[:200])
                return pd.DataFrame(columns=['Date', 'Symbol', 'Close'])

            df.columns = [c.strip().lower() for c in df.columns]
            if not {'data', 'valor'}.issubset(df.columns):
                print(f"CSV inesperado para série {serie} ({symbol}). Colunas: {df.columns}")
                return pd.DataFrame(columns=['Date', 'Symbol', 'Close'])

            df.rename(columns={'data': 'Date', 'valor': 'Close'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df['Close'] = df['Close'].astype(str).str.replace(',', '.').astype(float)
            df['Symbol'] = symbol
            df = df.dropna(subset=['Date', 'Close'])

            return df[['Date', 'Symbol', 'Close']]


    def _collect_bacen_data(self):
        hoje = datetime.today()

        fim1 = hoje
        inicio1 = fim1 - timedelta(days=365*10)  

        fim2 = inicio1 - timedelta(days=1)  
        inicio2 = fim2 - timedelta(days=365*10) 

        df_geral = pd.DataFrame() 

        for symbol, serie in self.bacen_symbols.items():
            print(f'Coletando dados de {symbol} (SGS {serie}) do BACEN...')
            df1 = self._baixar_periodo_bacen(inicio1, fim1, serie, symbol)
            df2 = self._baixar_periodo_bacen(inicio2, fim2, serie, symbol)

            df_symbol = pd.concat([df2, df1], ignore_index=True)

            df_geral = pd.concat([df_geral, df_symbol], ignore_index=True)

        if df_geral.empty:
            print("Nenhum dado coletado do BACEN.")
            return
        
        df_final = df_geral.drop_duplicates(subset=['Date', 'Symbol'])
        df_final = df_final.sort_values('Date').reset_index(drop=True)

        ret = self._process_and_store_data(df_final, symbol=None)

        if ret is not None:
            print(f"{ret} registros salvos no banco.")


    def _collect_fred_data(self):
        all_data = []

        fred = self.fred

        for symbol in self.symbols_fred:
            print(f'Coletando dados de {symbol} do FRED...')
            try:
                data = fred.get_series(symbol)  
                if data.empty:
                    print(f"Nenhum dado encontrado para {symbol}.")
                    continue

                df = pd.DataFrame({
                    "Date": data.index,
                    "Symbol": symbol,
                    "Close": data.values
                })
                all_data.append(df)

            except Exception as e:
                print(f"Erro ao coletar {symbol}: {e}")

        if not all_data:
            print("Nenhum dado coletado.")
            return

        final_df = pd.concat(all_data, ignore_index=True)
        final_df.dropna(subset=['Close'], inplace=True)

        ret = self._process_and_store_data(final_df, symbol=None)
        if ret is not None:
            print(f"{ret} registros salvos no banco.")

        else:
            print("Nenhum registro para salvar.")


    def _update_twelvedata_data(self):
        today = pd.Timestamp(self.today_date)
        print(f"Data de atualização TwelveData: {today}")
        
        for symbol in self.symbols_td:
            print(f'Atualizando dados de {symbol} via TwelveData...')

            start_date = self._get_symbol_update_range(symbol)

            end_date = today
            if start_date > end_date:
                print(f"Dados de {symbol} já estão atualizados.")
                continue

            params = {
                'symbol': symbol,
                'interval': self.interval_td,
                'apikey': self.api_key,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'format': 'JSON',
                'timezone': 'UTC'
            }

            try:
                response = requests.get(self.url, params=params)
                data = response.json()

                if data.get('status') == 'ok' and 'values' in data:
                    df = pd.DataFrame(data['values'])
                    df['symbol'] = symbol
                    df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'datetime'], inplace=True)
                    df = df.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                        'datetime': 'Date'
                    })

                    ret = self._process_and_store_data(df, symbol)

                    if ret is not None:
                        print(f"{ret} registros atualizados para {symbol}.")

                else:
                    msg = data.get('message', 'Sem mensagem de erro')
                    print(f"Erro ao buscar dados para {symbol}: {msg}")

            except Exception as e:
                print(f"Erro ao atualizar {symbol}: {e}")


    def _update_bacen_data(self):
        today = pd.Timestamp(self.today_date)
        print(f"Data de atualização BACEN: {today}")
        

        for symbol, serie in self.bacen_symbols.items():
            print(f'Atualizando dados de {symbol} (SGS {serie})...')

            start_date = self._get_symbol_update_range(symbol)

            url = (
                f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?"
                f"formato=csv&dataInicial={start_date.strftime('%d/%m/%Y')}"
                f"&dataFinal={today.strftime('%d/%m/%Y')}"
            )

            headers = {'User-Agent': 'Mozilla/5.0'}

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()

                data = pd.read_csv(
                    io.StringIO(response.text),
                    sep=';',
                    decimal=','
                )

            except Exception as e:
                print(f"Erro ao buscar dados para {symbol}: {e}")
                continue

            if data.empty:
                print(f"Nenhum dado novo para {symbol}.")
                continue

            data.rename(columns={'data': 'Date', 'valor': 'Close'}, inplace=True)
            data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
            data['Close'] = data['Close'].astype(str).str.replace(',', '.').astype(float)

            df = data[['Date', 'Close']].copy()

            ret = self._process_and_store_data(df, symbol)

            if ret is not None:
                print(f"{ret} registros atualizados para {symbol}.")
            

    def _update_fred_data(self):
        for symbol in self.symbols_fred:
            print(f'Atualizando dados de {symbol} do FRED...')

            try:
                start_date = self._get_symbol_update_range(symbol)

                data = self.fred.get_series(symbol, observation_start=start_date)

                if data.empty:
                    print(f"Nenhum dado novo para {symbol}.")
                    continue

                df = pd.DataFrame({
                    "Date": data.index,
                    "Close": data.values
                })

                ret = self._process_and_store_data(df, symbol)

                if ret is not None:
                    print(f"{ret} registros atualizados para {symbol}.")

            except Exception as e:
                print(f"Erro ao atualizar {symbol}: {e}")


    def _update_yfinance_data(self):
        today = pd.Timestamp(self.today_date)
        # end_date = today + pd.Timedelta(days=1)
        print(f"Data de atualização: {today}")
        
        for symbol in self.symbols_yf:
            print(f'Atualizando dados de {symbol} via yFinance...')

            try:
                start_date = self._get_symbol_update_range(symbol)

                end_date = today
                if start_date > end_date:
                    print(f"Dados de {symbol} já estão atualizados.")
                    continue

                data = yf.download(
                    tickers=symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=self.interval_yf,
                    progress=False
                )

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                if not data.empty:
                    data.reset_index(inplace=True)
                    data['Symbol'] = symbol
                    data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

                    ret = self._process_and_store_data(data, symbol)

                    if ret is not None:
                        print(f"{ret} registros atualizados para {symbol}.")

                else:
                    print(f"Nenhum dado novo para {symbol}.")

            except Exception as e:
                print(f"Erro ao atualizar {symbol}: {e}")

    def _get_symbol_update_range(self, symbol):
        ult = MarketData.objects.filter(asset__symbol=symbol).order_by('-date').first()
        if ult:
            start_date = pd.Timestamp(ult.date) + pd.Timedelta(days=1)
        else:
            start_date = pd.Timestamp('2000-01-01')
        return start_date


    def _process_data_to_format(self, df, symbol):
        try:
            df = df.copy()
            if symbol:
                df['Symbol'] = symbol
            df = df.dropna(subset=['Date', 'Close'])
            df = df.fillna(0)

            registros = []
            for _, row in df.iterrows():
                try:
                    asset = self._asset_for_symbol(row['Symbol'])
                    if asset is None:
                        print(f"Sem Asset para o símbolo {row['Symbol']}; registro ignorado.")
                        continue
                    registros.append(MarketData(
                        date=row['Date'].date(),
                        open=row['Open'] if 'Open' in row else 0,
                        high=row['High'] if 'High' in row else 0,
                        low=row['Low'] if 'Low' in row else 0,
                        close=row['Close'] if 'Close' in row else 0,
                        volume=int(row['Volume']) if 'Volume' in row else 0,
                        asset=asset,
                    ))
                except Exception as e:
                    print(f"Erro ao preparar registro de {symbol} em {row['Date']}: {e}")

            return registros
        except Exception as e:
            print(f"Erro ao processar dados para {symbol}: {e}")
            return []

    def _asset_for_symbol(self, symbol):
        """Resolve o Asset pelo símbolo (cache em memória)."""
        if not hasattr(self, '_asset_cache'):
            self._asset_cache = {a.symbol: a for a in Asset.objects.all()}
        return self._asset_cache.get(symbol)
    

    def _process_and_store_data(self, df, symbol):
        try:
            registros = self._process_data_to_format(df, symbol)
            MarketData.objects.bulk_create(registros, ignore_conflicts=True)
            print(f"{len(registros)} registros salvos para {symbol}.")
            return len(registros)
        except Exception as e:
            print(f"Erro ao salvar dados para {symbol}: {e}")
            return None