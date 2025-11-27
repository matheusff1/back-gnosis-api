from wsgiref import headers
import yfinance as yf
import pandas as pd
import requests
from decimal import Decimal
import os
import django
from django.conf import settings
from projetob.settings import TWELVE_DATA_API_KEY 
from fredapi import Fred
import io
from datetime import datetime, timedelta

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "projetob.settings")
django.setup()

from api.models import MarketData  

API_KEY = TWELVE_DATA_API_KEY  
FRED_KEY = '33585380c3c80ecd1fac65d1ad38a6a6'


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
BACEN_IDS = [1178, 4391,4189,11]


PERIOD_YF = 'max'
INTERVAL_YF = '1d'
INTERVAL_TD = '1day'
PERIOD_TD = 5000
URL = 'https://api.twelvedata.com/time_series'
TODAY = pd.Timestamp.now().normalize()


##ATENÇÃO: Rever as funções de coleta e atualização dos dados do BACEN, estão explicitas no código e precisam ser mais modulares e limpas.

class DataCollector:
    def __init__(self, symbols_yf=SYMBOLS_YF_ATT, symbols_td=SYMBOLS_TD, symbols_fred=SYMBOLS_FRED, url=URL, api_key=API_KEY, fred_key=FRED_KEY,
                 bacen_ids=BACEN_IDS,
                 today_date=None, interval_yf=INTERVAL_YF, period_yf=PERIOD_YF,
                 interval_td=INTERVAL_TD, period_td=PERIOD_TD):
        self.symbols_yf = symbols_yf
        self.symbols_td = symbols_td
        self.symbols_fred = symbols_fred
        self.api_key = api_key
        self.period_yf = period_yf
        self.interval_yf = interval_yf
        self.interval_td = interval_td
        self.period_td = period_td
        self.bacen_ids = bacen_ids
        self.url = url
        self.today_date = today_date if today_date is not None else pd.Timestamp.now().normalize()
        self.fred = Fred(api_key=fred_key)

    def collect_yfinance_data(self):
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


                    registros = []
                    for _, row in data.iterrows():
                        try:
                            registros.append(MarketData(
                                                    date=row['Date'].date(),
                                                    open=Decimal(str(row['Open'])),
                                                    high=Decimal(str(row['High'])),
                                                    low=Decimal(str(row['Low'])),
                                                    close=Decimal(str(row['Close'])),
                                                    volume=int(row['Volume']),
                                                    symbol=row['Symbol']
                                                ))
                        except Exception as e:
                            print(f"Erro ao preparar registro de {symbol} em {row['Date']}: {e}")

                    MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                    print(f"{len(registros)} registros salvos para {symbol}.")
                else:
                    print(f"Dados vazios para {symbol}.")
            except Exception as e:
                print(f"Erro ao coletar {symbol}: {e}")

    def collect_cds_data(self):
        #REVER FUNÇÃO, NÃO ESTÁ COLETANDO O TOTAL DOS DADOS
        try:
            cds_br_df = pd.read_csv(
            r'C:\Users\Usuario\Documents\Coisas\projeto-b-backup\server\api\src\data-base\cds_brasil_5y.csv'
            )
            cds_usa_df = pd.read_csv(
                r'C:\Users\Usuario\Documents\Coisas\projeto-b-backup\server\api\src\data-base\cds_usa_5y.csv'
            )

            cds_br_df = cds_br_df.rename(columns={
                'Date': 'Date',
                'Price': 'Close',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Change %': 'Var%'
            })
            cds_br_df['Symbol'] = 'CDS_Brasil_5Y'

            cds_usa_df = cds_usa_df.rename(columns={
                'Date': 'Date',
                'Price': 'Close',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Change %': 'Var%'
            })
            cds_usa_df['Symbol'] = 'CDS_USA_5Y'


            cds_br_df['Date'] = pd.to_datetime(cds_br_df['Date'], format="%m/%d/%Y")

            cds_usa_df['Date'] = pd.to_datetime(cds_usa_df['Date'], format="%m/%d/%Y")


            cds_br_df = cds_br_df[['Date', 'Open', 'High', 'Low', 'Close', 'Symbol']]
            cds_usa_df = cds_usa_df[['Date', 'Open', 'High', 'Low', 'Close', 'Symbol']]

            cds_df = pd.concat([cds_br_df, cds_usa_df], ignore_index=True)

            if not cds_df.empty:
                cds_df['Date'] = pd.to_datetime(cds_df['Date'], dayfirst=True)
                
                registros = []
                for _, row in cds_df.iterrows():
                    try:
                        registros.append(MarketData(
                            date=row['Date'].date(),
                            open=Decimal(str(row['Open'])),
                            high=Decimal(str(row['High'])),
                            low=Decimal(str(row['Low'])),
                            close=Decimal(str(row['Close'])),
                            volume=0,  
                            symbol=row['Symbol']
                        ))
                    except Exception as e:
                        print(f"Erro ao preparar registro de {row['Symbol']} em {row['Date']}: {e}")
                if registros:
                    MarketData.objects.bulk_create(registros)
                    print(f"{len(registros)} registros de CDS salvos no banco.")
        except Exception as e:
            print(f"Erro ao coletar dados de CDS: {e}")

    def collect_twelvedata_data(self):
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

                registros = []
                for _, row in df.iterrows():
                    try:
                        registros.append(MarketData(
                            date=pd.to_datetime(row['datetime']).date(),
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            volume=int(float(row['volume'])),
                            symbol=row['symbol']
                        ))
                    except Exception as e:
                        print(f"Erro ao preparar registro de {symbol} em {row.get('datetime')}: {e}")

                MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                print(f"{len(registros)} registros salvos para {symbol}.")
            else:
                print(f"Erro ao buscar dados para {symbol}: {data.get('message', 'Sem mensagem de erro')}")

    def collect_bacen_data(self):
        hoje = datetime.today()

        fim1 = hoje
        inicio1 = fim1 - timedelta(days=365*10)  

        fim2 = inicio1 - timedelta(days=1)  
        inicio2 = fim2 - timedelta(days=365*10)  

        def baixar_periodo(inicio, fim, serie, symbol):
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

        df1_swap = baixar_periodo(inicio1, fim1, 1178, "Swap_DI_5Y")
        df2_swap = baixar_periodo(inicio2, fim2, 1178, "Swap_DI_5Y")
        df_swap = pd.concat([df2_swap, df1_swap], ignore_index=True)

        df1_selic = baixar_periodo(inicio1, fim1, 4391, "Selic_Over")
        df2_selic = baixar_periodo(inicio2, fim2, 4391, "Selic_Over")
        df_selic = pd.concat([df2_selic, df1_selic], ignore_index=True)

        df1_selic_long = baixar_periodo(inicio1, fim1, 4189, "Selic_Over_Long")
        df2_selic_long = baixar_periodo(inicio2, fim2, 4189, "Selic_Over_Long")
        df_selic_long = pd.concat([df2_selic_long, df1_selic_long], ignore_index=True)

        df1_cdi = baixar_periodo(inicio1, fim1, 11, "CDI")
        df2_cdi = baixar_periodo(inicio2, fim2, 11, "CDI")
        df_cdi = pd.concat([df2_cdi, df1_cdi], ignore_index=True)

        df_final = pd.concat([df_swap, df_selic, df_selic_long, df_cdi], ignore_index=True)
        df_final = df_final.sort_values('Date').reset_index(drop=True)

        registros = []
        for _, row in df_final.iterrows():
            try:
                registros.append(MarketData(
                    date=row['Date'].date(),
                    close=row['Close'],
                    symbol=row['Symbol'],
                    open=0,
                    high=0,
                    volume=0, 
                    low=0
                ))
            except Exception as e:
                print(f"Erro ao preparar registro de {row['Symbol']} em {row['Date']}: {e}")

        if registros:
            MarketData.objects.bulk_create(registros, ignore_conflicts=True)
            print(f"{len(registros)} registros salvos Bacen.")
        else:
            print("Nenhum registro para salvar Bacen.")

    def update_yfinance_data(self):
        today = pd.Timestamp(self.today_date)
        print(f"Data de atualização: {today}")
        
        for symbol in self.symbols_yf:
            print(f'Atualizando dados de {symbol} via yFinance...')

            try:
                ult = MarketData.objects.filter(symbol=symbol).order_by('-date').first()
                if ult:
                    start_date = pd.Timestamp(ult.date) + pd.Timedelta(days=1)
                else:
                    start_date = pd.Timestamp('2000-01-01')

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

                    registros = []
                    for _, row in data.iterrows():
                        try:
                            registros.append(MarketData(
                                date=row['Date'].date(),
                                open=row['Open'],
                                high=row['High'],
                                low=row['Low'],
                                close=row['Close'],
                                volume=int(row['Volume']),
                                symbol=row['Symbol']
                            ))
                        except Exception as e:
                            print(f"Erro ao preparar registro: {e}")

                    MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                    print(f"{len(registros)} registros atualizados para {symbol}.")

                else:
                    print(f"Nenhum dado novo para {symbol}.")

            except Exception as e:
                print(f"Erro ao atualizar {symbol}: {e}")

    def collect_fred_data(self):
        all_data = []

        fred = Fred(api_key=FRED_KEY)

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

        registros = []
        for _, row in final_df.iterrows():
            try:
                registros.append(MarketData(
                    date=row['Date'].date(),
                    close=row['Close'],
                    symbol=row['Symbol'],
                    open=0,
                    high=0,
                    low=0,
                    volume=0
                ))
            except Exception as e:
                print(f"Erro ao preparar registro de {row['Symbol']} em {row['Date']}: {e}")

        if registros:
            MarketData.objects.bulk_create(registros, ignore_conflicts=True)
            print(f"{len(registros)} registros salvos no banco.")
        else:
            print("Nenhum registro para salvar.")

    def update_twelvedata_data(self):
        today = pd.Timestamp(self.today_date)
        print(f"Data de atualização TwelveData: {today}")
        
        for symbol in self.symbols_td:
            print(f'Atualizando dados de {symbol} via TwelveData...')

            ult = MarketData.objects.filter(symbol=symbol).order_by('-date').first()
            if ult:
                start_date = ult.date + pd.Timedelta(days=1)
            else:
                start_date = pd.to_datetime('2000-01-01')

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

                    registros = []
                    for _, row in df.iterrows():
                        try:
                            registros.append(MarketData(
                                date=pd.to_datetime(row['datetime']).date(),
                                open=row['open'],
                                high=row['high'],
                                low=row['low'],
                                close=row['close'],
                                volume=int(float(row['volume'])),
                                symbol=row['symbol']
                            ))
                        except Exception as e:
                            print(f"Erro ao preparar registro de {symbol} em {row.get('datetime')}: {e}")

                    MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                    print(f"{len(registros)} registros atualizados para {symbol}.")

                else:
                    msg = data.get('message', 'Sem mensagem de erro')
                    print(f"Erro ao buscar dados para {symbol}: {msg}")

            except Exception as e:
                print(f"Erro ao atualizar {symbol}: {e}")

    def update_bacen_data(self):
        today = pd.Timestamp(self.today_date)
        print(f"Data de atualização BACEN: {today}")
        
        series_map = {
            "Swap_DI_5Y": 1178,
            "Selic_Over": 4391,
            "Selic_Over_Long": 4189,
            "CDI": 11
        }

        for symbol, serie in series_map.items():
            print(f'Atualizando dados de {symbol} (SGS {serie})...')

            ult = MarketData.objects.filter(symbol=symbol).order_by('-date').first()
            if ult:
                start_date = pd.Timestamp(ult.date) + pd.Timedelta(days=1)
            else:
                start_date = pd.Timestamp('2000-01-01')

            url = (
                f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?"
                f"formato=csv&dataInicial={start_date.strftime('%d/%m/%Y')}"
                f"&dataFinal={today.strftime('%d/%m/%Y')}"
            )

            headers = {'User-Agent': 'Mozilla/5.0'}

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()

                import io
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
            df['Symbol'] = symbol

            registros = []
            for _, row in df.iterrows():
                try:
                    registros.append(MarketData(
                        date=row['Date'].date(),
                        close=row['Close'],
                        symbol=row['Symbol'],
                        open=0,
                        high=0,
                        low=0,
                        volume=0
                    ))
                except Exception as e:
                    print(f"Erro ao preparar registro de {symbol} em {row['Date']}: {e}")

            MarketData.objects.bulk_create(registros, ignore_conflicts=True)
            print(f"{len(registros)} registros atualizados para {symbol}.")

    def update_fred_data(self):
        for symbol in self.symbols_fred:
            print(f'Atualizando dados de {symbol} do FRED...')

            try:
                ult = MarketData.objects.filter(symbol=symbol).order_by('-date').first()
                if ult:
                    start_date = pd.Timestamp(ult.date) + pd.Timedelta(days=1)
                else:
                    start_date = pd.Timestamp('2000-01-01')

                data = self.fred.get_series(symbol, observation_start=start_date)

                if data.empty:
                    print(f"Nenhum dado novo para {symbol}.")
                    continue

                df = pd.DataFrame({
                    "Date": data.index,
                    "Symbol": symbol,
                    "Close": data.values
                })

                registros = []
                for _, row in df.iterrows():
                    try:
                        registros.append(MarketData(
                            date=row['Date'].date(),
                            close=row['Close'],
                            symbol=row['Symbol'],
                            open=0,
                            high=0,
                            low=0,
                            volume=0
                        ))
                    except Exception as e:
                        print(f"Erro ao preparar registro de {symbol} em {row['Date']}: {e}")

                MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                print(f"{len(registros)} registros atualizados para {symbol}.")

            except Exception as e:
                print(f"Erro ao atualizar {symbol}: {e}")

    def update_cds_data(self):
        try:
            cds_br_df = self.get_last_cds('BR')
            cds_usa_df = self.get_last_cds('US')

            cds_df = pd.concat([cds_br_df, cds_usa_df], ignore_index=True)

            start_date = MarketData.objects.filter(symbol='CDS_Brasil_5Y').order_by('-date').first()
            if start_date:
                start_date = pd.Timestamp(start_date.date) + pd.Timedelta(days=1)
            else:
                print("Nenhum dado anterior encontrado para CDS.")
                return
            
            cds_df = cds_df[cds_df['Date'] >= start_date]

            if not cds_df.empty:
                cds_df['Date'] = pd.to_datetime(cds_df['Date'], dayfirst=True)
                
                registros = []
                for _, row in cds_df.iterrows():
                    try:
                        registros.append(MarketData(
                            date=row['Date'].date(),
                            open=row['Open'],
                            high=row['High'],
                            low=row['Low'],
                            close=row['Close'],
                            volume=0,  
                            symbol=row['Symbol']
                        ))
                    except Exception as e:
                        print(f"Erro ao preparar registro de {row['Symbol']} em {row['Date']}: {e}")
                if registros:
                    MarketData.objects.bulk_create(registros, ignore_conflicts=True)
                    print(f"{len(registros)} registros de CDS atualizados no banco.")
        except Exception as e:
            print(f"Erro ao atualizar dados de CDS: {e}")

    def get_last_cds(self,country):
        try:
            if(country == 'BR'):
                url_final = 'brazil-cds-5-years-usd-historical-data'
                symbol = 'CDS_Brasil_5Y'
            elif(country == 'US'):
                url_final = 'united-states-cds-5-years-usd-historical-data'
                symbol = 'CDS_USA_5Y'

            url = f'https://www.investing.com/rates-bonds/{url_final}'

            headers = {"User-Agent": "Mozilla/5.0"}
    
            resp = requests.get(url, headers=headers)
            dfs = pd.read_html(resp.text)
            df = dfs[0]
            
            df.columns = ["Data", "Último", "Abertura", "Máxima", "Mínima", "Variação %"]

            df = df.rename(columns={
                "Data": "Date",
                "Último": "Close",
                "Abertura": "Open",
                "Máxima": "High",
                "Mínima": "Low"
            })

            df["Symbol"] = symbol
            df = df.drop(columns=["Variação %"], errors='ignore')

            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        except Exception as e:
            print(f"Erro ao buscar último registro de CDS para {country}: {e}")
            return None