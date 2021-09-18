import re
import csv
import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_rf_data():
    #url will be changed according to the nubmer of years
    url='https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=billratesAll'
    mostActiveStocksUrl = url
    page = requests.get(mostActiveStocksUrl)
    data = page.text
    soup = BeautifulSoup(page.content, 'html.parser')
    rows = soup.find_all('table',{'class':'t-chart'})

    df = pd.read_html(str(rows))[0]
    return df

