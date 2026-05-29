# primary imports
import os
from datetime import datetime
from concurrent import futures

# third-party imports
import yfinance as yf

def download_stock(stock):
    """ try to query yfinance for a stock, if failed note with print """
    try:
        print(stock)
        stock_df = yf.download(stock, start=start_time, end=now_time, progress=True)
        stock_df['Name'] = stock
        os.makedirs('data/raw_data', exist_ok=True)
        output_name = 'data/raw_data/' + stock + '_data.csv'
        print(output_name)
        stock_df.to_csv(output_name)
    except:
        bad_names.append(stock)
        print('bad: %s' % (stock))

if __name__ == '__main__':

    """ set the download window """
    # start_time = datetime(2010, 1, 1)
    now_time = datetime.now()
    start_time = datetime(now_time.year - 15, now_time.month, now_time.day)

    """ list of s_and_p companies """
    s_and_p = ['AAPL', 'NVDA', 'META', 'DAL', 'UAL', 'MRNA', 'BA']
        
    bad_names = []  # to keep track of failed queries

    # set the maximum thread number
    max_workers = 50

    workers = min(max_workers, len(s_and_p))
    with futures.ThreadPoolExecutor(workers) as executor:
        res = executor.map(download_stock, s_and_p)

    """ Save failed queries to a text file to retry """
    if len(bad_names) > 0:
        with open('failed_queries.txt', 'w') as outfile:
            for name in bad_names:
                outfile.write(name + '\n')

    # timing:
    finish_time = datetime.now()
    duration = finish_time - now_time
    minutes, seconds = divmod(duration.seconds, 60)
    print('getSandP_threaded.py')
    print(f'The threaded script took {minutes} minutes and {seconds} seconds to run.')