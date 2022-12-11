import os
import pandas as pd
import matplotlib.pyplot as plt

def symbol_to_path(symbol,base_dir="data"):
	return os.path.join(base_dir,"{}.csv".format(str(symbol)))

def get_data(symbols,dates):
	df = pd.DataFrame(index=dates)
	print df

	if 'SPY' not in symbols:
		symbols.insert(0,'SPY')

	for symbol in symbols:
		tmp = pd.read_csv(symbol_to_path(symbol),index_col="Date",parse_dates=True,usecols=['Date','Close'],na_values=['nan'])
		tmp = tmp.rename(columns={'Close':symbol})

		df = df.join(tmp)
		if symbol == 'SPY':
			df = df.dropna(subset=['SPY'])

	return df

def get_bollinger_bands(rm,rstd):
	return rm + 2*rstd, rm - 2*rstd

