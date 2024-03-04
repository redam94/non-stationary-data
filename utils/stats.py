import statsmodels.tsa.api as tsa

def pretty_adf(data, regression='ct'):
  crit, p, *_ = tsa.adfuller(data, regression=regression)
  print(f'ADF Statistic: {crit:.3f}')
  print(f'p-value: {p:.3f}')
  if p < 0.05:
    print("Reject the null hypothesis, the data is stationary")
  else:
    print("Fail to reject the null hypothesis, the data is non-stationary")
  return crit, p

def pretty_kpss(data, regression):
  crit, p, *_ = tsa.kpss(data, regression=regression)
  print(f'KPSS Statistic: {crit:.3f}')
  print(f'p-value: {p:.3f}')
  if p < 0.05:
    print("Reject the null hypothesis, the data is non-stationary")
  else:
    print("Fail to reject the null hypothesis, the data is stationary")
  return crit, p

def pretty_coint(y, x):
  crit, p_val, _ = tsa.coint(y, x)
  print(f'Critical value: {crit:0.3f}, p-value: {p_val:0.3f}')
  if p_val < 0.05:
    print("Reject the null hypothesis, the data is cointegrated")
  else:
    print("Fail to reject the null hypothesis, the data is not cointegrated")
  return crit, p_val