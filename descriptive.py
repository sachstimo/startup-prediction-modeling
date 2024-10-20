import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

path = 'data cannot be provided due to sensitivity'
df = pd.read_excel(path, sheet_name='Dataset', index_col=0)

#Describing the datasample in general

print(df.describe())
info = df.describe()
print(df.info())
print(type(info))
print(info)
info.to_excel('Datasample_Info.xlsx','Sheet 1')

#Defining different slices of the sample grouped by regions
#Grouping entire sample based on year of investment and calculating the average

byy_mean = df.groupby('year_inv').mean()
byy_std = df.groupby('year_inv').std()
byy_sum = df.groupby('year_inv').sum()
by_hub = df.groupby('wayra_hub').mean()
by_hub = by_hub.sort_values('tvpi_EUR', ascending= False)
by_region = df.pivot_table('stake_ini_pct', index = 'year_inv' , columns = 'hub_region', aggfunc = [np.mean, np.median])
byy_reg = df.groupby([df['date_agreement'].dt.year, 'startup_region']).mean()
byy_reg_mean = df.pivot_table('total_invamt_EUR', index = 'year_inv', columns = 'startup_region')


by_sector = df.groupby('sector').mean()
by_sector.to_excel('by_sector.xlsx')

by_status = df.pivot_table(columns = df['status'])
by_status.to_excel('by_status.xlsx')


#Plotting TVPI distribution for active and acquired startups
df_trim = df.copy()
df_trim = df[df['is_failed'] == 0]
g1 = sns.histplot(df_trim['tvpi_EUR'], bins = 40, kde = True, alpha = 0.5, label = 'TVPI [EUR]')
g1.annotate('N = ' + str(len(df_trim['tvpi_EUR']) + 1), xy = (50, 100))
g1.set_xticks([0, 1, 5, 10, 20, 30, 40, 50, 60])
g1.set_xticklabels(['0x', '1x', '5x', '10x', '20x', '30x', '40x', '50x', '60x'], rotation = 90)
g1.set_xlabel('TVPI [EUR]')
g1.set_ylabel('Number of Observations')
g1.set_title('TVPI distribution for active and acquired startups')


#Plotting the capital gain contributions

cutoff = 2018
df_y = df[df['year_inv'] <= cutoff]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex= True)
df['pct_totalcapgain_EUR'] = df['capitalgain_EUR'] / df['capitalgain_EUR'].sum()
df = df.sort_values('pct_totalcapgain_EUR', ascending = False)
df['cumpct_capgain'] = df['pct_totalcapgain_EUR'].cumsum()
df = df.reset_index()
print(df[['pct_totalcapgain_EUR', 'cumpct_capgain']].head(20))

df_y['pct_totalcapgain_EUR'] = df_y['capitalgain_EUR'] / df_y['capitalgain_EUR'].sum()
df_y = df_y.sort_values('pct_totalcapgain_EUR', ascending = False)
df_y['cumpct_capgain'] = df_y['pct_totalcapgain_EUR'].cumsum()
df_y = df_y.reset_index()
print(df[['pct_totalcapgain_EUR', 'cumpct_capgain']].head(20))

xticks_rel = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 0.75, 1]

ax1.plot(df.index / (len(df.index) + 1), df['cumpct_capgain'], color = 'black', label = 'overall')
ax1.plot(df_y.index / (len(df_y.index) + 1), df_y['cumpct_capgain'], color = 'blue', ls = '--', label = ('until ' + str(cutoff)), alpha = 0.8)
ax1.set_xticks(xticks_rel)
ax1.set_xticklabels(xticks_rel, rotation = 90)
ax1.set_ylabel('Cum. capital gain [%]')
ax1.set_title('Cumulative capital gain distribution in pct. of invested startups | f(x)')
ax1.legend()
ax1.grid()

ax2.plot(df.index / (len(df.index) + 1), df['pct_totalcapgain_EUR'], color = 'black')
ax2.plot(df_y.index / (len(df_y.index) + 1), df_y['pct_totalcapgain_EUR'], color = 'blue', ls = '--', alpha = 0.8)
ax2.set_xticks(xticks_rel)
ax2.set_xticklabels(xticks_rel, rotation = 90)
ax2.set_xlabel('Invested startups [in %]')
ax2.set_ylabel('Marginal capital gain [%]')
ax2.set_title("Marginal capital gain contribution in pct. of invested startups | f'(x)")
ax2.grid()


#Plotting the proceeds distributions

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)

df['pct_totalproceeds_EUR'] = df['proceeds_EUR'] / df['proceeds_EUR'].sum()
df = df.sort_values('proceeds_EUR', ascending = False)
df['cumpct_proceeds'] = df['pct_totalproceeds_EUR'].cumsum()
df = df.reset_index()
print(df[['pct_totalproceeds_EUR', 'cumpct_proceeds']].head(20))

df_y['pct_totalproceeds_EUR'] = df_y['proceeds_EUR'] / df_y['proceeds_EUR'].sum()
df_y = df_y.sort_values('proceeds_EUR', ascending = False)
df_y['cumpct_proceeds'] = df_y['pct_totalproceeds_EUR'].cumsum()
df_y = df_y.reset_index()

ax1.plot(df.index / (len(df.index) + 1), df['cumpct_proceeds'], color = 'black', label = 'overall')
ax1.plot(df_y.index / (len(df_y.index) + 1), df_y['cumpct_proceeds'], color = 'red', ls = '--', label = ('until ' + str(cutoff)), alpha = 0.8)
ax1.set_xticks(xticks_rel)
ax1.set_xticklabels(xticks_rel, rotation = 90)
ax1.set_ylabel('Cum. proceeds [%]')
ax1.legend()
ax1.set_title('Cumulative proceeds distribution in pct. of invested startups | f(x)')
ax1.grid()

ax2.plot(df.index / (len(df.index) + 1), df['pct_totalproceeds_EUR'], color = 'black')
ax2.plot(df_y.index / (len(df_y.index) + 1), df_y['pct_totalproceeds_EUR'], color = 'red', ls = '--', alpha = 0.8)
ax2.set_xticks(xticks_rel)
ax2.set_xticklabels(xticks_rel, rotation = 90)
ax2.set_xlabel('Invested startups [in %]')
ax2.set_ylabel('Marginal proceeds [%]')
ax2.set_title("Marginal proceeds contribution in pct. of invested startups | f'(x)")
ax2.grid()

#Plotting three maturity correlation plots separately

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fs = 15
sns.regplot(ax = ax1, fit_reg = False, x = df['maturity'], y = df['tvpi_EUR'])
ax1.set_yscale('log')
ax1.set_xlabel('Maturity', fontsize = fs)
ax1.set_ylabel('TVPI [EUR] (log-scaled)', fontsize = fs)
ax1.set_title('Maturity vs. TVPI [EUR] (log-scaled)', fontsize = fs)

sns.regplot(ax = ax2, x = df['maturity'], y = df['initial_invamt_EUR'])
ax2.set_xlabel('Maturity', fontsize = fs)
ax2.set_ylabel('Initial investment amount [in EUR]', fontsize = fs)
ax2.set_title('Maturity vs. Initial investment amount', fontsize = fs)

sns.regplot(ax = ax3, x = df['maturity'], y = df['ttexit'])
ax3.set_xlabel('Maturity', fontsize = fs)
ax3.set_ylabel('Time-to-exit [in years]', fontsize = fs)
ax3.set_title('Maturity vs. Time-to-exit', fontsize = fs)



#Printing the relation of IRR and TVPI to excel with rounded numbers

df['ttexit_r'] = round(df['ttexit'])
df['tvpi_EUR_r'] = round(df['tvpi_EUR'])
piv1 = df.pivot_table('irr_EUR', index = df['ttexit_r'], columns = df['tvpi_EUR_r'])
piv1.to_excel('PyInfo.xlsx', 'Sheet2')


#Plotting the IRR distribution plot

fig, ax = plt.subplots()
g1 = sns.histplot(df['irr_EUR'], bins = 40, kde = True)
g1.annotate('N = ' + str(len(df['irr_EUR']) - df['irr_EUR'].isna().sum()), xy = (5, 10))
g1.set(xlabel = 'IRR [EUR]', ylabel = 'Number of observations', title = 'Distribution of IRRs')

plt.show()