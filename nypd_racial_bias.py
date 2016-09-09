import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import networkx as nx
import csv 
import sys
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score



dir_path = '/Users/bobby/Documents/Notes/DS/A2/Data/Frisk/'
cd $dir_path

df = pd.read_csv('2014c.csv', header=0, dtype={'crimsusp':object, 'dob':object, 'datestop':object, 'ht_feet':np.int64, 'ht_inch':np.int64})

# arrests made
df.arstmade.value_counts()

# arrests made percentage
df.arstmade.value_counts() / len(df) * 100

def vc(att):
    return df[att].value_counts()

def vp(att):
    return df[att].value_counts() / len(df) * 100

# do analysis attr-wise, brute force approach. maybe group-by is a better approach
def arst_dist1(att):
    '''print the arrests perc grouped by the values in att'''
    for att_val in df[att].unique():
        df_att_val = df[ df[att] == att_val ]
        print 'Arrests percent for {0}={1}'.format(att, att_val)
        print df_att_val.arstmade.value_counts() / len(df_att_val) * 100

def arst_dist2(att):
    '''print the att dist groupbed by arrests'''
    arst_yes = df[df.arstmade == "Y"]
    arst_no = df[df.arstmade == "N"]
    print 'Dist for arrested people:'
    print arst_yes[att].value_counts() / len(arst_yes) * 100
    print 'Dist for NOT arrested people:'
    print arst_no[att].value_counts() / len(arst_no) * 100

# model fitting
data = df_all[features]
data=data.convert_objects(convert_numeric=True)

# clean race
data.race.replace('U','X', inplace=True)

for d in df_arr:
    d.race.replace('U','X', inplace=True)

# remove all rec with age > 100
data = data[~(data.age>100)]

# remove all ht < 60cm
data = data[~(data.ht_cm < 60.0)]
# remove all ht > 272cm -> tallest man ever
data = data[~(data.ht_cm > 272.0)]

# remove all 0 wt rec
data = data[~(data.weight == 0.0)]
data = data[~(data.weight > 1400.0)]

# remove data with sex=N
data = data[~(data.sex == 'N')]

# clean eye color
data=data[~(data.eyecolor == ' ')]
data.eyecolor.replace('UN', 'XX', inplace=True)
data.eyecolor.replace('P', 'PK', inplace=True)
data.eyecolor.replace('Z', 'ZZ', inplace=True)
data.eyecolor.replace('MC', 'MA', inplace=True)

# clean build
data.build.replace('ZZ', 'Z', inplace=True)
data=data[~(data.build == 'MA')]
data=data[~(data.build == 'PK')]
data=data[~(data.build == 'XX')]

# clean hair color
data[data.haircolr == 'Z']['haircolr'] = 'ZZ'

# clean oth pers
data.othpers.replace(' ', 'N', inplace=True)
data.othpers.replace('Y',1,inplace=True)
data.othpers.replace('N',0,inplace=True)

# clean inout by discarding blanks
data=data[~(data.inout == ' ')]
df.inout.replace('I',1,inplace=True)
df.inout.replace('O',0,inplace=True)


# convert categorial fields into bool fields
df_transformed = pd.get_dummies(df, columns=['sex', 'race', 'haircolr', 'eyecolor', 'build', 'age', 'ht_cm', 'weight'])

data_train, data_test, labels_train, labels_test = train_test_split(
df_transformed, labels, test_size=0.33, random_state=1121)

nb_model = GaussianNB()
nb_model.fit(data_train, labels_train)

expected = labels_test
predicted = nb_model.predict(data_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

et_score = cross_val_score(et, data_train, labels_train, n_jobs=-1).mean()


# plotting num of people over time
plot_data = [ Scatter(x=range(2003,2014), y=[len(d) for d in df_arr]) ]
layout = Layout(
    title='Number of people checked over the years',
    xaxis=XAxis( title='Time', showline=True,),
    yaxis=YAxis( title='Number of people', showline=True,)
)
fig = Figure(data=plot_data, layout=layout)
plot_url = py.plot(fig, filename='basic-line')

# plotting race dist as stacked bar graph
x=range(2003,2014)
race_y = []

for d in df_arr:
    y = {r:0 for r in race}
    v = d.race.value_counts() * 100.0 / len(d)
    for r,c in v.iteritems():
        y[r] = c
    race_y.append(y)

trace_arr = []
for r in race:
    y=[d[r] for d in race_y]
    race_trace = Bar(x = x, y = y, name = race_dict[r])
    trace_arr.append(race_trace)

plot_data = Data(trace_arr)
layout = Layout(
    barmode='stack',
    title='Race distribution by percentage over time',
    xaxis=XAxis( title='Year'),
    yaxis=YAxis( title='Percentage')
)
fig = Figure(data=plot_data, layout=layout)
plot_url = py.plot(fig, filename='stacked-bar')

# plotting grouped bar graph for comparing all-arrested dist
x = [race_dict[r] for r in race]

# for all data
v = df_all.race.value_counts() * 100.0 / len(df_all)
y_all = [v[r] for r in race]

trace_all = Bar(x = x, y = y_all, name = 'All')

# for arrested data
d = df_all[df_all.arstmade == 'Y']
v = d.race.value_counts() * 100.0 / len(d)
y_arrested = [v[r] for r in race]

trace_arrested = Bar(x = x, y = y_arrested, name = 'Arrested')

plot_data = Data([trace_all, trace_arrested])
layout = Layout(
    barmode='group',
    title='Race distribution of all checked people vs arrested people',
    xaxis=XAxis( title='Race'),
    yaxis=YAxis( title='Percentage')
)
fig = Figure(data=plot_data, layout=layout)
plot_url = py.plot(fig, filename='grouped-bar')

# pie chart
fig = {
    'data': [{'labels': x,
              'values': y_all,
              'type': 'pie'}],
    'layout': {'title': 'Distribution of people frisked/searched by race'}
}

url = py.plot(fig, validate=False, filename='Pie Chart Example')

#--------------------------------------------------------------------------------
# test code



useful_cols_current = df_arr[0].columns.tolist()
useful_cols_current = useful_cols_current[:-1]

df_all = pd.DataFrame(columns=useful_cols_current)

for df in df_arr:
    df_all = df_all.append(df)


df['arstmade'].replace('N', '0', inplace=True)
df['arstmade'].replace('Y', '1', inplace=True)
df['arstmade'] = df['arstmade'].astype(int)

all_reasons = sreasons + freasons + areasons + creasons
for reason in all_reasons:
        df[reason].replace('N', '0', inplace=True)
        df[reason].replace('Y', '1', inplace=True)
df[all_reasons] = df[all_reasons].astype(int)

reason_counts = {}
reason_perc = {}
for reason in all_reasons:
    reason_counts[reason] = df[reason].sum()
    reason_perc[reason] = reason_counts[reason] * 100.0 / len(df)


df_arrested = df[df.arstmade == 1]
reason_counts_arrested = {}
reason_perc_arrested = {}
for reason in all_reasons:
    reason_counts_arrested[reason] = df_arrested[reason].sum()
    reason_perc_arrested[reason] = reason_counts_arrested[reason] * 100.0 / len(df_arrested)


# plotting grouped bar graph for comparing reasons of all-arrested 
x = creasons

# for all data
y_all = [reason_perc[reason] for reason in creasons ]
trace_all = Bar(x = x, y = y_all, name = 'All')

# for arrested data
y_arrested = [reason_perc_arrested[reason] for reason in creasons ]
trace_arrested = Bar(x = x, y = y_arrested, name = 'Arrested')

plot_data = Data([trace_all, trace_arrested])
layout = Layout(
    barmode='group',
    title='Search reason percentages in all people vs arrested people',
    xaxis=XAxis( title='Reason'),
    yaxis=YAxis( title='Percentage')
)
fig = Figure(data=plot_data, layout=layout)
plot_url = py.plot(fig, filename='search_reasons')


for idx,row in df.iterrows():
    if row.searched == 'N':
        c = 0
        for att in creasons:
            c += row[att]
        if c > 0:
           df.loc[idx, 'searched'] = 'Y'

df_all = df
data = df
df_allc = df


#---------------------------------------------------------------------------------

# discretize 
df['age'] = pd.cut(df.age, bins=5, labels=['age'+str(i) for i in range(1,6)])
df['ht_cm'] = pd.cut(df.ht_cm, bins=5, labels=['ht'+str(i) for i in range(1,6)])
df['weight'] = pd.cut(df.weight, bins=[10, 80, 160, 240, 1400], labels=['wt'+str(i) for i in range(1,5)])

total = 0
wrongs = 0
additional = 0
for idx,row in df.iterrows():
    if row.searched == 'N' and row.frisked == 'N':
        total += 1
        c = 0
        for att in sreasons:
            if row[att] == 'Y':
                c += 1
        if c > 0:
            wrongs += 1
        c = 0
        for att in areasons:
            if row[att] == 'Y':
                c += 1
        if c > 0:
            additional += 1

print total
print wrongs
print additional


def del_invalid_rows(filename):
    '''for now deleting all rows where height fields cant be converted to int'''
    del_rows = 0
    csv_file = open(filename, 'rb')
    csv_reader = csv.reader(csv_file)

    header = csv_reader.next()
    pos = {}
    for i in range(len(header)):
        pos[header[i]] = i

    new_filename = filename[:filename.index('.csv')] + 'c.csv'
    with open(new_filename, 'wb') as csv_op_file:
        csv_writer = csv.writer(csv_op_file, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        row_num = 0

        csv_writer.writerow(header)

        for row in csv_reader:
            row_num += 1
            try:
                int(row[pos['ht_feet']])
                int(row[pos['ht_inch']])
                csv_writer.writerow(row)
            except:
                del_rows += 1
    csv_file.close()
    return del_rows


def clean(df):
    '''given a data frame, clean it nicely according to frisk data'''

    for col in useful_cols:
        df[col]

    # take only the useful columns
    df = df[useful_cols]

    # assume blank arstmade field as no
    df.arstmade.replace(' ', 'N', inplace=True)
    df.arstmade.replace('', 'N', inplace=True)

    # pistol
    df.pistol.replace(' ', 'N', inplace=True)
    df.pistol.replace('', 'N', inplace=True)
    df.pistol.replace('1', 'Y', inplace=True)

    df = clean_flags(df)

    df = clean_date(df)

    df = clean_height(df)
    return df

    

def clean_flags(df):
    # make reasons correct
    for reason in sreasons + freasons + areasons:
        df[reason].replace(' ', 'N', inplace=True)
        df[reason].replace('1', 'Y', inplace=True)

    # if searched/frisked is not set, but some search/frisk flag is, then set the flag to true
    for idx,row in df.iterrows():
        if row.searched == 'N':
            c = 0
            for att in creasons:
                if row[att] == 'Y':
                    c += 1
            if c > 0:
               df.set_value(idx, 'searched', 'Y')
        if row.frisked == 'N':
            c = 0
            for att in freasons:
                if row[att] == 'Y':
                    c += 1
            if c > 0:
               df.set_value(idx, 'frisked', 'Y')

    return df

def clean_height(df):
    # coverting ht from ft to inches
    df['ht_cm'] = (df.ht_feet * 12 + df.ht_inch) * 2.54
    df.drop('ht_feet', axis=1, inplace=True)
    df.drop('ht_inch', axis=1, inplace=True)
    cols=df.columns.tolist()
    cols.insert(cols.index('age') + 1, 'ht_cm')
    df = df[cols[:-1]]
    return df

def clean_date(df):
    # making date correct
    for idx,row in df.iterrows():
        date = row['datestop'].replace(' ','')
        date = '0' + date if len(date) == 7 else date
        date = date[4:] + '-' + date[0:2] + '-' + date[2:4]
        df.set_value(idx, 'datestop', date)

    return df
        

# preprocess data
df_arr = []
for i in range(2003,2014):
    try:
        filename = str(i) + '.csv'
        del_rows = del_invalid_rows(filename)
        print 'deleted rows for {0}:{1}'.format(i, del_rows)
        new_filename = str(i) + 'c.csv'
        df = pd.read_csv(new_filename, header=0, dtype={'crimsusp':object, 'dob':object, 'datestop':object, 'ht_feet':np.int64, 'ht_inch':np.int64})
        df = clean(df)
        df.to_csv(dir_path + new_filename)
        df_arr.append(df)
    except:
        print "Unexpected error in File:",str(i), sys.exc_info()[0]
        raise


# read preprocessed data
df_arr = []
for i in range(2003,2014):
    try:
        filename = str(i) + 'c.csv'
        df = pd.read_csv(filename, header=0, dtype={'crimsusp':object, 'dob':object, 'datestop':object, 'ht_feet':np.int64, 'ht_inch':np.int64})
        df_arr.append(df)
    except:
        print "Unexpected error in File:",str(i), sys.exc_info()[0]
        raise


# making date correct in file
csv_file = open('2014.csv', 'rb')
csv_reader = csv.reader(csv_file)
with open('2014_date.csv', 'wb') as csv_op_file:
    csv_writer = csv.writer(csv_op_file, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_ALL)
    # skip the header line and write it as is
    header = csv_reader.next()
    csv_writer.writerow(header)

    for row in csv_reader:
        date = row[3].replace(' ','')
        date = '0' + date if len(date) == 7 else date
        date = date[4:] + '-' + date[0:2] + '-' + date[2:4]
        row[3] = date
        csv_writer.writerow(row)
csv_file.close()


# write df_arr to file
for i in range(2003,2014):
    filename = str(i) + 'c.csv'
    df_arr[i-2003].to_csv(dir_path + filename)

# manual cleaning
def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

# Open up the csv file in to a Python object
csv_file = open('AllNetworkData.csv', 'r')

with open('AllNetworkData5Columns.csv', 'wb') as csv_op_file:
    for line in csv_file:
        trunc_line = line[0:find_nth(line, ',', 5)]
        csv_op_file.write(trunc_line)
        csv_op_file.write('\n')

csv_file.close()


# race 
race = ['B', 'Q', 'W', 'P', 'Z', 'A', 'I', 'X']
race_dict = {
'A':'ASIAN/PACIFIC ISLANDER',
'B':'BLACK',
'I':'AMERICAN INDIAN/ALASKAN NATIVE',
'P':'BLACK-HISPANIC',
'Q':'WHITE-HISPANIC',
'W':'WHITE',
'X':'UNKNOWN',
'Z':'OTHER'
}

# potentially useful columns. use only this set of columns
useful_cols = ['year',
 'pct',
 'datestop',
 'timestop',
 'inout',
 'trhsloc',
 'perobs',
 'crimsusp',
 'perstop',
 'typeofid',
 'explnstp',
 'othpers',
 'arstmade',
 'arstoffn',
 'sumissue',
 'sumoffen',
 'compyear',
 'comppct',
 'offunif',
 'officrid',
 'frisked',
 'searched',
 'contrabn',
 'adtlrept',
 'pistol',
 'riflshot',
 'asltweap',
 'knifcuti',
 'machgun',
 'othrweap',
 'pf_hands',
 'pf_wall',
 'pf_grnd',
 'pf_drwep',
 'pf_ptwep',
 'pf_baton',
 'pf_hcuff',
 'pf_pepsp',
 'pf_other',
 'radio',
 'ac_rept',
 'ac_inves',
 'rf_vcrim',
 'rf_othsw',
 'ac_proxm',
 'rf_attir',
 'cs_objcs',
 'cs_descr',
 'cs_casng',
 'cs_lkout',
 'rf_vcact',
 'cs_cloth',
 'cs_drgtr',
 'ac_evasv',
 'ac_assoc',
 'cs_furtv',
 'rf_rfcmp',
 'ac_cgdir',
 'rf_verbl',
 'cs_vcrim',
 'cs_bulge',
 'cs_other',
 'ac_incid',
 'ac_time',
 'rf_knowl',
 'ac_stsnd',
 'ac_other',
 'sb_hdobj',
 'sb_outln',
 'sb_admis',
 'sb_other',
 'repcmd',
 'revcmd',
 'rf_furt',
 'rf_bulg',
 'offverb',
 'offshld',
 'sex',
 'race',
 'dob',
 'age',
 'ht_feet',
 'ht_inch',
 'weight',
 'haircolr',
 'eyecolor',
 'build',
 'city',
 'sector',
 'xcoord',
 'ycoord']


useful_cols_current = ['year',
 'pct',
 'datestop',
 'timestop',
 'inout',
 'trhsloc',
 'perobs',
 'crimsusp',
 'perstop',
 'typeofid',
 'explnstp',
 'othpers',
 'arstmade',
 'arstoffn',
 'sumissue',
 'sumoffen',
 'compyear',
 'comppct',
 'offunif',
 'officrid',
 'frisked',
 'searched',
 'contrabn',
 'adtlrept',
 'pistol',
 'riflshot',
 'asltweap',
 'knifcuti',
 'machgun',
 'othrweap',
 'pf_hands',
 'pf_wall',
 'pf_grnd',
 'pf_drwep',
 'pf_ptwep',
 'pf_baton',
 'pf_hcuff',
 'pf_pepsp',
 'pf_other',
 'radio',
 'ac_rept',
 'ac_inves',
 'rf_vcrim',
 'rf_othsw',
 'ac_proxm',
 'rf_attir',
 'cs_objcs',
 'cs_descr',
 'cs_casng',
 'cs_lkout',
 'rf_vcact',
 'cs_cloth',
 'cs_drgtr',
 'ac_evasv',
 'ac_assoc',
 'cs_furtv',
 'rf_rfcmp',
 'ac_cgdir',
 'rf_verbl',
 'cs_vcrim',
 'cs_bulge',
 'cs_other',
 'ac_incid',
 'ac_time',
 'rf_knowl',
 'ac_stsnd',
 'ac_other',
 'sb_hdobj',
 'sb_outln',
 'sb_admis',
 'sb_other',
 'repcmd',
 'revcmd',
 'rf_furt',
 'rf_bulg',
 'offverb',
 'offshld',
 'sex',
 'race',
 'dob',
 'age',
 'ht_cm',
 'weight',
 'haircolr',
 'eyecolor',
 'build',
 'city',
 'sector',
 'xcoord',
 'ycoord']

 features=[
 'sex',
 'race',
 'age',
 'ht_cm',
 'weight',
 'haircolr',
 'eyecolor',
 'build',
 'cs_bulge',
 'cs_casng',
 'cs_cloth',
 'cs_descr',
 'cs_drgtr',
 'cs_furtv',
 'cs_lkout',
 'cs_objcs',
 'cs_other',
 'cs_vcrim',
 'othpers',
 'inout',
 'arstmade'
 ]

 # reasons. sreason is stop reason
sreasons='''
cs_bulge
cs_casng
cs_cloth
cs_descr
cs_drgtr
cs_furtv
cs_lkout
cs_objcs
cs_other
cs_vcrim
'''
sreasons = sreasons.split()

freasons='''
rf_attir
rf_bulg
rf_furt
rf_knowl
rf_othsw
rf_rfcmp
rf_vcact
rf_vcrim
rf_verbl
'''
freasons = freasons.split()

areasons='''
ac_rept
ac_inves
ac_assoc
ac_cgdir
ac_evasv
ac_incid
ac_other
ac_proxm
ac_stsnd
ac_time
'''
areasons = areasons.split()

# cause of search
creasons='''
sb_admis
sb_hdobj
sb_other
sb_outln
'''
creasons = creasons.split()
