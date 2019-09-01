import pandas as pd
import numpy as np
import texttable as tt
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn import preprocessing

class EDA:
   # init
   def __init__(self,df_default='df_train'):
       # feature column names
       self.feature_names = ['Duration(1)','Protocol_Type(2)','Service(3)','Flag(4)','Src_Bytes(5)','Dst_Byte(6)','Land(7)','Wrong_Fragment(8)','Urgent(9)','Hot(10)','Num_Failed_Logins(11)','Logged_In(12)','Num_compromised(13)','Root_Shell(14)','Su_Attempted(15)','Num_Root(16)','Num_File_creations(17)','Num_Shells(18)','Num_Access_Files(19)','Num_Outbound_Cmds(20)','Is_Hot_Login(21)','Is_Guest_Login(22)','Count(23)','Srv_Count(24)','Serror_Rate(25)','Srv_Serror_Rate(26)','Rerror_Rate(27)','Srv_Rerror_Rate(28)','Same_Srv_Rate(29)','Diff_Srv_Rate(30)','Srv_Diff_Host_Rate(31)','Dst_Host_Count(32)','Dst_Host_Srv_Count(33)','Dst_Host_Same_Srv_Rate(34)','Dst_Host_Diff_Srv_Rate(35)','Dst_Host_Same_Src_Port_Rate(36)','Dst_Host_Srv_Diff_Host_Rate(37)','Dst_Host_Serror_Rate(38)','Dst_Host_Srv_Serror_Rate(39)','Dst_Host_Rerror_Rate(40)','Dst_Host_Srv_Rerror_Rate(41)','Attack_Type(42)','Score(43)']
       
       # classes of attacks and normal
       self.dos_types = ["back", "land", "mailbomb", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm", "processtable", "worm"]
       self.probe_types = ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]
       self.r2l_types = ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"]
       self.u2r_types = ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"]
       self.attack_types = self.dos_types + self.probe_types + self.u2r_types + self.r2l_types
       self.normal_types = ["normal"]
       
       # dataframes
       self.all_df = {
           # full datasets
           'df_train' : pd.read_csv("../datasets/KDDTrain+.txt", delimiter=",", names=self.feature_names),
           'df_test' : pd.read_csv("../datasets/KDDTest+.txt", delimiter=",", names=self.feature_names),
           # combined train and test dataset
           'df_main' : pd.concat([
               pd.read_csv("../datasets/KDDTrain+.txt", delimiter=",", names=self.feature_names),
               pd.read_csv("../datasets/KDDTest+.txt", delimiter=",", names=self.feature_names)
           ]),
           # partial datasets
           'df_train20' : pd.read_csv("../datasets/KDDTrain+_20Percent.txt", delimiter=",", names=self.feature_names),
           'df_test21' : pd.read_csv("../datasets/KDDTest-21.txt", delimiter=",", names=self.feature_names),
       },
       self.df_default = self.all_df[0][df_default]
       #subclasses
       self.df_dos = self.df_default[self.df_default['Attack_Type(42)'].isin(self.dos_types)]
       self.df_probe = self.df_default[self.df_default['Attack_Type(42)'].isin(self.probe_types)]
       self.df_u2r = self.df_default[self.df_default['Attack_Type(42)'].isin(self.u2r_types)]
       self.df_r2l = self.df_default[self.df_default['Attack_Type(42)'].isin(self.r2l_types)]
       self.df_normal = self.df_default[self.df_default['Attack_Type(42)'].isin(self.normal_types)]

     
   def __replace_values(self,value):

       if value in self.dos_types:
           return 'DoS'
       elif value in self.probe_types:
           return 'Probe'
       elif value in self.u2r_types:
           return 'U2R'
       elif value in self.r2l_types:
           return 'R2L'
       elif value in self.normal_types:
           return 'Normal'
       else:
           print(value)
           return 'NaN'
       
   
   def set_df_default(self,df_default='df_train'):
       self.df_default = self.all_df[0][df_default]
       self.df_dos = self.df_default[self.df_default['Attack_Type(42)'].isin(self.dos_types)]
       self.df_probe = self.df_default[self.df_default['Attack_Type(42)'].isin(self.probe_types)]
       self.df_u2r = self.df_default[self.df_default['Attack_Type(42)'].isin(self.u2r_types)]
       self.df_r2l = self.df_default[self.df_default['Attack_Type(42)'].isin(self.r2l_types)]
       self.df_normal = self.df_default[self.df_default['Attack_Type(42)'].isin(self.normal_types)]
   
   def feature_ranges(self, df='default'):
       ranges = {}
       if df == 'default':
           df = self.df_default.copy()
       elif df == 'attacks':
           df = self.df_default.copy()
           df = df[df['Attack_Type(42)'].isin(self.attack_types)]
       elif df == 'normal':
           df = self.df_normal.copy()
       elif df == 'dos':
           df = self.df_dos.copy()
       elif df == 'probe':
           df = self.df_probe.copy()
       elif df == 'u2r':
           df = self.df_u2r.copy()
       elif df == 'r2l':
           df = self.df_r2l.copy()
       else:
           print('Invalid df option. Choose a df from one of the following: default, attacks, normal, dos, probe, u2r, r2l')
           return
       
       df.drop(['Protocol_Type(2)','Service(3)','Flag(4)','Attack_Type(42)'],axis=1,inplace=True)
       
       for feature, values in df.items():
           ranges[feature] = []
           ranges[feature].append(df[feature].max())
           ranges[feature].append(df[feature].min())
           ranges[feature].append(len(df[feature].unique()))


       ranges = pd.DataFrame(ranges, index=['Max','Min','Unique'])
       return ranges
   
   
   def graph_feature_ranges(self, df='default'):
       ranges = {}
       if df == 'default':
           df = self.df_default.copy()
       elif df == 'attacks':
           df = self.df_default.copy()
           df = df[df['Attack_Type(42)'].isin(self.attack_types)]
       elif df == 'normal':
           df = self.df_normal.copy()
       elif df == 'dos':
           df = self.df_dos.copy()
       elif df == 'probe':
           df = self.df_probe.copy()
       elif df == 'u2r':
           df = self.df_u2r.copy()
       elif df == 'r2l':
           df = self.df_r2l.copy()
       else:
           print('Invalid df option. Choose a df from one of the following: default, attacks, normal, dos, probe, u2r, r2l')
           return 
       df.drop(['Protocol_Type(2)','Service(3)','Flag(4)','Attack_Type(42)'],axis=1,inplace=True)
       
       columns = df.columns.tolist()
       df_values = df.values 
       min_max_scaler = preprocessing.MinMaxScaler()
       df_values_scaled = min_max_scaler.fit_transform(df_values)
       df = pd.DataFrame(df_values_scaled, columns=columns)
       
       # plotly boxplot
       fig = go.Figure(
           data=[go.Box(y=df[column],name=column, boxmean=True) for column in df.columns]
       )
       # format the layout
       fig.update_layout(
           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
           yaxis=dict(zeroline=False, gridcolor='white')
       )
       fig.show()
       
#         return df
   
       
   def class_distribution(self):
       class_dist = {
           'columns': ['Normal','DoS','Probe','U2R','R2L'],
           'values': [self.df_normal.shape[0],self.df_dos.shape[0],self.df_probe.shape[0],self.df_u2r.shape[0],self.df_r2l.shape[0]],
       }
       
       # plotly donut graph
       fig = go.Figure(data=[go.Pie(labels=class_dist['columns'], values=class_dist['values'], hole=.3)])
       fig.show()
       
#         return pd.DataFrame(class_dist)
   
   
   def corr_matrix(self):
       corr = self.df_default.corr()
       
       # heatmap
       mask = np.zeros_like(corr, dtype=np.bool)
       mask[np.triu_indices_from(mask)] = True
       # Set up the matplotlib figure
       f, ax = plt.subplots(figsize=(11, 9))
       # Generate a custom diverging colormap
       cmap = sns.diverging_palette(220, 10, as_cmap=True)
       sns.heatmap(corr,mask=mask,square=True,cbar_kws={"shrink": .75})
       
#         return corr
   
   def scatter_matrix(self,df='default'):
       if df == 'default':
           df = self.df_default.copy()
       elif df == 'attacks':
           df = self.df_default.copy()
           df = df[df['Attack_Type(42)'].isin(self.attack_types)]
       elif df == 'normal':
           df = self.df_normal.copy()
       elif df == 'dos':
           df = self.df_dos.copy()
       elif df == 'probe':
           df = self.df_probe.copy()
       elif df == 'u2r':
           df = self.df_u2r.copy()
       elif df == 'r2l':
           df = self.df_r2l.copy()
       else:
           print('Invalid df option. Choose a df from one of the following: default, attacks, normal, dos, probe, u2r, r2l')
           return
       
       df['Attack_Type(42)'] = df['Attack_Type(42)'].apply(self.__replace_values)
       
       fig = px.scatter_matrix(df, 
           color="Attack_Type(42)", 
           labels={col:col.replace('_', ' ') for col in df.columns}) # remove underscore
       fig.update_traces(diagonal_visible=False)
       fig.show()
       
#        return df




