from numpy import * 
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
#Initial data imput
Kp=[]
with open('kpindex.txt', 'r') as archivo:
    for linea in archivo:
        if linea.startswith('#'):
            continue
        tokens = linea.split()
        if len(tokens) == 10:
            YYY, MM, DD, hh_h, hh_m, days, days_m, kp, ap, D = map(float, tokens)
            Kp.append((YYY, MM, DD, hh_h, hh_m, days, days_m, kp, ap, D))
        else:
            print(f"La línea no tiene el formato correcto: {linea}")
Kp=array(Kp)
Kp2017=Kp[248376:251296,-3]
data17 = pd.read_csv("dsc_fc_summed_spectra_2017_v01.csv", \
delimiter = ',', parse_dates=[0], \
infer_datetime_format=True, na_values='0', \
header = None)
data = data17.iloc[:,:4]
data2 = pd.DataFrame(index=range(int(len(data[0][:])/180)), columns=range(4))
for i in range(len(data2[0][:])):
    data2[0][i] = data[0][180*i]
    data2[1][i] = data[1][180*i:180*(i+1)-1].mean()
    data2[2][i] = data[2][180*i:180*(i+1)-1].mean()
    data2[3][i] = data[3][180*i:180*(i+1)-1].mean()
data=data2.assign(kpp=Kp2017)

#Data preprocessing
def cleaning(a):
    data[a] = asarray(data[a]).astype('float32')
    max=0
    min=0
    for i in data[a]:
        if abs(i)>max:
            max=abs(i)
    data[a]=data[a]/max
    return(max)
cleaning(1)
cleaning(2)
cleaning(3)
maxkp=cleaning("kpp")
def moving_average(data, column_name, window_size=5):
    moving_avg_series = data[column_name].rolling(window=window_size+1, min_periods=1).mean()
    return moving_avg_series
data["kpp"]=moving_average(data,"kpp")
data[1]=moving_average(data,1)
data[2]=moving_average(data,2)
data[3]=moving_average(data,3)
def add_offset_column(data, original_column_name,new_name):
    data[new_name] = data[original_column_name].shift(1)
    return data

data=add_offset_column(data,1,"1off1")
data=add_offset_column(data,2,"2off1")
data=add_offset_column(data,3,"3off1")
timpers=12
for i in range(1,timpers):
    data=add_offset_column(data,"1off"+str(i),"1off"+str(i+1))
    data=add_offset_column(data,"2off"+str(i),"2off"+str(i+1))
    data=add_offset_column(data,"3off"+str(i),"3off"+str(i+1))
numoffs=0
for j in range(numoffs):
    for i in data:
        i=add_offset_column(data,i,str(i)+"n")
data=data.iloc[(timpers+numoffs):,:]
print(data)

#Neural network
per_train=80
listcols=[]
for i in range(1,5+3*timpers):
    if i!=4:
        listcols.append(i+(5+3*timpers)*numoffs)

X_train =data.iloc[:int(size(data,0)*per_train/100),listcols]
y_train =data.iloc[:int(size(data,0)*per_train/100),4]
X_val =data.iloc[int(size(data,0)*per_train/100):,listcols]
y_val =data.iloc[int(size(data,0)*per_train/100):,4]

model = keras.Sequential([
    keras.layers.Input(shape=((timpers+1)*3,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear')])
model.compile(optimizer="adam", loss='mean_squared_error')
epochs = 100
batch_size = 256
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

#Model testing
pred=model.predict(X_val)
plt.plot(range(int(size(data,0)*(1-per_train/100))+1),pred*maxkp,label="Model")
plt.plot(range(int(size(data,0)*(1-per_train/100))+1),y_val*maxkp,label="Real data")
plt.title("Comparison of model with real data")
plt.xlabel("Time(3h)")
plt.ylabel("Kp")
plt.legend()
plt.show()