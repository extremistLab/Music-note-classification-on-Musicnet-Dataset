import streamlit as st
import numpy as np
from scipy.io import wavfile
import pandas as pd
from pydub import AudioSegment
from scipy.io import wavfile
import wave
import struct
import os
from scipy.io.wavfile import write
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import shutil
def inst_recg (file, audio, sample_rate):
    extracted_features_df=pd.read_csv("metadata/ensemble100.csv")
    y=np.array(extracted_features_df['class'].tolist())
    labelencoder=LabelEncoder()
    y=to_categorical(labelencoder.fit_transform(y))
    model=keras.models.load_model('saved_models/ensemble95.hdf5')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict_classes(mfccs_scaled_features)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    st.write(prediction_class[0])

def comp_recg (file, audio, sample_rate):
    extracted_features_df=pd.read_csv("metadata/composer100.csv")
    y=np.array(extracted_features_df['class'].tolist())
    labelencoder=LabelEncoder()
    y=to_categorical(labelencoder.fit_transform(y))
    model=keras.models.load_model('saved_models/composer_name_acc80.hdf5')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict_classes(mfccs_scaled_features)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    st.write(prediction_class[0])
    

def get_splitdata(path):
    sampFreq, sound = wavfile.read(path)
    data=(sound)
    ch = np.where(sound[:-1]*sound[1:]<0)[0]
    ch = ch[sound[ch]<0]
    data=list(ch)
    ndt=[]
    for i in data:
        a=i/44100
        a=("%.2f" % a)
        try:
            if ndt[-1] != a:
                ndt.append(a)
        except:
            ndt.append(a)
    blank=[]
    second_s=[0.0]
    second_e=[]
    for i in ndt:
        blank.append(int(float(i)*1000))
        second_s.append(float(i))
        second_e.append(float(i))
    second_s=second_s[:-1]
    df = pd.DataFrame(list(zip(second_s,second_e ,blank)),columns =['Seconds Start', 'Seconds end','id'])
    df['file_name'] = df['id'].astype(str) + '.wav'
    return df

def audio_split(path,df):
    Audio = AudioSegment.from_wav(path)
    samplerate, data = wavfile.read(path)
    for ind in df.index:
        t1=df['Seconds Start'][ind]
        t2=df['Seconds end'][ind]
        t1 = int(t1 * 1000) 
        t2 = int(t2 * 1000)
        newAudio = Audio[t1:t2]
        name="new/"+df['file_name'][ind]
        newAudio.export(name, format="wav")

def get_file(path):
    samplerate, data = wavfile.read(path)
    data = data/ 1.414
    data = data * 32767
    write("temp2/example.wav", samplerate, data.astype(np.int16))
    #audio_split("temp2/example.wav")
    path="temp2/example.wav"
    return path

def note_detect(audio_file):
    file_length=audio_file.getnframes() 
    f_s=audio_file.getframerate()
    sound = np.zeros(file_length) 
    for i in range(file_length) : 
        wdata=audio_file.readframes(1)
        data=struct.unpack("<h",wdata)
        sound[i] = int(data[0])
    sound=np.divide(sound,float(2**15))
    counter = audio_file.getnchannels() 
    fourier = np.fft.fft(sound)
    fourier = np.absolute(fourier)
    imax=np.argmax(fourier[0:int(file_length/2)]) 
    i_begin = -1
    threshold = 0.3 * fourier[imax]
    for i in range (0,imax+100):
        if fourier[i] >= threshold:
            if(i_begin==-1):
                i_begin = i                
        if(i_begin!=-1 and fourier[i]<threshold):
            break
    i_end = i
    imax = np.argmax(fourier[0:i_end+100])
    freq=(imax*f_s)/(file_length*counter)
    note=0
    name = np.array(["C0","C#0","D0","D#0","E0","F0","F#0","G0","G#0","A0","A#0","B0","C1","C#1","D1","D#1","E1","F1","F#1","G1","G#1","A1","A#1","B1","C2","C#2","D2","D#2","E2","F2","F#2","G2","G2#","A2","A2#","B2","C3","C3#","D3","D3#","E3","F3","F3#","G3","G3#","A3","A3#","B3","C4","C4#","D4","D4#","E4","F4","F4#","G4","G4#","A4","A4#","B4","C5","C5#","D5","D5#","E5","F5","F5#","G5","G5#","A5","A5#","B5","C6","C6#","D6","D6#","E6","F6","F6#","G6","G6#","A6","A6#","B6","C7","C7#","D7","D7#","E7","F7","F7#","G7","G7#","A7","A7#","B7","C8","C8#","D8","D8#","E8","F8","F8#","G8","G8#","A8","A8#","B8","Beyond B8"])
    frequencies = np.array([16.35,17.32,18.35,19.45,20.60,21.83,23.12,24.50,25.96    ,27.50    ,29.14    ,30.87    ,32.70    ,34.65    ,36.71    ,38.89    ,41.20    ,43.65    ,46.25    ,49.00    ,51.91    ,55.00    ,58.27    ,61.74    ,65.41    ,69.30    ,73.42    ,77.78    ,82.41    ,87.31    ,92.50    ,98.00    ,103.83    ,110.00    ,116.54    ,123.47    ,130.81    ,138.59    ,146.83    ,155.56    ,164.81    ,174.61    ,185.00    ,196.00    ,207.65    ,220.00    ,233.08    ,246.94    ,261.63    ,277.18    ,293.66    ,311.13    ,329.63    ,349.23    ,369.99    ,392.00    ,415.30    ,440.00    ,466.16    ,493.88    ,523.25    ,554.37    ,587.33    ,622.25    ,659.26    ,698.46    ,739.99    ,783.99    ,830.61    ,880.00    ,932.33    ,987.77    ,1046.50    ,1108.73    ,1174.66    ,1244.51    ,1318.51    ,1396.91    ,1479.98    ,1567.98    ,1661.22    ,1760.00    ,1864.66    ,1975.53    ,2093.00    ,2217.46    ,2349.32    ,2489.02    ,2637.02    ,2793.83    ,2959.96    ,3135.96    ,3322.44    ,3520.00    ,3729.31    ,3951.07    ,4186.01    ,4434.92    ,4698.64    ,4978.03    ,5274.04    ,5587.65    ,5919.91    ,6271.93    ,6644.88    ,7040.00    ,7458.62    ,7902.13,8000])
    for i in range(0,frequencies.size-1):
            if(freq<frequencies[0]):
                note=name[0]
                break
            if(freq>frequencies[-1]):
                note=name[-1]
                break
            if freq>=frequencies[i] and frequencies[i+1]>=freq :
                if freq-frequencies[i]<(frequencies[i+1]-frequencies[i])/2 :
                    note=name[i]
                else :
                    note=name[i+1]
                break

        
    return note

try:
    os.mkdir("new")
except:
    dq=os.listdir("new")
    if len(dq) != 0:
        for i in dq:
            pth="new/"+i
            try:
                os.remove(pth)
            except:
                pass
try:
    os.mkdir("temp2")
except:
    dq=os.listdir("temp2")
    if len(dq) != 0:
        for i in dq:
            pth="temp2/"+i
            os.remove(pth)
try:
    os.mkdir("temp")
except:
    dq=os.listdir("temp")
    if len(dq) != 0:
        for i in dq:
            pth="temp/"+i
            os.remove(pth)
try:
    os.mkdir("copy")
except:
    dq=os.listdir("copy")
    if len(dq) != 0:
        for i in dq:
            pth="copy/"+i
            os.remove(pth)

file = st.sidebar.file_uploader("Please Upload wav Audio File Here",type=["wav"])
if file is None:  
    st.text("Please upload a wav file")
else:
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    inst_recg (file, audio, sample_rate)
    comp_recg (file, audio, sample_rate)
    with open(os.path.join("copy","data.wav"),"wb") as f: 
      f.write(file.getbuffer())
      f.close()
    path=get_file("copy/data.wav")
    df=get_splitdata(path)
    audio_split(path,df)
    note=[]
    for ind in df.index:
        path="new/"+df['file_name'][ind]
        audio_file = wave.open(path)
        Detected_Note = note_detect(audio_file)
        note.append(Detected_Note)
    df['note'] = note
    name = ["C0","C#0","D0","D#0","E0","F0","F#0","G0","G#0","A0","A#0","B0","C1","C#1","D1","D#1","E1","F1","F#1","G1","G#1","A1","A#1","B1","C2","C#2","D2","D#2","E2","F2","F#2","G2","G2#","A2","A2#","B2","C3","C3#","D3","D3#","E3","F3","F3#","G3","G3#","A3","A3#","B3","C4","C4#","D4","D4#","E4","F4","F4#","G4","G4#","A4","A4#","B4","C5","C5#","D5","D5#","E5","F5","F5#","G5","G5#","A5","A5#","B5","C6","C6#","D6","D6#","E6","F6","F6#","G6","G6#","A6","A6#","B6","C7","C7#","D7","D7#","E7","F7","F7#","G7","G7#","A7","A7#","B7","C8","C8#","D8","D8#","E8","F8","F8#","G8","G8#","A8","A8#","B8","Beyond B8"]
    frequencies = [16.35,17.32,18.35,19.45,20.60,21.83,23.12,24.50,25.96    ,27.50    ,29.14    ,30.87    ,32.70    ,34.65    ,36.71    ,38.89    ,41.20    ,43.65    ,46.25    ,49.00    ,51.91    ,55.00    ,58.27    ,61.74    ,65.41    ,69.30    ,73.42    ,77.78    ,82.41    ,87.31    ,92.50    ,98.00    ,103.83    ,110.00    ,116.54    ,123.47    ,130.81    ,138.59    ,146.83    ,155.56    ,164.81    ,174.61    ,185.00    ,196.00    ,207.65    ,220.00    ,233.08    ,246.94    ,261.63    ,277.18    ,293.66    ,311.13    ,329.63    ,349.23    ,369.99    ,392.00    ,415.30    ,440.00    ,466.16    ,493.88    ,523.25    ,554.37    ,587.33    ,622.25    ,659.26    ,698.46    ,739.99    ,783.99    ,830.61    ,880.00    ,932.33    ,987.77    ,1046.50    ,1108.73    ,1174.66    ,1244.51    ,1318.51    ,1396.91    ,1479.98    ,1567.98    ,1661.22    ,1760.00    ,1864.66    ,1975.53    ,2093.00    ,2217.46    ,2349.32    ,2489.02    ,2637.02    ,2793.83    ,2959.96    ,3135.96    ,3322.44    ,3520.00    ,3729.31    ,3951.07    ,4186.01    ,4434.92    ,4698.64    ,4978.03    ,5274.04    ,5587.65    ,5919.91    ,6271.93    ,6644.88    ,7040.00    ,7458.62    ,7902.13,8000]
    df['Frequency']=0.0
    for i in range (len(name)):
        df.loc[(df.note == name[i]),'Frequency']=frequencies[i]
    df.drop(['id','file_name'], axis=1,inplace=True)
    df.to_csv("temp/data.csv",index=False)
    tempdf=df
    tempdf.drop(['Frequency'], axis=1,inplace=True)
    st.table(tempdf)

