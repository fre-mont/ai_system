# %% [markdown]
# #### Install Dependencies 
import subprocess
import json
import base64
import requests
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import numpy as np
import pandas as pd
import pprint
import random
import pickle

from sklearn.preprocessing import MinMaxScaler
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

# 최신 크롬 드라이버 사용하도록 세팅: 현재 OS에 설치된 크롬 브라우저 버전에 맞게 cache에 드라이버 설치
from selenium.webdriver.chrome.service import Service
service = Service(ChromeDriverManager().install())

client_credentials_manager = SpotifyClientCredentials(
    client_id='28d5b804b7844ec5a151805549cbc4f7', client_secret='e4684deb41a84adbbdfe94f366f84d78')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#웹크롤링


# 최신 크롬 드라이버 사용하도록 세팅: 현재 OS에 설치된 크롬 브라우저 버전에 맞게 cache에 드라이버 설치
service = Service(ChromeDriverManager().install())


# 스포티파이 토큰 얻기

client_id = "28d5b804b7844ec5a151805549cbc4f7"
client_secret = "e4684deb41a84adbbdfe94f366f84d78"
endpoint = "https://accounts.spotify.com/api/token"

# python 3.x 버전
encoded = base64.b64encode("{}:{}".format(
    client_id, client_secret).encode('utf-8')).decode('ascii')

headers = {"Authorization": "Basic {}".format(encoded)}
payload = {"grant_type": "client_credentials"}
response = requests.post(endpoint, data=payload, headers=headers)
access_token = json.loads(response.text)['access_token']



# project.py 실행)  
# !python project.py -c haarcascade_frontalface_default.xml -m output/emotion_model.hdf5

command = ['python', 'project.py', '-c',
           'haarcascade_frontalface_default.xml', '-m', 'output/emotion_model.hdf5']
subprocess.run(command, check=True)


# project.py에서 저장한 변수 불러오기
with open('result.txt', 'r') as file:
    text = file.read()

emotion = text

print("당신의 감정은 ", emotion, "입니다.")
# %% [markdown]
# #### Load Data 

# %%
artist_name =[]
track_name = []
track_popularity =[]
track_id =[]

def get_track_info():   # 연도를 설정하여 1000개의 데이터를 가져옴
    
    #### 전체 데이터가 1000개가 안되는 연도도 존재 

    # 사용자로부터 연도 입력받기 
    year = input("어떤 연도의 음악을 듣고 싶나요? (1950~2023)")
    q_par = "year:" + year
    if int(year) > 2023 or int(year) < 1950:
        print('다시 입력하세요.(1950-2023) : ')
        year = input()
        q_par = "year:" + year

    print("""
            노래 로드 중...
            
            """)
     # 반복문으로 데이터 읽기 
    for i in range(0,1000,50):
        track_results = sp.search(q=q_par, type='track', limit=50, offset=i)
        for i, t in enumerate(track_results['tracks']['items']):
            ## 가수 이름, 트랙 이름, 트랙 id, 인기도에 해당하는 정보 리스트 생성 
            artist_name.append(t['artists'][0]['name'])
            track_name.append(t['name'])
            track_id.append(t['id'])
            track_popularity.append(t['popularity'])

get_track_info() # 데이터 로드하기 

# 리스트 데이터를 기반으로 데이터프레임 생성
track_df = pd.DataFrame({'artist_name': artist_name, 'track_name': track_name,
                        'track_id': track_id, 'track_popularity': track_popularity})



# %% [markdown]
# ### Spotify token으로 Audio feature 가져오기 

# %% [markdown]
# 샘플링 

# %%
# 스포티파이 토큰 얻기
import requests
import base64
import json

client_id = "28d5b804b7844ec5a151805549cbc4f7"
client_secret = "e4684deb41a84adbbdfe94f366f84d78"
endpoint = "https://accounts.spotify.com/api/token"

# python 3.x 버전
encoded = base64.b64encode("{}:{}".format(client_id, client_secret).encode('utf-8')).decode('ascii')

headers = {"Authorization": "Basic {}".format(encoded)}
payload = {"grant_type": "client_credentials"}
response = requests.post(endpoint, data=payload, headers=headers)
access_token = json.loads(response.text)['access_token']


# %%
## 1000개 데이터 중 100개 샘플링
def sampling(track_df):
    # df = track_df.sort_values(by=['track_popularity'], ascending=False)[['track_popularity', 'track_name', 'artist_name', 'track_id']].head(100)
    df = track_df.sample(n=100)
    id = track_df['track_id']

    # Top100 음악 오디오 특징 데이터 추출 및 전처리, 곡 명 삽입

    headers = {"Authorization": "Bearer {}".format(access_token)}
    trackinfo_list = []

    for artist, name ,ad in zip(df['artist_name'], df['track_name'], df['track_id']):
        address = "https://api.spotify.com/v1/audio-features/" + ad
        r = requests.get(address, headers=headers).text  ## 10초 
        r_dict = json.loads(r)  # dict 타입으로 변환
        r_dict['name'] = name
        r_dict['artist_name'] = artist
        # del r_dict['type'], r_dict['id'], r_dict['uri'], r_dict['track_href'], r_dict['analysis_url'], r_dict['time_signature'], r_dict['duration_ms']
        trackinfo_list.append(r_dict)

    return pd.DataFrame(trackinfo_list)


play_list = sampling(track_df)  # 100곡 샘플링 
play_list = play_list.drop_duplicates(subset=['name'], keep='first')  # 중복 곡 제거

# 가져온 데이터에 대해 scaler 적용 
new_data = play_list[['danceability', 'energy',
                      'acousticness', 'valence', 'instrumentalness']]

new_data["pos"] = np.where(new_data["valence"] >= 0.6, 1, 0)

## 데이터 정규화 작업
scaler = MinMaxScaler()  # MinMaxScaler
scaled_val = scaler.fit_transform(new_data.values)
scaled_df = pd.DataFrame(scaled_val, index=new_data.index, columns=new_data.columns)  # 데이터 프레임생성 

# # 변환된 데이터를 K-means 모델에 적용하여 군집 예측
# new_data_labels = km.predict(scaled_df)

# %%
# cluster = [[0.28150904, 0.39094467],
#            [0.78353783, 0.6780397],
#            [0.29227031, 0.63323106],
#            [0.81575847, 0.41410231]]

km = pickle.load(open('kmeans.pkl', 'rb'))
new_data_labels = km.predict(scaled_df)
new_data['labels'] = new_data_labels

cluster_mapping = {
    'Angry': 0,
    'Happy': 3,
    'Sad': 2,
    'Neutral': 1
}


# ## 클러스터링 결과로 분류 
# index = list(new_data[new_data['labels'] == cluster_mapping.get(emotion)].index)

# ## 만약 5곡 이하면, 바로 벡터 유사도 적용 
# if len(index) <= 5:
#     play_list = play_list 

# else:
#     play_list = play_list.loc[index]

# play_list.reset_index(inplace=True)
# play_list['dist'] = 0


### 벡터 거리 유사도 계산을 위한 2차원 클러스터 벡터 ## ## 
cluster = [[0.1655862,  0.08796138],    # Angry
            [0.7281191,  0.843921],     # Calm(Neutral)
            [0.18056842, 0.80277347],   # Sad 
            [0.67148181, 0.10245991]]   # Happy



## 유사도 거리 (벡터 길이) => valence와 acousticness 사용 
def distance(p1, p2):
    distance_x = p2[0]-p1[0]
    distance_y = p2[1]-p1[1]
    distance_vec = [distance_x, distance_y]
    norm = (distance_vec[0]**2 + distance_vec[1]**2)**(1/2)
    return norm

play_list['dist'] = None

##### 예측한 감정 라벨 #######
emotion_cluster = cluster_mapping.get(emotion)

# 유사도 거리 계산 후, dist 열에 저장
# if emotion_cluster is not None:
for track in range(len(play_list)):
    # 예외처리 (값이 없는 경우도 존재함)
    if track in play_list.index:
        play_list['dist'][track] = distance(cluster[emotion_cluster], [
                                            play_list['valence'][track], play_list['acousticness'][track]])
    else:
        pass

# Nan 행 제거
play_list.dropna(subset=['dist'], inplace=True)
recommend = play_list.sort_values(
    by=['dist'], ascending=True).head()  # 오름차순 정렬 후 상위 5개 추출


# %% [markdown]
# #### 추천 리스트 출력

# %%
# 추천 리스트 출력 (곡 이름, 아티스트 이름)
choice_list = list(enumerate(
    zip(recommend['name'].values, recommend['artist_name'].values), start=1))

print("당신의 감정은 ", emotion, "입니다.")
print("추천음악:")
x = []
for number, (name, artist_name) in choice_list:
    x.append(f'{name} - {artist_name}')
    print(f"{number}. {name} - {artist_name}")

# %% [markdown]
# #### 유튜브 링크와 연결 

# %%

number = int(input("듣고 싶은 노래가 있나요? 검색할 노래의 인덱스 번호를 입력해주세요. :"))

SEARCH_KEYWORD = x[number-1].replace(' ', '+')

driver = webdriver.Chrome(service=service)
# 스크래핑 할 URL 세팅
URL = "https://www.youtube.com/results?search_query=" + SEARCH_KEYWORD
# 크롬 드라이버를 통해 지정한 URL의 웹 페이지 오픈
driver.get(URL)



# %%



