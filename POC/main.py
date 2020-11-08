!pip install spotipy

import pandas as pd
import streamlit as st

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

import pickle
from scaler_transform import ScalerTransform

kmeans = pickle.load(open("modelos_treinados/save.pkl", "rb"))

cid = 'YOUR_CODE'
secret = 'YOUR_SECRET_CODE'

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager
=
client_credentials_manager)


df_grupo_musicas = pd.read_excel("dados/grupo_musicas.xlsx", usecols=['name', 'album', 'artist', 'genre', 'cluster', 'url'])

def main():


    st.markdown("""
    <style>
    body {
        color: #191970;
        background-color: #FFFFFF;
    }
    </style>
        """, unsafe_allow_html=True)

    #st.sidebar.header('Selecione uma opção')

    st.sidebar.title('Descrição:')
    st.sidebar.markdown(
        '''
        Esta POC compreende em um *simples recomendador de músicas* utilizando machine learning com 
        algoritimo não-supervisionado. O modelo é importado já treinado, o treinamento foi realizado com 
        as músicas e seus atributos, extraídos direto da API do spotify.
        
        Poderá fazer o download do dataframe e selecionar um genero desejado após a busca, além de verificar 
        em uma *wordcloud* os generos contidos.
        
        Consiste em digitar o **nome de uma música** e será recomendado as músicas com atributos semelhantes.
        Os principais atributos são:
        - Popularidade;
        - Dançabilidade;
        - Duração da música;
        - Tempo da batida;
        - Energia;
        - Nível de vocalização;
        - Nível instrumental;
        
        Há uma opção com link da página de um dashboard em **Power bi**, no qual nas páginas:
        - 1 - Poderá ter uma visão geral das músicas que contém na POC e algumas estatísticas;
        - 2 - Poderá realizar algumas análises.
        '''
    )


    st.markdown("<h1 style='text-align: center; color: red;'>Recomendador de músicas do spotify</h1>", unsafe_allow_html=True)


    st.subheader("Para visualizar o dashboard completo, clique no link abaixo:")
    link = '[powerbi](https://app.powerbi.com/view?r=eyJrIjoiOTA1MGRlNTctOWI0MC00MmIyLTljYTItOGYyOWM0YTcxMzQ2IiwidCI6IjI0MDM4ODcxLTk4MzgtNDEyNC04MDJlLTcyMTY3ZGUyNTAzMCJ9)'
    st.markdown(link, unsafe_allow_html=True)

    st.header("Digite abaixo uma música no qual você queira receber sugestões de outras com as mesmas características")

    track = st.text_input("")

    if track != "":

        track_id = sp.search(q='track:' + track, type='track')
        id_track = track_id['tracks']['items'][0]['id']

        def getTrackFeatures(id):
            meta = sp.track(id)
            features = sp.audio_features(id)

            # meta
            name = meta['name']
            album = meta['album']['name']
            artist = meta['album']['artists'][0]['name']
            release_date = meta['album']['release_date']
            length = meta['duration_ms']
            popularity = meta['popularity']

            # features
            track_id = features[0]['id']
            mode = features[0]['mode']
            acousticness = features[0]['acousticness']
            danceability = features[0]['danceability']
            energy = features[0]['energy']
            instrumentalness = features[0]['instrumentalness']
            liveness = features[0]['liveness']
            valence = features[0]['valence']
            loudness = features[0]['loudness']
            speechiness = features[0]['speechiness']
            tempo = features[0]['tempo']
            duration_ms = features[0]['duration_ms']
            time_signature = features[0]['time_signature']

            track = [name, album, artist, release_date, length, popularity, track_id, mode, acousticness, danceability,
                     energy, instrumentalness, liveness, valence, loudness, speechiness, tempo, duration_ms,
                     time_signature]

            return track


        ids_test = [id_track]

        # loop over track ids
        tracks = []
        for i in ids_test:
            time.sleep(.5)
            track = getTrackFeatures(i)
            tracks.append(track)

        # create dataset
        df = pd.DataFrame(tracks,
                          columns=['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'track_id',
                                   'mode',
                                   'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence',
                                   'loudness', 'speechiness', 'tempo', 'duration_ms', 'time_signature'])


        df_num_p = df.select_dtypes(['float64', 'int64']).reset_index()

        pipeline = ScalerTransform()

        scaled_df = pipeline.data_scaler(df_num_p)

        df_scaled = pd.DataFrame(scaled_df, columns=df_num_p.columns).iloc[:, 1:]

        cluster_prev = kmeans.predict(df_scaled)[0]

        #df_grupo_musicas

        st.write('O grupo definido foi: ', cluster_prev)
        #st.dataframe(df_grupo_musicas[df_grupo_musicas['cluster'] == cluster_prev].reset_index())

        df_group = df_grupo_musicas[df_grupo_musicas['cluster'] == cluster_prev].reset_index()
        df_group.drop(['index'], axis=1, inplace=True)

        option_select = df_group['genre'].unique().tolist()
        option_select.insert(0, 'Todos')

        options = st.selectbox('Selecione os generos', option_select, 0)

        if options == 'Todos':
        #st.write(df_group.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.dataframe(df_group)
        else:
            st.dataframe(df_group[df_group['genre'].isin([options])])

        # download do df
        import base64
        def get_table_download_link(df):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            csv = df.to_csv(index=False, sep=';')
            b64 = base64.b64encode(
                csv.encode()
            ).decode()  # some strings <-> bytes conversions necessary here
            return f'<a href="data:file/csv;base64,{b64}" download="grupo_musicas.csv">Download csv file</a>'

        st.markdown(get_table_download_link(df_group.drop(['cluster'], axis=1)), unsafe_allow_html=True)

        st.markdown("")
        st.markdown('O grupo recomendado contém as mesmas características da música inserida compreende os seguintes generos:')

        text = df_group['genre'].apply(lambda x: x[1:-1]).str.replace("'", "").tolist()

        # post wordcloud
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wordcloud = WordCloud().generate(str(text))

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot()

if __name__ == "__main__":
    main()
