# -*- coding: utf-8 -*-
"""
Created on Sat May 28 21:01:56 2022

@author: User
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

os_path = Path(__file__).parents[0] / 'DoMod.csv'
with open(os_path, encoding="utf8", errors='ignore') as f:
    DF = pd.read_csv(f,sep=',',header=0)
DF=DF.drop('Unnamed: 0',axis=1)
DF['rok']=list(map(lambda x: x[:4],DF.Okres))
DF = DF.loc[:,['rok',*DF.columns[:-1]]]
for i in range(2,len(DF.columns)):
    DF.iloc[:,i]=DF.iloc[:,i].astype('int')

st.set_page_config(page_title='Model prognostyczny dla leków', page_icon = ':bar_chart:', layout='wide')

st.sidebar.header('Panel sterowania: ')

Model = st.sidebar.radio(
    'Wybierz model:',
    ('Prognoza','Uzupełnianie danych')
 )
if Model == 'Uzupełnianie danych':
    rok = st.sidebar.multiselect(
    "Wybierz lata:",
    options=DF['rok'].unique(),
    default=['2019','2021'])
    
    new_title = '<b style="font-family:sans-serif;text-align: center; color:rgb(0, 0, 180); font-size: 62px;">Istotni Statystycznie ***</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.title(':bar_chart: Model uzupełniający dane dla leków')
    st.header('Regresja wieloraka')
    st.subheader('Jest to metoda pozwalająca szacować wartości danej wielkości za pomocą znanych już wartości innych wielkości. Polega na '+
         'przedstawieniu w postaci równania liniowego zależności zmiennej objaśnianej w oparciu o zmienne objaśniające. Poniżej znajdują się dane w postaci tabelarycznej użyte do budowy modelu. Przedstawiają one ilość sprzedaży danego leku w poszczególnych miesiącach od marca 2019 do lutego 2022.')
    st.subheader('Dwie pierwsze kolumny to zmienne identyfikujące. Kolejnych siedem to zmiennej objaśniające (kolor niebieski). Ostatnia kolumna (kolor czerwony) przedstawia zmienną, której wartości będziemy szacować - zmienna objaśniana. Chcemy przewidywać ilość sprzedaży Gripexu Hot w kolejnych miesiącach na podstawie wybranych leków tego samego typu.')
    st.markdown('##')
    DF_selection= DF.query(
        "rok == @rok")
    #.style.set_properties(**{'color': 'blue'}, subset=['GRIPEX HOT        '],axis=0)
    #.style.set_properties(**{'color': ['blue','blue','blue','blue','blue','blue','blue','red']}, subset=DF.iloc[:,2:].columns,axis=0)


    st.dataframe(DF_selection.style.set_properties(**{'color': 'red'}, subset=['GRIPEX HOT        '],axis=0).set_properties(**{'color':'blue'},subset=DF.iloc[:,2:9].columns,axis=0))
    st.markdown('---')
    st.header(':cyclone: Korelacja zmiennych')
    st.subheader('W celu sprawdzenia "mocy" zależności między poszczególnymi zmiennymi budujemy macierz korealacji. Na przecięciach kolumn z wierszami znajdują się wartości współczynników korelacji liniowej - Pearsona. Liczba ta mieści się w zakresie od -1 do 1. Im ta liczba jest większa co do wartości bezwzględnej tym zależność liniowa między dwoma zmiennymi rośnie.'+
                 ' Obok dla lepszego rozeznania widnieją wykresy rozrzutu. Są one potwierdzeniem na liniową zależność dla zmiennych z powyższymi warunkami.')

    cols=DF_selection.iloc[:,2:].columns
    cm = np.corrcoef(DF_selection[cols].values.T)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=cm,x=cols,y=cols,text=cm.round(2),texttemplate="%{text}",textfont={"size":25},
                            type='heatmap',colorscale='deep'))
    fig.update_layout(#margin = dict(t=200,r=200,b=200,l=200),
        title = '<b>Macierz korelacji',title_x=0.5,title_font={"size":40},
        showlegend = False,
        width = 900, height = 900,)
        #autosize = False )


    fig1 = px.scatter_matrix(DF_selection.iloc[:,2:])

    fig1.update_layout(width=900,height=900,title = '<b>Wykresy rozrzutu',title_x=0.5,title_font={"size":40},)#plot_bgcolor='#FFFFFF')

    lc,rc = st.columns(2)
    lc.plotly_chart(fig)
    rc.plotly_chart(fig1)

    st.subheader('Chcielibyśmy aby zmienne objaśniane były mocno skorelowane ze zmienną objaśnianą i jednocześnie słabo skorelowane między sobą.')
    st.markdown('---')
    st.header(':bulb: Budowa modelu i interpretacja')
    st.subheader('Dużą zaletą modelu jest jego prostota. Nasze równanie będzie zawierało dwie zmienne objaśniające. Selekcja predyktorów do modelu oparta jest o metodę RFE (Recursive Feature Elimination).')




    ###########MODEL#############
    #Model regresji z wszystkimi predyktorami
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    names = DF_selection.iloc[:,2:9].columns
    X=DF_selection[names]
    y=DF_selection['GRIPEX HOT        ']
    model.fit(X,y)

    #Selekcja predyktorow w oparciu o RFE
    #dokumentacja: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    from sklearn.feature_selection import RFE
    #n_features_to_select=2, wybieram dwa najważniejsze predyktory
    selector1 = RFE(model, n_features_to_select=2, step=1)
    selector1 = selector1.fit(X, y)
    #True wskazuje ktore dwa sposrod zaproponowanych sa wybierane
    wybrane = DF_selection.iloc[:,2:9].columns[selector1.support_]
    #Ranking poszczegolnych predyktorow
    #selector1.ranking_
    st.info('**Uzyskane zmienne: '+str(wybrane[0])+', '+str(wybrane[1])+'.**')


    #Model regresji z dwoma najwazniejszymi predyktorami wybranymi w powyzszej selekcji
    model_new=LinearRegression()
    names_new=wybrane
    X_new=DF_selection[names_new]   # y
    model_new.fit(X_new,y)





    # Wspolczynniki rowaniania regresji 
    st.info('**Współczyniki równania regresji: '+str(round(model_new.coef_[0],3))+', '+str(round(model_new.coef_[1],3))+'**')
    # Wyraz wolny w rownaniu regresji
    st.info('**Wyraz wolny w równaniu regresji: '+str(round(model_new.intercept_,3))+'**')
    st.subheader('Wzrost zmiennej '+str(wybrane[0])+'o jedną jedonstkę powoduje zmianę wartości GRIPEXU HOT o '+str(round(model_new.coef_[0],3))+', a wzrost zmiennej '+str(wybrane[1])+'o jedną jednostkę powoduję zmianę wartości GRIPEXU HOT o '+str(round(model_new.coef_[1],3)))



    st.success('**Równanie regresjii wielorakiej: '+str('GRIPEX HOT        ') + '= ' + '('+str(round(model_new.coef_[0],3)) + ')' + '\*'+str(wybrane[0])+ '+ ' + '('+str(round(model_new.coef_[1],3))+')'+'\*'+str(wybrane[1])+'+ '+'('+str(round(model_new.intercept_,3))+')'+'**')

    st.subheader('Współczynnik determinacji mówi nam jak dobrze model został dopasowany do danych. Im bliżej 1 tym lepsza jakość modelu.')

    #Wpsolczynnik R^2 mozna tez policzyc tak:
    st.error('**Współczynnik determinacji: '+str(round(model_new.score(X_new,y),3))+'**')

    #Przewidywana cena lekY na podstawie ceny lek1, lek2 w oparciu o zbudowany model regresji
    y_pred=model_new.predict(X_new)
    #Wartosci reszt
    residuals = (y - y_pred)




    #Test niezaleznosci dla reszt. Im blizsza 2 wartosc otrzymanej statystyki tym bardziej mozna przyjac niezaleznosc reszt 
    # Wiecej o tescie Durbina-Watsona w Pythonie: https://www.statology.org/durbin-watson-test-python/
    from statsmodels.stats.stattools import durbin_watson 
    d_b = durbin_watson(residuals) 
    print(d_b)


    #RMSE
    from sklearn.metrics import mean_squared_error
    MSE=mean_squared_error(y, y_pred)
    import math
    RMSE=math.sqrt(MSE)
    #print('standardowy bład oszacowania', RMSE)
    st.subheader('Standardowy błąd oszacowania to statystyka, która informuje nas o tym jak średnio przewidziane wartości odstają od faktycznych.')
    st.warning('**Standardowy bład oszacowania: '+str(round(RMSE,3))+'**')



    st.markdown('---')
    st.header(':chart_with_downwards_trend: Wizualizacja wyników')
    st.subheader('Wykres przedstawia zależność zmiennej objaśnianej od dwóch zmiennych objaśnijących. Dodatkowo widzimy płaszczyznę będącą wykresem funkcji regresji wielorakiej zbudowanej na podstawie modelu. Jeżeli model jest dobry to punkciki powinny znajdować się w niewielkim otoczeniu od płaszczyzny.')
    x = np.linspace(0, DF_selection[wybrane[0]].max(), 100)
    y = np.linspace(0, DF_selection[wybrane[1]].max(), 100)

    X, Y = np.meshgrid(x, y)

    Z1 = X*model_new.coef_[0] + Y*model_new.coef_[1] + model_new.intercept_

    fig2 = px.scatter_3d(DF_selection, x=wybrane[0], y=wybrane[1], z='GRIPEX HOT        ')
    fig2.update_traces(marker=dict(size=5))
    fig2.add_traces(go.Surface(x=X, y=Y, z=Z1, name='Płaszczyzna regresji',opacity=0.6))
    fig2.update_layout(height=1000,width=1000,showlegend=False)

    st.plotly_chart(fig2,True)

    st.markdown('---')
    st.header('Szacowanie na podstawie danych z wybranego miesiąca')

    def Lek_pred(lek1,lek2):
        return round(lek1*model_new.coef_[0] + lek2*model_new.coef_[1] + model_new.intercept_,3)


    lc1, rc1 = st.columns((3,3))
    lek1 = lc1.number_input('Podaj ilość sprzedaży pierwszego leku - '+str(wybrane[0])+': ',value=85000,min_value=0,step=100)
    lek2 = rc1.number_input('Podaj ilość sprzedaży drugiego leku - '+str(wybrane[1])+': ',value=70,min_value=0,step=100)
    st.subheader(f'\t Przewidziana ilość sprzedaży GRIPEXU HOT wynosi: {Lek_pred(lek1,lek2)}')
else:
    new_title = '<b style="font-family:sans-serif;text-align: center; color:rgb(0, 0, 180); font-size: 62px;">Istotni Statystycznie ***</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.title(':chart_with_upwards_trend: Model prognostyczny dla leków')
    st.header('Szeregi czasowe')
    st.subheader('Szereg czasowy to ciąg obserwacji przedstawiający formowanie się danego zjawiska w kolejnych okresach czasu (dniach, miesiącach, kwartałach, latach). '+
         'Modelem szeregu czasowego służącym do określenia przyszłej wartości zmiennej prognozowanej w momencie prognozowania jest model formalny, którego zmiennymi objaśniającymi mogą być tylko zmienne czasowe oraz przyszłe wartości lub otrzymane prognozy.')
    st.subheader('Wykres poniżej przedstawia ilościową sprzedaż Gripexu Hot w poszczególnych miesiącach od marca 2019 do lutego 2022. Jest to przykład szeregu czasowego.')
    
    fig = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=list(DF.Okres.astype('string')),
                            ticktext = DF.Okres.astype('string'),linecolor='black',tickwidth=1,tickcolor='black',ticks="outside"),
    yaxis = dict(linecolor='black',title='<b>Liczba sprzedaży [w sztukach]',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black')
    ))
    fig.add_trace(go.Scatter(
        x = DF.Okres,
        y = DF['GRIPEX HOT        '],
        name = "GRIPEX HOT",
        line_color = 'red',
        mode='lines+markers',
        marker_size=10,
        line_width=3
        ))

    # Use string to set start xaxis range
    fig.update_layout(plot_bgcolor='white',height=800,font=dict(
            size=18,
            color="Black"),title='<b>Sprzedaż ilościowa Gripexu Hot w podziale na miesiące',title_x=0.5)
    st.plotly_chart(fig,True)
    st.subheader('Zakłada się, że obserwowany przebieg zjawiska składa się z części systematycznej (trend, sezonowość) w oparciu, o które buduje się model. Dodatkowo wyróżnia się część przypadkową (szum). Chcąc wyodrębnić wymienione składniki dokonuje się dekompozycji szeregu czasowego')
    st.header('Dekompozycja szeregu czasowego')
    
    
    
    l1,c1,r1 = st.columns(3)
    
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    DF1 = DF
    DF1['miesiac'] = pd.to_datetime(DF1['Okres'])
    DF1.set_index(DF1.miesiac,inplace=True)
    

    decomp = seasonal_decompose(DF1.iloc[:,[9]])
    
    fig1 = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=list(DF1.Okres.astype('string')),
                            ticktext = DF1.Okres.astype('string'),linecolor='black',tickwidth=1,tickcolor='black',ticks="outside",tickfont=dict(size=12)),
    yaxis = dict(linecolor='black',title='<b>Trend',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black')
    ))
    fig1.add_trace(go.Scatter(
        x = DF1.Okres,
        y = decomp.trend.values,

        line_color = 'navy',
        mode='lines+markers',
        marker_size=8,
        line_width=4
        ))


    
    fig1.update_layout(plot_bgcolor='white',font=dict(
            size=18,
            color="Black"),title='<b>Dekompozycja - trend',title_x=0.5,height=300)
   

    fig2 = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=list(DF1.Okres.astype('string')),
                            ticktext = DF1.Okres.astype('string'),linecolor='black',tickwidth=1,tickcolor='black',ticks="outside",tickfont=dict(size=12)),
    yaxis = dict(linecolor='black',title='<b>Sezonowość',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black',)
    ))
    fig2.add_trace(go.Scatter(
        x = DF1.Okres,
        y = decomp.seasonal.values,

        line_color = 'navy',
        mode='lines+markers',
        marker_size=8,
        line_width=4
        ))


   
    fig2.update_layout(plot_bgcolor='white',font=dict(
            size=18,
            color="Black"),title='<b>Dekompozycja - sezonowość',title_x=0.5,height=300)
    
    
    fig3 = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=list(DF1.Okres.astype('string')),
                            ticktext = DF1.Okres.astype('string'),linecolor='black',tickwidth=1,tickcolor='black',ticks="outside",tickfont=dict(size=12)),
    yaxis = dict(linecolor='black',title='<b>Szum',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black')
    ))
    fig3.add_trace(go.Scatter(
        x = DF1.Okres,
        y = decomp.resid.values,

        line_color = 'navy',
        mode='markers',
        marker_size=10,
        ))
    fig3.add_hline(y=0,line_width=3,line_color='black')

    
    fig3.update_layout(plot_bgcolor='white',font=dict(
            size=18,
            color="Black"),title='<b>Dekompozycja - szum',title_x=0.5,height=300)
    
    l1.plotly_chart(fig1)
    c1.plotly_chart(fig2)
    r1.plotly_chart(fig3)

    st.markdown('---')
    
    
    df  = DF[['Okres','GRIPEX HOT        ']]
    df['Okres1'] = list(range(1,len(list(df.Okres))+1))
    df = df.loc[:,['Okres1','Okres','GRIPEX HOT        ']]
    
    from sklearn.linear_model import LinearRegression
    X = df.iloc[:,[0]]
    y = df.iloc[:,2]
    
    model = LinearRegression()
    model.fit(X,y)
    
    pred = model.predict(X)

    fig = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=list(DF.Okres.astype('string')),
                            ticktext = DF.Okres.astype('string'),linecolor='black',tickwidth=1,tickcolor='black',ticks="outside"),
    yaxis = dict(linecolor='black',title='<b>Liczba sprzedaży [w sztukach]',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black')
    ))
    fig.add_trace(go.Scatter(
        x = df.Okres,
        y = df['GRIPEX HOT        '],
        name = "GRIPEX HOT",
        line_color = 'red',
        mode='lines+markers',
        marker_size=8,
        line_width=3
        ))
    fig.add_trace(go.Scatter(
        x = df.Okres,
        y = pred,
        name = "Linia trendu",
        mode='lines+markers',
        line_color = 'dodgerblue',
        marker_size=6,
        opacity = 0.8))

    # Use string to set start xaxis range
    fig.update_layout(plot_bgcolor='white',height=500,font=dict(
            size=18,
            color="Black"),title='<b>Sprzedaż ilościowa Gripexu Hot - prognozy vs. rzeczywistość',title_x=0.5)
    from sklearn.metrics import mean_squared_error
    MSE=mean_squared_error(y, pred)
    RMSE=np.sqrt(MSE)
    
    st.header(':clock10: Model trendu liniowego')
    st.subheader('Do budowy modelu używamy funkcji liniowej. Wykres przedstawia sprzedaż ilościową Gripexu Hot wraz z linią trendu.')
    st.markdown('###')
    lc,rc = st.columns((1,3))
    
    lc.markdown('###')
    lc.markdown('###')
    lc.write('**Równanie:**')
    lc.success('**Ilość = '+str(round(model.coef_[0],3))+'*'+'Okres + '+str(round(model.intercept_,3))+'**')
    lc.write('**Błąd średniokwadratowy:**')
    lc.warning('**RMSE: '+str(round(RMSE,3))+'**')
    lc.write('**Współczynnik determinacji**')
    lc.info('**'+r'$$R^2$$'+' = '+str(round(model.score(X,y),3))+'**')
    rc.plotly_chart(fig,True)
    st.markdown('###')
    st.markdown('###')
    st.header('Prognoza na najbliższe lata:')
    lc1,rc1 = st.columns((2,2))
    m,r = lc1.number_input('Podaj miesiąc: ',min_value=1,max_value=12,step=1),rc1.number_input('Podaj rok: ',value=2023,min_value=2022,max_value=2026,step=1)
    
    a = 36
    t = []
    for i in range(2022,2027):
        for j in range(1,13):
            t.append([j,i,a])
            a+=1
    def szukaj(m,r):
        for i in t:
            if i[0] == m and i[1] == r:
                return i[2]
    st.subheader('Przewidziana ilość sprzedaży w '+str(int(m))+'-'+str(int(r))+' to: '+str(int(round(model.predict([[szukaj(m,r)]])[0],0))))
    
    st.markdown('---')
    
    st.header(':clock3: Metoda średniej ruchomej')
    lc, rc = st.columns((1,3))
    lc.subheader('Sposób prognozowania tą metodą polega na policzeniu średniej z '+r'$k$'+' ostatnich okresów. Jest to sposób prognozy krótkoterminowej. Największa trudność polega na optymalnym doborze '+r'$k$.') 
   
    k = lc.number_input('Wybierz k:',value=3,min_value=1,max_value=10,step=1)
    df['Rolling_Mean'] = df.iloc[:,2].rolling(k).mean()
    
    from sklearn.metrics import mean_squared_error
    MSE=mean_squared_error(y[k:], list(df['Rolling_Mean'][k:]))
    RMSE=np.sqrt(MSE)
    lc.write('**Błąd średniokwadratowy:**')
    lc.warning('**RMSE: '+str(round(RMSE,3))+'**')
    
    fig = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=list(df.Okres.astype('string')),
                            ticktext = df.Okres.astype('string'),linecolor='black',tickwidth=1,tickcolor='black',ticks="outside"),
    yaxis = dict(linecolor='black',title='<b>Liczba sprzedaży [w sztukach]',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black')
    ))
    fig.add_trace(go.Scatter(
        x = df.Okres,
        y = df['GRIPEX HOT        '],
        name = "GRIPEX HOT",
        line_color = 'red',
        mode='lines+markers',
        marker_size=8,
        line_width=3
        ))
    fig.add_trace(go.Scatter(
        x = df.Okres,
        y = df['Rolling_Mean'],
        name = "Średnia ruchoma",
        mode='lines+markers',
        line_color = 'dodgerblue',
        marker_size=6,
        opacity = 0.8))

    # Use string to set start xaxis range
    fig.update_layout(plot_bgcolor='white',height=550,font=dict(
            size=18,
            color="Black"),title='<b>Sprzedaż ilościowa Gripexu Hot - prognozy vs. rzeczywistość',title_x=0.5)
    rc.plotly_chart(fig,True)
    pred_mean = df.iloc[len(df.iloc[:,2])-k:,2].mean()
    st.subheader('Przewidziana ilość sprzedaży w 3-2022 to: '+str(int(round(pred_mean,0))))
    st.markdown('---')
    
    st.header(':clock330: Model SARIMA')
    ll, rr = st.columns((1,3))
    
    ll.subheader('Metoda SARIMA pozwala na dostosowanie modelu, określając kolejność autoregresji, różnicowania i średniej kroczącej, jak również sezonowych odpowiedników tych składników. Tym samym umożliwiając dokładne modelowanie szeregów czasowych. ')
    ll.markdown('###')
    ll.subheader('Wybierz parametry modelu: ')
    ll.markdown('###')
    p = ll.number_input('Wybierz p (parametr autoregresji):',value=2,min_value=0,max_value=10,step=1)
    d = ll.number_input('Wybierz d (stopień zróżnicowania szeregu):',min_value=0,max_value=2,step=1)
    q = ll.number_input('Wybierz q (parametr średniej ruchomej):',min_value=0,max_value=10,step=1)
    s = ll.number_input('Wybierz s (parametr sezonowości):',value=12,min_value=0,max_value=12,step=1)
     
    
    ll.markdown('###')
    ll.markdown('###')
    ll.markdown('###')
    
    
    
    ##from statsmodels.tsa.arima.model import ARIMA
    ##arima_model = ARIMA(df.iloc[:,2],order=(p,d,q))
    ##model = arima_model.fit()
    
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    sarima_model = SARIMAX(df.iloc[:,2],order=(p,d,q),seasonal_order=(p,d,q,s))
    model=sarima_model.fit()
    
    
    
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    MSE=mean_squared_error(y, list(model.predict().values))
    MAE=mean_absolute_error(y, list(model.predict().values))
    RMSE=np.sqrt(MSE)
    ll.subheader('Średni błąd absolutny: ')
    ll.info('**MAE: '+str(round(MAE,3))+'**')
    ll.subheader('Błąd średniokwadratowy: ')
    ll.warning('**RMSE: '+str(round(RMSE,3))+'**')
    
    
   
    
    fig = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=list(df.Okres.astype('string')),
                            ticktext = df.Okres.astype('string'),linecolor='black',tickwidth=1,tickcolor='black',ticks="outside"),
    yaxis = dict(linecolor='black',title='<b>Liczba sprzedaży [w sztukach]',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black')
    ))
    fig.add_trace(go.Scatter(
        x = df.Okres,
        y = df['GRIPEX HOT        '],
        name = "GRIPEX HOT",
        line_color = 'red',
        marker_size=8,
        mode='lines+markers',
        line_width=3
        ))
    fig.add_trace(go.Scatter(
        x = df.Okres,
        y = model.predict().values,
        name = "SARIMA",
        mode='lines+markers',
        marker_size=6,
        line_color = 'dodgerblue',
        opacity = 0.8))

    # Use string to set start xaxis range
    fig.update_layout(plot_bgcolor='white',font=dict(
            size=18,
            color="Black"),title='<b>Sprzedaż ilościowa Gripexu Hot w podziale na miesiące',title_x=0.5,height=550)
    
    
    t = []
    for i in range(2022,2027):
        for j in range(1,13):
            if j>9:
                t.append(str(i)+'-'+str(j))
            else:
                t.append(str(i)+'-'+'0'+str(j))
    u = list(df.Okres.astype('string'))
    okresy = u+t[2:]
    
    fig1 = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,tickfont=dict(size=14),title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=okresy[:72],
                            ticktext = okresy[:72],linecolor='black',tickwidth=1,tickcolor='black',ticks="outside"),
    yaxis = dict(linecolor='black',title='<b>Liczba sprzedaży [w sztukach]',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black',
                )
    ))
    fig1.add_trace(go.Scatter(
        x = okresy[:36],
        y = df['GRIPEX HOT        '],
        name = "GRIPEX HOT",
        line_color = 'red',
        marker_size=8,
        mode='lines+markers',
        line_width=3
        ))
    fig1.add_trace(go.Scatter(
        x = okresy[36:],
        y = model.predict(36,72),
        name = "SARIMA_pred",
        mode='lines+markers',
        marker_size=6,
        line_color = 'green',
        opacity = 0.8))

    # Use string to set start xaxis range
    fig1.update_layout(plot_bgcolor='white',font=dict(
            size=18,
            color="Black"),title='<b>Sprzedaż ilościowa Gripexu Hot w podziale na miesiące',title_x=0.5,height=550)
    

    rr.plotly_chart(fig,True)
    rr.plotly_chart(fig1,True)
    
    st.header('Prognoza na najbliższe lata:')
    a,b = st.columns(2)
    b1 = b.number_input('Podaj rok:',value=2024,min_value=2022,max_value=2024,step=1)
    a1 = a.number_input('Podaj miesiąc:',value=6,min_value=1,max_value=12,step=1)
    
    tab = list(df.iloc[[34,35],2]) + list(model.predict(36,72))
    a = 0
    t = []
    for i in range(2022,2027):
        for j in range(1,13):
            t.append([j,i,a])
            a+=1
    def szukaj(m,r):
        for i in t:
            if i[0] == m and i[1] == r:
                return i[2]
    st.subheader('Przewidziana ilość sprzedaży w '+str(int(a1))+'-'+str(int(b1))+' to: '+str(int(round(tab[szukaj(a1,b1)],0))) )
    st.markdown('---')

    
    # holt winters 
    # single exponential smoothing
    #from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
    # double and triple exponential smoothing
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    st.header(':clock230: Model Holta Wintersa')
    lc,rc = st.columns((1,3))
    lc.subheader('Model Holta-Wintersa jest jedną z technik prognozowania wykorzystujących tzw. wygładzenie wykładnicze. Wygładzenie polega na stworzeniu ważonej średniej ruchomej, której wagi określa się według schematu - im starsza informacja o badanym zjawisku, tym mniejszą wartość stanowi ona dla aktualnej prognozy.')
    lc.markdown('###')
    wyb1 = lc.selectbox('Wybierz typ trendu: ',['addytywny','multiplikatywny'])
    wyb2 = lc.selectbox('Wybierz typ sezonowości: ',['addytywny','multiplikatywny'])
    wyb3 = lc.selectbox('Czy stłumić składnik trendu: ',[False,True])
    
    lc.markdown('###')
    lc.markdown('###')
 
    
    
    df['HWES'] = ExponentialSmoothing(df['GRIPEX HOT        '],damped=wyb3,trend=wyb1[:3],seasonal=wyb2[:3],seasonal_periods=12).fit().fittedvalues
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    MSE=mean_squared_error(y, list(df['HWES']))
    MAE=mean_absolute_error(y, list(df['HWES']))
    RMSE=np.sqrt(MSE)
    lc.subheader('Średni błąd absolutny: ')
    lc.info('**MAE :'+str(round(MAE,3))+'**')
    lc.subheader('Błąd średniokwadratowy: ')
    lc.warning('**RMSE :'+str(round(RMSE,3))+'**')
  

   
    fig = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=list(df.Okres.astype('string')),
                            ticktext = list(df.Okres.astype('string')),linecolor='black',tickwidth=1,tickcolor='black',ticks="outside"),
    yaxis = dict(linecolor='black',title='<b>Liczba sprzedaży [w sztukach]',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black')
    ))
    fig.add_trace(go.Scatter(
        x = df.Okres,
        y = df['GRIPEX HOT        '],
        name = "GRIPEX HOT",
        line_color = 'red',
        marker_size=8,
        mode='lines+markers',
        line_width=3
        ))
    fig.add_trace(go.Scatter(
        x = df.Okres,
        y = df['HWES'],
        name = "HWES",
        mode='lines+markers',
        marker_size=6,
        line_color = 'dodgerblue',
        opacity = 0.8))

    # Use string to set start xaxis range
    fig.update_layout(plot_bgcolor='white',height=550,font=dict(
            size=18,
            color="Black"),title='<b>Sprzedaż ilościowa Gripexu Hot - prognozy vs. rzeczywistość',title_x=0.5)
    
    
    t = []
    for i in range(2022,2027):
        for j in range(1,13):
            if j>9:
                t.append(str(i)+'-'+str(j))
            else:
                t.append(str(i)+'-'+'0'+str(j))
    u = list(df.Okres.astype('string'))
    okresy = u+t[2:]

    
    fitted_model = ExponentialSmoothing(df['GRIPEX HOT        '],damped=wyb3,trend=wyb1[:3],seasonal=wyb2[:3],seasonal_periods=12).fit()
    test_predictions = fitted_model.forecast(36) 
    
    fig1 = go.Figure(layout =go.Layout(
    xaxis = dict(showgrid=True,tickfont=dict(size=14),title='<b>Okres', ticklabelmode="period", dtick="M1", tickformat="%b\n%",tickangle=45,tickvals=okresy[:72],
                            ticktext = okresy[:72],linecolor='black',tickwidth=1,tickcolor='black',ticks="outside"),
    yaxis = dict(linecolor='black',title='<b>Liczba sprzedaży [w sztukach]',tickwidth=1,tickcolor='black',ticks="outside",gridcolor='black')
    ))
    fig1.add_trace(go.Scatter(
        x = okresy[:36],
        y = df['GRIPEX HOT        '],
        name = "GRIPEX HOT",
        line_color = 'red',
        mode='lines+markers',
        marker_size=8,
        line_width=3
        ))
    fig1.add_trace(go.Scatter(
        x = okresy[36:],
        y = test_predictions.values,
        name = "HWES_pred",
        mode='lines+markers',
        marker_size=6,
        line_color = 'green',
        opacity = 1))

    # Use string to set start xaxis range
    fig1.update_layout(plot_bgcolor='white',height=550,font=dict(
            size=18,
            color="Black"),title='<b>Prognoza sprzedaży ilościowej Gripexu Hot przy urzyciu modelu Holta-Wintersa',title_x=0.5)
    
    
   
    rc.plotly_chart(fig,True)
    rc.markdown('###')
    rc.markdown('###')
    rc.plotly_chart(fig1,True)
    
    st.header('Prognoza na najbliższe lata:')
    a,b = st.columns(2)
    b1 = b.number_input('Podaj rok : ',value=2024,min_value=2022,max_value=2024,step=1)
    a1 = a.number_input('Podaj miesiąc : ',value=6,min_value=1,max_value=12,step=1)
    
    tab = list(df.iloc[[34,35],2]) + list(test_predictions.values)
    a = 0
    t = []
    for i in range(2022,2027):
        for j in range(1,13):
            t.append([j,i,a])
            a+=1
    def szukaj(m,r):
        for i in t:
            if i[0] == m and i[1] == r:
                return i[2]
    st.subheader('Przewidziana ilość sprzedaży w '+str(int(a1))+'-'+str(int(b1))+' to: '+str(int(round(tab[szukaj(a1,b1)],0))) )

    


