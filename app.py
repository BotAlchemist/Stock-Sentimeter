# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 08:14:40 2021

@author: Sumit
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
import datetime  
import time
from wordcloud import WordCloud
from pytrends.request import TrendReq
import seaborn as sns
import plotly.express as px



#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
st.set_page_config(layout='wide',page_title="Sentimeter")
st.set_option('deprecation.showPyplotGlobalUse', False)



html_header='''
<div style= 'background-color: #00B28F; padding:10px';>
<h1 style= "color:white; text-align:center;"><b> Stock-Sentimeter</b></h1>
</div>
<br>
'''
st.sidebar.markdown(html_header, unsafe_allow_html=True)
st.sidebar.text("Developed by Sumit Srivastava")
st.sidebar.text("sumvast@gmail.com")

# Create a SentimentIntensityAnalyzer object.
vader = SentimentIntensityAnalyzer()
pytrends = TrendReq(hl='en-US',timeout=(10,25))


df_pg= pd.read_csv("NSE_stock.csv")
df_pg_trends= pd.read_csv("NSE_stock_google.csv")
product_list= df_pg['Product'].unique()
product_list_google= df_pg_trends['Product'].unique()

dict_pg= dict(df_pg.values)
dict_pg_trends= dict(df_pg_trends.values)


i_source = st.sidebar.selectbox("Source", ['Twitter', 'Google Trends'])

if i_source == "Twitter":

    ################################################################### Update & fetch tweets #######################################
    consumerKey = 'VsyMctyRBOnED0LBs3p6god55'
    consumerSecret = '8QLZGdnvyR4tqGJXsYgux574JiiiMb2SUgWecBectWApQVEZfk'
    accessToken = '783919955607560192-XMnzqtx7E7ggihOBdcaQKUtEKX9oTu7'
    accessTokenSecret = 'z85SZocmC8kQMlFa34XiY7KJcdjxVhokCcHhYwx4KWd8s'
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    with st.beta_expander("Refresh"):
        update_col1, update_col2, update_col3, update_col4 ,update_col5= st.beta_columns(5)
        
        
        today = datetime.date.today()
        yesterday = today + datetime.timedelta(days=-5)
    
    
        password= update_col1.text_input("Password", type="password")
        since_date = update_col2.date_input('Since', yesterday)
        until_date = update_col3.date_input('Until', today)
        delta= until_date- since_date
        no_of_days= delta.days
        
        no_of_tweets= update_col4.text_input("No. of tweets", 10)
        no_of_tweets= int(no_of_tweets)
        update_col5.markdown("### ")
        if update_col5.button("Refresh"):
            if password== 'Tweepy@007':
                #st.write("Updating...")
                
                
                my_bar = st.progress(0)
                i_counter=0
                tweets_results=[]
                for search_words in product_list:
                    new_search = search_words + " -filter:retweets"
                    
                    for i_days in range(no_of_days + 1):
                        tweets= tweepy.Cursor(api.search,q=new_search,
                                               lang="en",
                                               since_id= since_date + datetime.timedelta(days=i_days) ,
                                               until= since_date + datetime.timedelta(days=i_days)
                                               ).items(no_of_tweets)
                        for tweet in tweets:
                            
                            sentiment_result= vader.polarity_scores(tweet.text)
                            compound_value = sentiment_result['compound']
                            if compound_value >= 0.05:
                                overall_value= "Positive"
                            elif compound_value <= -0.05:
                                overall_value= "Negative"
                            else:
                                overall_value= "Neutral"
                            
                            
                            
                            tweets_results.append([search_words, dict_pg[search_words],  tweet.created_at, tweet.text,overall_value, round(sentiment_result['neg']*100,2), round(sentiment_result['neu']*100,2), round(sentiment_result['pos']*100,2) ])
                        
                    time.sleep(2)
                    my_bar.progress((i_counter + 1)/ len(product_list))
                    i_counter+=1
                
                tweets_results_df = pd.DataFrame(tweets_results, columns= ['Keyword',"Category", 'Created At','Tweet' ,'Overall','Negative', 'Neutral','Positive'])
                
                
                #tweets_results_df =pd.datetime(tweets_results_df['Created At'])
                #st.write(tweets_results_df)
                tweets_results_df.to_csv("Tweets.csv", index=False)
                
            else:
                st.error("Please enter password")
            
    st.markdown(''' ---''')
    ################################################## Get sentiment & result ##############################################
    df_tweets= pd.read_csv("Tweets.csv")
    

        
    df_tweets['Created At']= pd.to_datetime(df_tweets['Created At']).dt.date
    
    
    st.sidebar.markdown("### Select Categories")
    filter_category=[]
    
    select_all = st.sidebar.checkbox("Select All", True)
    
    if select_all:
        for i_category in df_tweets['Category'].unique():
            if st.sidebar.checkbox(i_category, True):
                filter_category.append(i_category)
    else:
        for i_category in df_tweets['Category'].unique():
            if st.sidebar.checkbox(i_category, False):
                filter_category.append(i_category)
        
            
    if len(filter_category) >0:
        df_category= df_tweets[df_tweets['Category'].isin(filter_category)]
        
        
        
        filter_1, filter_2, filter_3= st.beta_columns(3)
        filter_keywords= filter_1.multiselect("Filter Keywords", df_category['Keyword'].unique())
        
        #print(df_category['Created At'].min(), df_category['Created At'].max())
        since_date = filter_2.date_input('From', df_category['Created At'].min(), key='date_since')
        until_date = filter_3.date_input('Until', df_category['Created At'].max(), key= 'date_till')
        
        st.markdown('''---''')
        
        
        
        
        if len(filter_keywords) >0:
            df_keyword= df_category[df_category['Keyword'].isin(filter_keywords)]
        else:
            df_keyword= df_category.copy()
            
            
        df_keyword= df_keyword.set_index("Created At")
        df_keyword= df_keyword[(df_keyword.index >=since_date ) & (df_keyword.index <= until_date)]
        
        #print(df_keyword.info())
        #print(since_date, )
        
        count_of_tweets= df_keyword['Overall'].value_counts()
        count_of_neg_tweet= count_of_tweets['Negative']
        count_of_neu_tweet= count_of_tweets['Neutral']
        count_of_pos_tweet= count_of_tweets['Positive']
        
        sentiment_1, sentiment_2, sentiment_3= st.beta_columns(3)
        
        sentiment_1.subheader("No. of tweets")
        bar_chart_dict={}
        bar_chart_dict['Negative']= count_of_neg_tweet
        bar_chart_dict['Neutral']= count_of_neu_tweet
        bar_chart_dict['Positive']= count_of_pos_tweet
        index= ['P&G']
        bar_chart_df = pd.DataFrame(data=bar_chart_dict, index=index);
        bar_chart_df.plot.bar(rot=0, color="ryg" );
        sentiment_1.pyplot() 
        
        sentiment_2.subheader("Positve to Negative Ratio")
        labels = ['Negative', 'Positive']
        sizes = [df_keyword['Negative'].mean(), df_keyword['Positive'].mean()]
        #colors
        colors = ['red','green']
         
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
        #draw circle
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')  
        plt.tight_layout()
        #plt.title("Positve to Negative Ratio")
        sentiment_2.pyplot() 
        
        
        sentiment_3.subheader("Word Cloud")
        #df_neu= df_keyword.copy() #[df_keyword['Overall']== "Neutral"]
        text_tweet_list= df_keyword['Tweet'].values.tolist()
        text_tweet_str=  ' , '.join([str(elem) for elem in text_tweet_list])
        text_tweet_str= text_tweet_str.replace('https', ' ')
        
        wordcloud = WordCloud(width=400, height=300, background_color='white',max_font_size = 50,collocations=False).generate(text_tweet_str)
        
        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        sentiment_3.pyplot()
        
        
        
        sentiment_chart1, sentiment_chart2= st.beta_columns((1,1))
        
        df_timelapse= df_keyword.reset_index()
        df_timelapse= df_timelapse[['Created At', 'Negative', 'Positive']]
        df_timelapse= df_timelapse.groupby(['Created At'])["Negative", "Positive"].apply(lambda x : x.astype(float).mean())
        #st.write(df_timelapse)
        #df_timelapse= df_timelapse.set_index('Created At')
        df_timelapse['Overall'] = df_timelapse['Positive'] - df_timelapse["Negative"]
        df_timelapse['Negative'] *= -1
        sentiment_chart1.subheader("Sentiment over the period")
        sentiment_chart1.line_chart(df_timelapse[['Positive', 'Negative', 'Overall']], width=300)
        
        
        
        i_sentiment= sentiment_chart2.selectbox("Filter sentiment", ("All", 'Positive', 'Negative'))
        if i_sentiment == 'All':
            sentiment_chart2.write(df_keyword[['Tweet']])
        elif i_sentiment == "Positive":
            df_senti= df_keyword[df_keyword['Overall']== 'Positive']
            sentiment_chart2.write(df_senti[['Tweet']])
        elif i_sentiment == "Negative":
            df_senti= df_keyword[df_keyword['Overall']== 'Negative']
            sentiment_chart2.write(df_senti[['Tweet']])
        
    
    else:
        st.warning("Please select at least one Category")
            

############################################################### Google Trends ############################################    
else:
    with st.beta_expander("Refresh"):
        update_col1, update_col2, update_col3, update_col4 ,update_col5= st.beta_columns(5)
        
        
        today = datetime.date.today()
        yesterday = today + datetime.timedelta(days=-365*3)
    
    
        password= update_col1.text_input("Password", type="password")
        since_date = update_col2.date_input('Since', yesterday)
        until_date = update_col3.date_input('Until', today)
        delta= until_date- since_date
        no_of_days= delta.days
        
        location= update_col4.text_input("Location", 'IN')
        
        update_col5.markdown("### ")
        if update_col5.button("Refresh"):
            if password== 'Tweepy@007':
                
                dict_pytrends={}
                df_all_region= pd.DataFrame()
                my_bar = st.progress(0)
                i_counter=0
                for search_words in product_list_google:
                   
                    frame= str(since_date) + ' ' + str(until_date)
                    pytrends.build_payload([search_words], timeframe=frame, geo='IN')
                    df_pytrends= pytrends.interest_over_time()
                    df_pytrends= df_pytrends[[search_words]]
                    df_pytrends= df_pytrends.reset_index()
                    df_pytrends['date']= pd.to_datetime(df_pytrends['date']).dt.date
                    dict_pytrends[search_words] = df_pytrends[search_words].values.tolist()
                    #st.write(df_pytrends)
                    
                   
                    df_region= pytrends.interest_by_region(resolution='REGION',inc_geo_code=True,inc_low_vol=False)
                    df_region.geoCode = df_region.geoCode.str[-2:]
                    df_region = df_region.reset_index()#.set_index(['geoName','geoCode'])
                    
                    df_all_region['geoName']= df_region['geoName'].values.tolist()
                    df_all_region['geoCode']= df_region['geoCode'].values.tolist()
                    df_all_region[search_words]= df_region[search_words].values.tolist()
                    
                    time.sleep(5)
                    my_bar.progress((i_counter + 1)/ len(product_list_google))
                    i_counter+=1
                    
                dict_pytrends['Date'] = df_pytrends['date'].values.tolist()
                df_pytrends_all= pd.DataFrame(dict_pytrends)
                
                
                df_pytrends_all=df_pytrends_all.set_index("Date")
                df_pytrends_all.to_csv("GoogleTrendsNoCategory.csv", index=True)
                
                
                
                df_pytrends_final= pd.DataFrame(columns=['Date', 'Trend', 'Keyword', 'Category'])
                for i_col  in df_pytrends_all.columns:
                    df_temp= df_pytrends_all[[i_col]]
                    df_temp.columns= ['Trend']
                    df_temp['Keyword']= i_col
                    df_temp['Category'] = dict_pg_trends[i_col]
                    df_temp= df_temp.reset_index()
                    #st.write(df_temp)
                    df_pytrends_final=df_pytrends_final.append(df_temp)
                    
                
                df_pytrends_final.to_csv("GoogleTrends.csv", index=False)
                df_all_region.to_csv("GoogleTrendsRegion.csv", index= False)
                
                #st.write(df_all_region)

###########################################################################################################
    df_pytrends= pd.read_csv("GoogleTrends.csv")
    df_pytrends_nocat= pd.read_csv('GoogleTrendsNoCategory.csv')
    df_pytrends_region= pd.read_csv('GoogleTrendsRegion.csv')
    df_pytrends_region= df_pytrends_region.set_index('geoName')
    df_pytrends_nocat= df_pytrends_nocat.replace(0, 1)
    df_pytrends['Date']= pd.to_datetime(df_pytrends['Date']).dt.date
    
    df_pytrends_nocat['Date']= pd.to_datetime(df_pytrends_nocat['Date']).dt.date
    df_pytrends_nocat= df_pytrends_nocat.set_index('Date')
    
    st.sidebar.markdown("### Select Categories")
    filter_category=[]
    select_all = st.sidebar.checkbox("Select All", True)
    
    if select_all:
        for i_category in df_pytrends['Category'].unique():
            if st.sidebar.checkbox(i_category, True, key= "google_select"):
                filter_category.append(i_category)
    else:
        for i_category in df_pytrends['Category'].unique():
            if st.sidebar.checkbox(i_category, False):
                filter_category.append(i_category)
                
            
    if len(filter_category) >0:
        
        df_category= df_pytrends[df_pytrends['Category'].isin(filter_category)]   
        
        filter_1, filter_2, filter_3, filter_4= st.beta_columns(4)
        filter_keywords= filter_1.multiselect("Filter Keywords", df_category['Keyword'].unique())
        
        #print(df_category['Created At'].min(), df_category['Created At'].max())
        since_date = filter_2.date_input('From', df_category['Date'].min(), key='date_since')
        until_date = filter_3.date_input('Until', df_category['Date'].max(), key= 'date_till')
        frequency= filter_4.selectbox('Frequency', ['Weekly', 'Monthly', 'Yearly'])
        
        
        st.markdown('''---''')
        if len(filter_keywords) >0:
            df_keyword= df_category[df_category['Keyword'].isin(filter_keywords)]
            
        else:
            df_keyword= df_category.copy()
            
        
        df_pytrends_region= df_pytrends_region[df_keyword['Keyword'].unique()]    
        
        df_keyword= df_keyword.set_index("Date")
        df_keyword= df_keyword[(df_keyword.index >=since_date ) & (df_keyword.index <= until_date)]
        
        filtered_keywords= df_keyword['Keyword'].unique().tolist()
        
        
        df_plot= df_pytrends_nocat[filtered_keywords]
        df_plot= df_plot[(df_plot.index >=since_date ) & (df_plot.index <= until_date)]
        
        if frequency == "Weekly":
            df_plot = df_plot
        elif frequency == 'Monthly':
            df_plot.index = pd.to_datetime(df_plot.index)
            df_plot= df_plot.resample('1M').mean()

        elif frequency == "Yearly":
            df_plot.index = pd.to_datetime(df_plot.index)
            df_plot= df_plot.resample('1Y').mean()
            
            
        
        
        
        
        
        st.markdown("### Google trend line")
        st.line_chart(df_plot)
        
        trend_chart1, trend_chart2= st.beta_columns((2,1))
        #st.write(df_plot)
        
        df_change= (df_plot.iloc[-1] - df_plot.iloc[0])/ df_plot.iloc[0] * 100
        df_change.columns= ['Change Percentage']
        
        trend_chart2.markdown("### % change during the period")
        trend_chart2.write(df_change)
        
        trend_chart1.markdown("### Graph showing % change in the trend")
        trend_chart1.bar_chart(df_change)
        
       
        trend_chart1.markdown("### Heatmap showing relative number of searches across the period")
        sns.heatmap(df_plot, cmap ='RdYlGn', linewidths = 0.30, annot = False)
        trend_chart1.pyplot() 
        
        
       
        st.markdown("### Region wise trend distribution")
        st.bar_chart(df_pytrends_region)
        
        
        
        ######################################## PLot MAP ########################################
        
        
        df_map= df_pytrends_region.copy()
        #st.write(df_map)
        df_latitude= pd.read_csv("regional_latitude_india.csv")
        
        
        map_values= df_map.sum(axis = 1, skipna = True).values.tolist()
        df_map['Values']= map_values
        df_map= df_map[['Values']]
        df_map= df_map.reset_index()
        df_map.columns= ['state', 'Values']
        
        df_map= pd.merge(df_map, df_latitude, on='state')
        #df_map= df_map.drop(["state", 'longitude', 'latitude'], axis=1)
        df_map= df_map.drop(['id', 'longitude', 'latitude'], axis=1)
        #df_map= df_map.set_index("id")
        #df_map['id'] = range(1, len(df_map) + 1)
        
        
        
        #st.write(df_map)
        
        fig = px.choropleth(
            df_map,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations='state',
            color='Values',
            color_continuous_scale='Greens'
        )

        fig.update_geos(fitbounds="locations", visible=False)

        #fig.show()
        st.markdown("### Map distribution")
        st.write(fig)
        
        
        
        
        #states = alt.topo_feature(data.us_10m.url, feature='states')
        
        #st.write(states)
        #st.write(type(states))
        
        #gdf = gpd.read_file('Admin2.shp')
        #st.write(gdf)
        
        
        
        #c= alt.Chart(states).mark_geoshape().encode(
        #    color='Values:Q'
        #).transform_lookup(
        #    lookup='id',
         #   from_=alt.LookupData(df_map, 'id', ['Values'])
        #).project('albersUsa')
        #st.map(df_map)
        #trend_chart2.markdown("### Map distribution")
        #trend_chart2.write(c)
        
        #states = alt.topo_feature(data.us_10m.url, feature='states')
        #pop = data.population_engineers_hurricanes()
        
        # c=alt.Chart(states).mark_geoshape().encode(
        #     color='Values:Q'
        # ).transform_lookup(
        #     lookup='state',
        #     from_=alt.LookupData(df_map, 'state', list(df_map.columns))
        # ).properties(
        #     width=500,
        #     height=300
        # ).project(
        #     type='albersUsa'
        # )
            
        # data = pd.DataFrame({
        #     'awesome cities' : ['Chicago', 'Minneapolis', 'Louisville', 'Topeka'],
        #     'lat' : [41.868171, 44.979840,  38.257972, 39.030575],
        #     'lon' : [-87.667458, -93.272474, -85.765187,  -95.702548]
        # })
        
        # # Adding code so we can have map default to the center of the data
        # DATE_COLUMN = 'date/time'
        # DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
        #             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
        
        # @st.cache
        # def load_data(nrows):
        #     data = pd.read_csv(DATA_URL, nrows=nrows)
        #     lowercase = lambda x: str(x).lower()
        #     data.rename(lowercase, axis='columns', inplace=True)
        #     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        #     return data
        
        # data_load_state = st.text('Loading data...')
        # data = load_data(10000)
        # data_load_state.text("Done! (using st.cache)")
        
        # if st.checkbox('Show raw data'):
        #     st.subheader('Raw data')
        #     st.write(data)
        
        # st.subheader('Number of pickups by hour')
        # hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
        # st.bar_chart(hist_values)
        
        # # Some number in the range 0-23
        # hour_to_filter = st.slider('hour', 0, 23, 17)
        # filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
        
        # st.subheader('Map of all pickups at %s:00' % hour_to_filter)
        # airports = pd.read_csv(data.airports.url)
        # st.write(airports.head(10))
        # airports = airports.groupby('state').agg({'iata': 'count'})
        # st.write(airports.head(10))
        # airports['id'] = range(1, len(airports) + 1)
        # airports['pct'] = airports['iata'] / airports['iata'].sum()
        # st.write(airports.head(10))
        # states = alt.topo_feature(data.us_10m.url, feature='states')
        
        # c= alt.Chart(states).mark_geoshape().encode(
        #     color='pct:Q'
        # ).transform_lookup(
        #     lookup='id',
        #     from_=alt.LookupData(airports, 'id', ['pct'])
        # ).project('albersUsa')
        # #st.map(df_map)
        # st.write(c)
                        
        

    else:
        st.warning("Please select at least one Category")
                    
         
            
