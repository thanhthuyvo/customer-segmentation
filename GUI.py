import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
from importlib import reload
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import altair as alt
import pickle
import plotly.figure_factory as ff
import time
import warnings

from st_aggrid import AgGrid,GridUpdateMode,DataReturnMode, JsCode,ColumnsAutoSizeMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
warnings.filterwarnings("ignore")

import streamlit as st

def convert_date(df):
    string_to_date = lambda x : datetime.strptime(str(x), "%Y%m%d").date()
    data['date'] = data['date'].apply(string_to_date)
    data['date'] = data['date'].astype('datetime64[ns]')


def cluster_function_k5(prediction):
    if prediction ==0:
        return "Almost lost"
    elif prediction ==1:
        return "Lost"
    elif prediction ==2:
        return "Star"
    elif prediction ==3:
        return "Regular"
    return "New"

def cluster_function_k6(prediction):
    if prediction ==5:
        return "Star"
    elif prediction ==2:
        return "Big Spender"
    elif prediction ==3:
        return "Cooling Down"
    elif prediction ==1:
        return "Loyal"
    elif prediction ==0:
        return "Regular"
    return "Lost Cheap"

import random
def new_table_key():
    mykey = 'key_' + str(random.randrange(100000))
    st.session_state['mykey'] = mykey

condition = st.sidebar.selectbox(
    "Select the visualization",
    ("Introduction", "EDA", "KMeans Clustering","New Prediction")
)


def visual_chart(data_RFM):
    width = st.sidebar.slider("Plot width", 1, 25, 12)
    height = st.sidebar.slider("Plot height", 1, 25, 6)

    
    

    original_title_null = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Information about each customer\'s clustering</b></p>'
    st.markdown(original_title_null,unsafe_allow_html=True)


    rfm_agg = data_RFM.groupby('type').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'count']}).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)


    rfm_agg = rfm_agg.reset_index()

    st.dataframe(rfm_agg)

    original_title_null = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Data Visualization</b></p>'
    st.markdown(original_title_null,unsafe_allow_html=True)

    # col1,col2=st.columns(2)

    fig_pie, ax = plt.subplots(figsize=(width, height))
    plt.pie(df_predict.type.value_counts(),labels=df_predict.type.value_counts().index,autopct='%.0f%%')
    ax.legend()
    plt.title("Percentage of each customer segment",fontweight="bold",fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_pie)

    

    #fig = plt.gcf()
    fig_treemap=plt.figure(figsize=(width, height))
    ax = fig_treemap.add_subplot()
    fig_treemap.patch.set_visible(False)
    ax.axis('off')
    fig_treemap.set_size_inches(14, 10)

    colors_dict = {'ACTIVE':'yellow','BIG SPENDER':'royalblue', 'LIGHT':'cyan',
                 'LOST Cheap':'red', 'LOYAL':'purple', 'POTENTIAL':'green', 'STARS':'gold'}

    squarify.plot(sizes=rfm_agg['Count'],
    text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                for i in range(0, len(rfm_agg))], alpha=0.5 )


    plt.title("Customers Segments",fontsize=26,fontweight="bold",y=1.05)
    st.pyplot(fig_treemap)



    fig_scaterplot=px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="type",
            hover_name="type", size_max=100)
    st.plotly_chart(fig_scaterplot, use_container_width=True)

# ------------- Introduction ------------------------

if condition == 'Introduction':
    #st.image(os.path.join(os.path.abspath(''), 'data', 'dataset-cover.jpg'))
    st.subheader('About')

    st.image("Data/RFM.jpeg")
    
    ## FALTA O CHECK ON GITHUB
    st.write("""
    This application provides the entire purchase history up to the end of June 1998 
    of the cohort of 23,570 individuals who made their first-ever purchase at CDNOW in the first quarter of 1997.
    
    The prediction are made by dividing customers into different segments based on the RFM (Recency-Frequency-Monetary) score.
    """)

    st.subheader('Overview about RFM Segmentation')
    
    st.write("""
    RFM analysis is a method for segmenting customer behavior based on data.
    RFM stands for recency, frequency, and monetary value.


    Customers will be divided into groups according to when they last made a purchase, how frequently they've made purchases in the past, and how much money they've spent altogether. These three variables have all shown to be reliable indicators of a  customer's willingness to engage in marketing messages and offers.
    
    To begin, I would conduct RFM analysis to obtain the desired values and divide customers into various groups based on our experience and specific field. Those features will be used as an input in K-means, to determine similarity and we can segment customers into different clusters. K-Means uses Euclidean distance as a distance metric to calculate the distance between the data points

    """)

    st.image("Data/Segments_clustering.png")
# ------------- EDA ------------------------

elif condition == 'EDA':
    data = pd.read_csv("Data/CDNOW_master.txt", delim_whitespace=True, header = None, names = ['customer_id', 'date', 'number_of_cds', 'dollar_value'])
    des=data.describe()
    inf=data.info()

    string_to_date = lambda x : datetime.strptime(str(x), "%Y%m%d").date()
    data['date'] = data['date'].apply(string_to_date)
    data['date'] = data['date'].astype('datetime64[ns]')

    data=data.loc[data["dollar_value"]>0]

    original_title_data = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Data</b></p>'
    st.markdown(original_title_data,unsafe_allow_html=True)
    st.dataframe(data)

    original_title_describe = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Descriptive Statistics</b></p>'
    #st.subheader("Describe")
    st.markdown(original_title_describe,unsafe_allow_html=True)
    st.write(des)

    #st.metric("Shape",str(data.shape))

    original_title_null = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Check null and duplicated data</b></p>'
    st.markdown(original_title_null,unsafe_allow_html=True)
    data_check_null=data.isnull().sum().to_frame('counts')
    st.write(data_check_null)

    col1, col2 = st.columns(2)

    st.markdown("""
        <style>
        div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: rgb(30, 103, 119);
        overflow-wrap: break-word;
        }

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: blue;
        size=25
        }
        </style>
        """
    , unsafe_allow_html=True)
    


    col1.metric("Transactions timeframe from",str(data['date'].min().date()))
    col2.metric("To", str(data['date'].max().date()))

    col1.metric("Transactions don\'t have a customer id",data[data.customer_id.isnull()].shape[0])
    col2.metric("Unique customer_id",len(data.customer_id.unique()))

    #st.write(('{:,} transactions don\'t have a customer id'.format()))
    #st.write('{:,} unique customer_id'.format(len(data.customer_id.unique())))


    ###Biểu đồ
    original_title_describe = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Visulization</b></p>'
    most_customer=data.groupby('customer_id').size().reset_index()
    most_customer.columns = ['customer_id', 'frequency']
 
    most_customer_frequecy=most_customer.sort_values("frequency",ascending=False).head(10)

    fig_most_customer=plt.figure(figsize=(11, 7))
    sns.barplot(x="customer_id", y="frequency", data=most_customer_frequecy, palette="flare_r",order=most_customer_frequecy["customer_id"])
    plt.title("Top 10 customer with high frequency of spending",fontsize=20,color="blue",y=1.05)
    st.pyplot(fig_most_customer)

    most_customer_monetery=data.groupby('customer_id').sum().reset_index()
    most_customer_monetery=most_customer_monetery.sort_values("dollar_value",ascending=False).head(10)
    fig_most_customer_monetery=plt.figure(figsize=(11, 7))
    sns.barplot(x="customer_id", y="dollar_value", data=most_customer_monetery, palette="flare_r",order=most_customer_monetery["customer_id"])
    plt.title("Top 10 customer spending the most",fontsize=20,color="blue",y=1.05)
    st.pyplot(fig_most_customer_monetery)


    
elif condition == 'KMeans Clustering':
    model_k_5 = joblib.load("Data/Project_3_KMeans_k5_sklearn.joblib")
    model_k_6 = joblib.load("Data/Project_3_KMeans_k6_sklearn.joblib")

    sel_col, disp_col=st.columns(2)

    list_Model=["K = 5","K = 6"]
    select_model_k = st.sidebar.selectbox(
        'Select the Model with ',
        [i for i in list_Model]  
    )

    original_title_scatter = '<p style="color:Blue; font-size: 30px; text-align: left;"><b>Since kmean is sensitive to distance, we need to check the deviation of the dataset</b></p>'
    st.markdown(original_title_scatter,unsafe_allow_html=True)

    code_Kmeans_model ='''
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column])
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return
        '''
    
    st.code(code_Kmeans_model,language="python")

    code_plt_check_skew ='''
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

RFM_Table_scaled = np.log(RFM_Table_scaled+1)
plt.figure(figsize=(9, 9))
plt.subplot(3, 1, 1)
check_skew(RFM_Table_scaled,'Recency')
plt.subplot(3, 1, 2)
check_skew(RFM_Table_scaled,'Frequency')
plt.subplot(3, 1, 3)
check_skew(RFM_Table_scaled,'Monetary')
plt.tight_layout()
        '''
    
    st.code(code_plt_check_skew,language="python")
    st.image("Data/checkskew.png")


    original_title_scatter = '<p style="color:Blue; font-size: 30px; text-align: left;"><b>Apply StandardScaler to normalize data</b></p>'
    st.markdown(original_title_scatter,unsafe_allow_html=True)

    code_scale_data ='''
scaler = StandardScaler()
scaler.fit(RFM_Table_scaled)
RFM_Table_scaled = scaler.transform(RFM_Table_scaled)
        '''
    
    st.code(code_scale_data,language="python")

    original_title_scatter = '<p style="color:Blue; font-size: 30px; text-align: left;"><b>Plot elbow method to select which k is the best fit</b></p>'
    st.markdown(original_title_scatter,unsafe_allow_html=True)

    code_elbow ='''
from sklearn.cluster import KMeans
sse = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(RFM_Table_scaled)
    sse[k] = kmeans.inertia_

plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.savefig('elbow_method.png')
plt.show()
        '''
    
    st.code(code_elbow,language="python")




    st.image("Data/elbow_method.png")
    original_title_null = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Based on this Elbow method figure, we can consider running a Kmeans model with </b></p>'
    st.markdown(original_title_null,unsafe_allow_html=True)
    if select_model_k=="K = 5":

        code_k5 ='''
model_Kmeans_5 = KMeans(n_clusters=5, random_state=42)
model_Kmeans_5.fit(RFM_Table_scaled)
model_Kmeans_5.labels_.shape
        '''
    
        st.code(code_k5,language="python")
        
        rfm_agg_kmeans_k_5=pd.read_csv("Data/rfm_agg_kmeans_k_5.csv", index_col=0)

        st.write(rfm_agg_kmeans_k_5)

    
        st.image("Data/piechart_k5.png")

        st.image("Data/Unsupervised_Segments_5.png")

        #fig=plt.figure(figsize=(8, 6))

        original_title_scatter = '<p style=" color:Black; font-size: 25px; text-align: center;"><b>Scatter Plot</b></p>'
        st.markdown(original_title_scatter,unsafe_allow_html=True)
        fig=px.scatter_3d(rfm_agg_kmeans_k_5, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',color = 'type', opacity=0.3)
        fig.update_traces(marker=dict(size=20),selector=dict(mode='markers'))
        #fig.title("Top 10 customer with high frequency of spending",fontsize=20,color="blue",y=1.05)
        st.plotly_chart(fig, use_container_width=True)

    elif select_model_k=="K = 6":
        code_k5 ='''
model_Kmeans_5 = KMeans(n_clusters=5, random_state=42)
model_Kmeans_5.fit(RFM_Table_scaled)
model_Kmeans_5.labels_.shape
        '''
    
        st.code(code_k5,language="python")
        rfm_agg_kmeans_k_6=pd.read_csv("Data/rfm_agg_kmeans_k_6.csv", index_col=0)

        st.write(rfm_agg_kmeans_k_6)

    
        st.image("Data/piechart_k6.png")

        st.image("Data/Unsupervised_Segments_6.png")

        original_title_scatter = '<p style="color:Black; font-size: 25px; text-align: center;"><b>Scatter Plot</b></p>'
        st.markdown(original_title_scatter,unsafe_allow_html=True)

        fig=px.scatter_3d(rfm_agg_kmeans_k_6, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',color = 'type', opacity=0.3)
        fig.update_traces(marker=dict(size=20),selector=dict(mode='markers'))
        #fig.title("Top 10 customer with high frequency of spending",fontsize=20,color="blue",y=1.05)
        st.plotly_chart(fig, use_container_width=True)



    

    # fig, ax = plt.subplots()
    # ax.hist(rfm_agg_kmeans_k_5)
    # st.pyplot(fig)


elif condition == 'New Prediction':

    model_k_5 = joblib.load("Data/Project_3_KMeans_k5_sklearn.joblib")
    model_k_6 = joblib.load("Data/Project_3_KMeans_k6_sklearn.joblib")


    flag = False
    lines = None

    

    list_Model=["K = 5","K = 6"]
    select_model_k = st.sidebar.selectbox(
        'Select the Model with ',
        [i for i in list_Model]  
    )

    
    original_title_null = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Here is the prediction result with K= </b></p>'
    st.markdown(original_title_null,unsafe_allow_html=True)


    list_type_data=["With Raw Data","With RFM Format"]
    select_type_data = st.sidebar.selectbox(
        'Select the Model with ',
        [i for i in list_type_data]  
    )

    if select_type_data=="With Raw Data":

        type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
        if type=="Upload":
            # Upload file
            uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
            if uploaded_file_1 is not None:
                #data = pd.read_csv(uploaded_file_1)

                check_type=str(uploaded_file_1.type)

                if check_type=="text/csv":
                    data = pd.read_csv(uploaded_file_1,encoding= 'unicode_escape')
                    data.rename(columns={ data.columns[1]: "customer_id",data.columns[1]: "date",data.columns[1]: "number_of_cds", data.columns[1]: "dollar_value"}, inplace = True)
                elif check_type=="text/plain":
                    data = pd.read_csv(uploaded_file_1, delim_whitespace=True, header = None, names = ['customer_id', 'date', 'number_of_cds', 'dollar_value'])

                
                original_title_null = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Raw Data</b></p>'
                st.markdown(original_title_null,unsafe_allow_html=True)
                st.dataframe(data)

                string_to_date = lambda x : datetime.strptime(str(x), "%Y%m%d").date()
                data['date'] = data['date'].apply(string_to_date)
                data['date'] = data['date'].astype('datetime64[ns]')

                max_date = data['date'].max().date()
                Recency = lambda x : (max_date - x.max().date()).days
                Monetary = lambda x : round(sum(x), 2)
                df_predict = data.groupby('customer_id').agg({'date': Recency,
                                                        "customer_id": ["count"],  
                                                        "dollar_value": Monetary })
                df_predict.columns=["Recency","Frequency","Monetary"]
                df_predict = df_predict.sort_values('Monetary', ascending=False)

                #df_predict = df_predict[['Recency','Frequency','Monetary']]
                
                st.dataframe(df_predict)
                # st.write(lines.columns)
                flag = True       
        if type=="Input":        
            sel_col, disp_col=st.columns(2)
            check_stop=1

            original_title_note = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Note:</b></p>'
            st.markdown(original_title_note,unsafe_allow_html=True)
            
            text_="""
            - Here is an example of input data.
            - To change the values\'s cell, Click on it and type your expected value
            - To add another row, Click Add
            - To remove a row, Click Delete
            - After all, tick the checkbox to select how many row you wanna use as input
            """            
            st.markdown(text_,unsafe_allow_html=True)
        

            DATA = {
                'customer_id': ['001', '002', '003','004','005'],
                'date': ['20220528', '20220225', '20220724','20230228','20230228'],
                'number_of_cds': ['4', '10', '50',"3","15"],
                'dollar_value': ['500', '1000', '20',"300","550"]
            }

            df=pd.DataFrame(DATA)

            #df = pd.read_csv("Data/CDNOW_sample.txt", delim_whitespace=True, header = None, names = ['customer_id', 'date', 'number_of_cds', 'dollar_value'])

            string_to_add_row = "\n\n function(e) { \n \
            let api = e.api; \n \
            let rowIndex = e.rowIndex + 1; \n \
            api.applyTransaction({addIndex: rowIndex, add: [{}]}); \n \
                }; \n \n"
            
            cell_button_add = JsCode('''
                class BtnAddCellRenderer {
                    init(params) {
                        this.params = params;
                        this.eGui = document.createElement('div');
                        this.eGui.innerHTML = `
                        <span>
                            <style>
                            .btn_add {
                            background-color: limegreen;
                            border: none;
                            color: white;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 10px;
                            font-weight: bold;
                            height: 2.5em;
                            width: 8em;
                            cursor: pointer;
                            }

                            .btn_add :hover {
                            background-color: #05d588;
                            }
                            </style>
                            <button id='click-button' 
                                class="btn_add" 
                                >&CirclePlus; Add</button>
                        </span>
                    `;
                    }

                    getGui() {
                        return this.eGui;
                    }

                };
                ''')
            
            string_to_delete = "\n\n function(e) { \n \
                let api = e.api; \n \
                let sel = api.getSelectedRows(); \n \
                api.applyTransaction({remove: sel}); \n \
                    };\n\n"
            
            cell_button_delete = JsCode('''
            class BtnCellRenderer {
                init(params) {
                    console.log(params.api.getSelectedRows());
                    this.params = params;
                    this.eGui = document.createElement('div');
                    this.eGui.innerHTML = `
                    <span>
                        <style>
                        .btn {
                        background-color: #F94721;
                        border: none;
                        color: white;
                        font-size: 10px;
                        font-weight: bold;
                        height: 2.5em;
                        width: 8em;
                        cursor: pointer;
                        }

                        .btn:hover {
                        background-color: #FB6747;
                        }
                        </style>
                        <button id='click-button'
                            class="btn"
                            >&#128465; Delete</button>
                    </span>
                `;
                }

                getGui() {
                    return this.eGui;
                }

            };
            ''')

            gd = GridOptionsBuilder.from_dataframe(df)
            gd.configure_pagination(enabled=True)
            gd.configure_default_column(editable=True, groupable=True)
            gd.configure_column('', headerTooltip='Click on Button to add new row', editable=False, filter=False,
                            onCellClicked=JsCode(string_to_add_row), cellRenderer=cell_button_add,
                            autoHeight=True, wrapText=True, lockPosition='left')
            gd.configure_column('Delete', headerTooltip='Click on Button to remove row',
                                    editable=False, filter=False, onCellClicked=JsCode(string_to_delete),
                                    cellRenderer=cell_button_delete,
                                    autoHeight=True, suppressMovable='true')


            sel_mode = 'multiple'
            gd.configure_selection(selection_mode=sel_mode, use_checkbox=True)
            gridoptions = gd.build()
            grid_table = AgGrid(df, gridOptions=gridoptions,
                                update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
                                height=500,
                                allow_unsafe_jscode=True,enable_enterprise_modules=True,data_return_mode=DataReturnMode.AS_INPUT,
                                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS
                                )

            sel_row = grid_table["selected_rows"]
            
            st.subheader("Created Data")
            if sel_row!=[]:
                df_selected = pd.DataFrame(sel_row)
                data=df_selected[["customer_id","date","number_of_cds","dollar_value"]]
                st.dataframe(data)

                # data["number_of_cds"] = pd.to_numeric(df["number_of_cds"])
                # data["date"] = pd.to_numeric(df["date"])
                # data["dollar_value"] = pd.to_numeric(df["dollar_value"])

                data[['dollar_value']] = data[['dollar_value']].astype(float)
                data[['date',"number_of_cds"]] = data[['date',"number_of_cds"]].astype(int)


                most_customer=data.groupby('customer_id').size().reset_index()
                most_customer.columns = ['customer_id', 'frequency']
                fig_most_customer=plt.figure(figsize=(11, 7))
                sns.barplot(x="customer_id", y="frequency", data=most_customer.sort_values("frequency",ascending=False).head(10), palette="Blues_d")
                plt.title("Top 10 customer with high frequency of spending")
                st.pyplot(fig_most_customer)


                most_customer_monetery=data.groupby('customer_id').sum().reset_index()
                fig_most_customer_monetery=plt.figure(figsize=(11, 7))
                sns.barplot(x="customer_id", y="dollar_value", data=most_customer_monetery.sort_values("dollar_value",ascending=False).head(10), palette="Blues_d")
                plt.title("Top 10 customer spending the most")
                st.pyplot(fig_most_customer_monetery)

                #df[['number_of_cds', 'date',"dollar_value"]] = df[['two', 'three']].astype(float)


                string_to_date = lambda x : datetime.strptime(str(x), "%Y%m%d").date()
                data['date'] = data['date'].apply(string_to_date)
                data['date'] = data['date'].astype('datetime64[ns]')

                max_date = data['date'].max().date()
                Recency = lambda x : (max_date - x.max().date()).days
                Monetary = lambda x : round(sum(x), 2)
                df_predict = data.groupby('customer_id').agg({'date': Recency,
                                                        "customer_id": ["count"],  
                                                        "dollar_value": Monetary })
                df_predict.columns=["Recency","Frequency","Monetary"]
                df_predict = df_predict.sort_values('Monetary', ascending=False)

                #df_predict = df_predict[['Recency','Frequency','Monetary']]
                st.subheader("Convert to RFM Data")
                st.dataframe(df_predict)
                # st.write(lines.columns)
                flag = True  


            # df = get_data()

            # st.write('### Init')
            # st.dataframe(df, height=150)

            # st.write('### Data editor')
            # st.experimental_data_editor(df,num_rows='dynamic')

            # st.write('### Update')
            # st.dataframe(update, height=150)

            # favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
            # st.markdown(f"Your favorite command is **{favorite_command}** ")


            # if Recency_list!="" and Frequency_list!="" and Monetary_list!="":
            #     df_predict=pd.DataFrame({"Recency":Recency_list,"Frequency":Frequency_list,"Monetary":Monetary_list})
            #     flag = True   
            
            
            
            
        
        if flag:
            if type=="Upload":

                original_title_null = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Here is the result after Predict</b></p>'
                st.markdown(original_title_null,unsafe_allow_html=True)
                if select_model_k=="K = 5":     
                    predict=model_k_5.predict(df_predict)
                    df_predict["cluster"]=predict
                    df_predict["type"]=df_predict["cluster"].map(lambda x: (cluster_function_k5(x)))
                    st.dataframe(df_predict)
                    #st.write(predict)
                elif  select_model_k=="K = 6":
                    predict=model_k_6.predict(df_predict)
                    df_predict["cluster"]=predict
                    df_predict["type"]=df_predict["cluster"].map(lambda x: (cluster_function_k6(x)))
                    st.dataframe(df_predict)

            if type=="Input":
                original_title_null = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Here is the result after Predict</b></p>'
                st.markdown(original_title_null,unsafe_allow_html=True)
                if select_model_k=="K = 5":     
                    predict=model_k_5.predict(df_predict)
                    df_predict["cluster"]=predict
                    df_predict["type"]=df_predict["cluster"].map(lambda x: (cluster_function_k5(x)))
                    st.dataframe(df_predict)
                elif  select_model_k=="K = 6":
                    predict=model_k_6.predict(df_predict)
                    df_predict["cluster"]=predict
                    df_predict["type"]=df_predict["cluster"].map(lambda x: (cluster_function_k6(x)))
                    st.dataframe(df_predict)

            visual_chart(df_predict)
            




    elif select_type_data=="With RFM Format":

        type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
        if type=="Upload":
            # Upload file
            uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
            if uploaded_file_1 is not None:
                df_predict = pd.read_csv(uploaded_file_1)
                df_predict = df_predict[['Recency','Frequency','Monetary']]
                st.dataframe(df_predict)
                # st.write(lines.columns)
                flag = True      
                
        if type=="Input":        
            sel_col, disp_col=st.columns(2)

            check_stop=1

            Recency_list=[]
            Frequency_list=[]
            Monetary_list=[]

          
            Recency_txt=sel_col.text_input("Input Recency",key ="Recency")
            Frequency_txt=sel_col.text_input("Input Frequency",key = "Frequency")
            Monetary_txt=sel_col.text_input("Input Monetary",key = "Monetary")

            Recency_list.append(Recency_txt)
            Frequency_list.append(Frequency_txt)
            Monetary_list.append(Monetary_txt)
            

            if Recency_txt!="" and Frequency_txt!="" and Monetary_txt!="":
                df_predict=pd.DataFrame({"Recency":Recency_list,"Frequency":Frequency_list,"Monetary":Monetary_list})
                flag = True   
            
            
            
        
        if flag:
            if type=="Upload":
                original_title_null = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Here is the result</b></p>'
                st.markdown(original_title_null,unsafe_allow_html=True)
                if select_model_k=="K = 5":     
                    predict=model_k_5.predict(df_predict)
                    df_predict["cluster"]=predict
                    df_predict["type"]=df_predict["cluster"].map(lambda x: (cluster_function_k5(x)))
                    st.dataframe(df_predict)
                    #st.write(predict[11000])
                elif  select_model_k=="K = 6":
                    predict=model_k_6.predict(df_predict)
                    df_predict["cluster"]=predict
                    df_predict["type"]=df_predict["cluster"].map(lambda x: (cluster_function_k6(x)))
                    st.dataframe(df_predict)
                visual_chart(df_predict) 

            if type=="Input":
                if st.button('Predict', help='Be certain to check the parameters on the sidebar'):
                    if select_model_k=="K = 5":     
                        predict=model_k_5.predict(df_predict)
                        st.write(predict[0])
                        
                        if predict[0]==0:
                            st.success(f'The predicted value is Regular')
                        elif predict[0]==1:
                            st.success(f'The predicted value is Loyal')
                        elif predict[0]==2:
                            st.success(f'The predicted value is Star')
                        elif predict[0]==3:
                            st.success(f'The predicted value is Big Spender')
                        elif predict[0]==4: 
                            st.success(f'The predicted value is Lost Cheap')

                    elif  select_model_k=="K = 6":
                        predict=model_k_6.predict(df_predict)
                        st.write(predict[0])
                        if predict[0]==0:
                            st.success(f'The predicted value is Regular')
                        elif predict[0]==1:
                            st.success(f'The predicted value is Loyal')
                        elif predict[0]==2:
                            st.success(f'The predicted value is Big Spender')
                        elif predict[0]==3:
                            st.success(f'The predicted value is Cooling Down')
                        elif predict[0]==4: 
                            st.success(f'The predicted value is Lost Cheap')
                        elif predict[0]==5: 
                            st.success(f'The predicted value is Star')
        

        
        
        