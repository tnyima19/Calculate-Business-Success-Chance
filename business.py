
"""
Name:       Tenzing Nyima
Email:      TenzingNyima71@myhunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
Title:      Success Calulator of a Small Business.
URL:        https://tnyima19.github.io/success-calc/
"""
import pandas as pd
import numpy as np
import folium
from folium import plugins
import random
from haversine import haversine, Unit
import pickle
from math import sin, cos, sqrt, atan2,asin, radians
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn import svm


def open_business(filename):
    percentage = 0.50
    df = pd.read_csv(filename,nrows=9000, skiprows=lambda i: i>0 and random.random() > percentage)
    df = df.drop(columns={'Community Districts','Secondary Address Street Name', 'License Status','Address City', 'Address Street Name','DCA License Number',
                          'Borough Boundaries','City Council Districts','Police Precincts',
                          'Community Board','Council District','BIN','BBL','NTA','Census Tract', 'Detail',
                          'License Type','Contact Phone Number','Zip Codes', 'Location','Address Building',
                          'Business Name 2','Borough Code'})
    df = df[df['Address State'] == 'NY']
    df = df[df['Industry'] != 'Pedicab Driver']
    df = df[df['Industry'] != 'Sightseeing Guide']
    df = df[df['Industry'] != 'Electronics Store']
    df = df[df['Industry'] != 'Ticket Seller']
    df = df[df['Industry'] != 'Home Improvement Contractor']
    types_business = df['Industry'].unique() # types of business.
    business_interested = ['Laundries','General Vendor', 'Sidewalk Cafe','Stoop Line Stand','Tobacco Retail Dealer','Electronic Cigarette Dealer']
    df = df.loc[df['Industry'].isin(business_interested)]
    df = df.rename(columns={'Address ZIP':'Zip Codes'})
    df['Address Borough'] = df['Address Borough'].str.lower()
    df['Zip Codes'] = df['Zip Codes'].apply(lambda x: int(x))
    #print(df)
    return df

def open_stations(file_name):
    df = pd.read_csv(file_name)
    df = df.drop(columns={'URL','NOTES','OBJECTID'})
    latitude,longitude = split_longitude_latitude(df)
    df["Latitude"] = latitude
    df["Longitude"] = longitude
    df["Latitude"] = df["Latitude"].astype(float).fillna(np.nan)
    df["Longitude"] = df["Longitude"].astype(float).fillna(np.nan)
    df= df.drop(columns={'the_geom'})
    print(df)
    return df

def split_longitude_latitude(df):
    latitude = []
    longitude = []
    for num in df['the_geom']:
        num = num.replace('POINT','')
        num = num.replace('(', '')
        num = num.replace(')','')
        geo = num.split()
        lon = geo[0]
        lat = geo[1]
        latitude.append(lat)
        longitude.append(lon)

    df = df.drop(columns={'the_geom'})
    return latitude,longitude

def find_times_sq(df):
    df = df[df['NAME'].str.contains("Times Sq")].iloc[0]
    return df['Latitude'],df['Longitude']

def remove_na(df):
    df = df[df['Latitude'].notna()]
    df = df[df['Longitude'].notna()]
    return df

def get_age(df):
    df['License Expiration Date'] = pd.to_datetime(df['License Expiration Date'])
    df['License Creation Date'] = pd.to_datetime(df['License Creation Date'])
    df['Duration'] = (df['License Expiration Date'] - df['License Creation Date'])
    df['Duration'] = df['Duration']/np.timedelta64(1,'Y')
    df = df.drop(columns=['License Expiration Date', 'License Creation Date'])
    #df = df.drop(df.columns[[1]])
    return df

def haversine_(lat1,lon1, lat2, lon2):
    R= 3959.87433
    d_lat = radians(lat2-lat1)
    d_lon = radians(lon2-lon1)
    lat2 = radians(lat1)
    lat2=radians(lat2)
    a = sin(d_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(d_lon / 2)**2
    if a >=0:
        c = 2*asin(sqrt(a))
        return R * c
    else:
        a = a* -1
        c = 2*(asin(sqrt(a)))
        return -1*(R * c)
    
def get_distance(latitude1, longitude1, latitude2, longitude2):
    location_1 = (latitude1,longitude1)
    location_2 = (latitude2, longitude2)
    distance = haversine(location_1,location_2, unit='km')
    #distance=haversine(latitude1,longitude1,latitude2,longitude2)
    return distance

def distance_from_times_sq(df_b, lon='times_sq_long', lat='times_sq_lat', dis='distance_from_times_sq'):
    #print(df_b['Longitude'])
    #print(df_b['Latitude'])
    print("I am in times qu")
    distances = []
    lo = df_b['Longitude'].to_list()
    lt = df_b['Latitude'].to_list()
    t_lo = df_b[lon].iloc[1]
    t_lt= df_b[lat].iloc[1]
    for num in range(len(lo)):
        total = get_distance(lt[num],lo[num],t_lt,t_lo)
        distances.append(total)
    
    return distances

def encode_categorical_col(df, column):
    """Takes a column of categorical data and uses categorical encoding
    to create a dataframe with k-1 columns, where k is the number of
    different nominal values of column. Your function should create k
    columns, one for each value, labels by prefix concatenated with teh
    value. The columns hsould be sored and the Data frme restricted to the
    first k-1 columns retured."""
    df = df.drop(columns=['Business Name'])
    df = pd.get_dummies(df, drop_first=True)
    #print("i am in")
    #print(df)
    return df

def distance_from_nearest_station(df_b, df_s, lon='Longitude', lat='Latitude', dis='Distance_from_station'):
    bus_long = df_b[lon].tolist()
    bus_lat = df_b[lat].tolist()
    #print(df_s[lon])
    #print(df_s[lat])
    subway_long = df_s['Longitude'].tolist()
    subway_lat = df_s['Latitude'].tolist()

    mininum_distances = []
    for num in range(len(bus_long)):
        #print(bus_long[num])
        min_distance = 999999999999
        #print(bus_lat[num])
        for num2 in range(len(subway_long)):
            min_distance = min(min_distance, get_distance(bus_lat[num], bus_long[num],subway_lat[num2], subway_long[num2]))
            #print(min_distance)
        mininum_distances.append(min_distance)

    df_b[dis] = mininum_distances
    return df_b

def split_test(df_x, dfy):
    return train_test_split(df_x, dfy)

def fit_poly(xes,yes,epsilon=100, verbose=False):
    """
    ADD IN DOCSTRING
    """
    error = 2*epsilon
    deg = 0
    print("POLYNOMICAL FEAUTERS (LINEAR REGRESSION)")
    while (error > epsilon) and deg < 4:
        deg = deg+1
        transformer = PolynomialFeatures(degree=deg)
        x_poly = transformer.fit_transform(xes)
        clf = LinearRegression(fit_intercept=False)
        clf.fit(x_poly, yes)
        pred_poly = clf.predict(x_poly)
        error = mean_squared_error(pred_poly, yes)
        new_y_pred = [ ]
        for num in pred_poly:
            if num >= 0.5:
                new_y_pred.append(1)
            else:
                new_y_pred.append(0)
        #print("mse:", error)
        print("accuracy:", accuracy_score(yes, new_y_pred))
        if verbose:
            print(f'MSE cost for deg {deg} poly model: {error:.3f}')
    if deg == 4:
        return None
    return deg

def fit_lasso_with_poly(xes,yes,poly_deg = 2):
    transformer = PolynomialFeatures(degree= poly_deg) #Easy way to create new columns when you enter a degree 
    #creates a second and third column 
    #USE L1 OR L2 , shrink down the weight of other irrevalant to 0. ---------------------------------> IMPORTANT.
    x_train_poly = transformer.fit_transform(xes)
    mod = LassoCV().fit(x_train_poly,yes)
    return pickle.dumps(mod)

def fit_ridge_with_poly(xes,yes,poly_deg =2):
    transformer = PolynomialFeatures(degree= poly_deg) #Easy way to create new columns when you enter a degree 
    #creates a second and third column 
    #USE L1 OR L2 , shrink down the weight of other irrevalant to 0. ---------------------------------> IMPORTANT.
    x_train_poly = transformer.fit_transform(xes)
    mod = RidgeCV().fit(x_train_poly,yes)
    return pickle.dumps(mod)

def fit_ridge(xes,yes,poly_deg =2):
    #transformer = PolynomialFeatures(degree= poly_deg) #Easy way to create new columns when you enter a degree 
    #creates a second and third column 
    #USE L1 OR L2 , shrink down the weight of other irrevalant to 0. ---------------------------------> IMPORTANT.
    #x_train_poly = transformer.fit_transform(xes)
    mod = RidgeCV().fit(xes,yes)
    return pickle.dumps(mod)

def fit_lasso(xes,yes,poly_deg =2):
    #transformer = PolynomialFeatures(degree= poly_deg) #Easy way to create new columns when you enter a degree 
    #creates a second and third column 
    #USE L1 OR L2 , shrink down the weight of other irrevalant to 0. ---------------------------------> IMPORTANT.
    #x_train_poly = transformer.fit_transform(xes)
    mod = RidgeCV().fit(xes,yes)
    return pickle.dumps(mod)

def predict_using_trained_model(mod_pkl, xes, y_test, x_test, poly_deg =2):
    transformer = PolynomialFeatures(degree= poly_deg)
    x_test_poly = transformer.fit_transform(x_test)
    y_true = y_test

    mod = pickle.loads(mod_pkl)
    y_pred = mod.predict(x_test_poly)
    #print("true Y values")
    #print(y_true)
    #print("predict y values")
    new_y_pred = [ ]
    for num in y_pred:
        if num >= 0.5:
            new_y_pred.append(1)
        else:
            new_y_pred.append(0)
    #print(new_y_pred)
    print("Accuracy Score:", accuracy_score(y_true, new_y_pred)) #-----------------------------> get accuracy score
    mse = mean_squared_error(y_true, new_y_pred)
    r_2 = r2_score(y_true, new_y_pred)
    return mse, r_2

def predict_mod(mod_pkl, x_test, y_test):
    mod = pickle.loads(mod_pkl)
    y_pred = mod.predict(x_test)
    #print(y_pred)
    new_y_pred = [ ]
    for num in y_pred:
        if num >= 0.5:
            new_y_pred.append(1)
        else:
            new_y_pred.append(0)
    #print(new_y_pred)
    print("Accuracy Score: ", accuracy_score(y_test, new_y_pred))
    mse = mean_squared_error(y_test, new_y_pred)
    r_2 = r2_score(y_test, new_y_pred)
    return mse,r_2

def agglomerative_cluster(x_train, y_train, x_test, y_test):
    print("ALGOMERATIVE CLUSTERING")
    clustering = AgglomerativeClustering(n_clusters = 2, linkage='ward')
    clustering.fit(x_train)
    y_pred =clustering.fit_predict(x_test)
    #print(y_pred)
    #print(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Algomerative Clustering:", accuracy)
    return accuracy
def random_forest(x_train, y_train,x_test, y_test):
    print("I am RANDOM FOREST")
    mod_g = RandomForestClassifier()
    mod_g.fit(x_train,y_train)
    #predict 
    pred_y = mod_g.predict(x_test)
    matrix = confusion_matrix(y_test, pred_y)
    print(matrix)
    accuracy = accuracy_score(y_test, pred_y)
    print("Accuracy Random Forest:",accuracy)
    #mse = mean_squared_error(y_test, pred_y)
    #print(mse)
    return accuracy

def gaussian_bayes(x_train, y_train, x_test, y_test):
    print("I AM NAIVE BAYES")
    mod_n = GaussianNB()
    mod_n.fit(x_train, y_train)
    pred_y = mod_n.predict(x_test)
    #sklearn predict_proba gives you value between 0 - 1 -=---------------------> important where model is successful busines or not. 
    #if you calculate predict probab and plot as histogram. to see how decisive the model is. 
    matrix = confusion_matrix(y_test, pred_y)
    print(matrix)
    accuracy = accuracy_score(y_test, pred_y)
    print("Accuracy Naive Bayes:",accuracy)
    #another way
    #AREA UNDER CURVE(use )
    mse = mean_squared_error(y_test, pred_y)
    print(mse)
    return accuracy

def svm_model(x_train, y_train, x_test, y_test):
    print("I AM SVM")
    mod_s = svm.SVC(kernel='rbf')
    mod_s.fit(x_train, y_train)
    pred_y = mod_s.predict(x_test)
    matrix = confusion_matrix(y_test, pred_y)
    #print(matrix)
    accuracy = accuracy_score(y_test, pred_y)
    print("Accuracy SVM: ",accuracy)
    mse = mean_squared_error(y_test, pred_y)
    print(mse)
    return accuracy

def gaussian_mixture(x_train, y_train, x_test, y_test):
    print("I am GAUSSIAN MIXTURE")
    g_mix = GaussianMixture(n_components=2, random_state=2022)
    g_mix.fit(x_train)
    #closest = g_mix.cluster_centers
    y_pred = g_mix.predict(x_test)
    #print("I am GAUSSIAN MIXTURE")
    #print("prediction")
    #print(y_pred)
    #print("true y: below")
    #print(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Gaussian Mixtrure: ",accuracy)
    return accuracy

def k_means(x_train, y_train, x_test, y_test):
    print("I AM KMEANS")
    num_clusters = 2
    kmean = KMeans(n_clusters =2, random_state=2022)
    kmean.fit(x_train)
    closest = kmean.cluster_centers_
    #print(closest)
    labels = kmean.labels_
    #print(labels)
    #print("y_test ")
    #print(y_test)
    y_pred = kmean.predict(x_test)
    #print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Kmeans:", accuracy)
    return accuracy

def view_map(df, type="crime"):
    mapObj = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    locations = list(zip(df.Latitude, df.Longitude))
    heat_map = plugins.HeatMap(locations, radius=5, blur=2)
    mapObj.add_child(heat_map)
    #save map at file location in html
    if(type== "crime"):
        mapObj.save('crimes.html')
    elif(type == "success"):
        mapObj.save('success.html')
    else:
        mapObj.save('unsucessful.html')


def main():
    df_business = open_business("Legally_Operating_Businesses.csv")
    df_stations = open_stations("DOITT_SUBWAY_STATION_01_13SEPT2010.csv")
    times_sq_lat, times_sq_long= find_times_sq(df_stations)
    df_business = remove_na(df_business)
    df_business['times_sq_long'] = times_sq_long
    df_business['times_sq_lat'] = times_sq_lat
    df_business= get_age(df_business)
    df_business['Distance From Times Sq'] = distance_from_times_sq(df_business,lon='times_sq_long',lat='times_sq_lat',dis='distance_from_times_sq')
    #print(df_business)
    
    #GET 
    #Show types of business and count
    sns.histplot(df_business['Industry']).set_title('Types of Businesses')
    plt.show()
    #drop na
    #na_list = df_business.columns[df_business.isna().any()].tolist()
    df_business = df_business[~df_business['Duration'].isna()]
    #print(df_business['Duration'])
    #print(na_list)
    column_names = list(df_business.columns.values)
    
    mean_age = df_business['Duration'].mean()
    std_age = df_business['Duration'].std()
    upper_limit_age = mean_age + 3* std_age
    lower_limit_age = 0+ 1 *std_age
    #print(lower_limit_age)
    
    df_business['Success'] = df_business['Duration'].apply(lambda x: 1 if ((x>lower_limit_age)) else 0)
   
    business_names_tobacco = df_business[df_business['Industry'] == 'Tobacco Retail Dealer']
    print("TOBACCO RETAIL DEALERS")
    print((business_names_tobacco['Business Name']).unique())
    
    #success
    only_successful_types = df_business[df_business['Success'] == 1]
    
    sns.histplot(only_successful_types['Industry']).set_title('Successful businesses Types')
    plt.show()
    only_unsuccessful_types = df_business[df_business['Success'] == 0]
    sns.histplot(only_unsuccessful_types['Industry']).set_title('Unsuccessful_business')
    plt.show()
    
    df_business = encode_categorical_col(df_business, 'Industry')
    print(df_business)
    df_business = distance_from_nearest_station(df_business, df_stations)
    print(df_business)

    #df_business['Zip Codes'] = df_business['Zip Codes'].astype(int)
    print(column_names)
    df_business = df_business.reset_index()
    #remove extreme distances.
    
    #CREATE HISTOGRAM OF NUMBER OF SUCCESSFUL BUSINESS AROUND TIMES SQ.
    #NUMBER OF SUCCESSFUL BUSINESS OUT OF TIMES SQ.
    #df_business = df_business[df_business['Distance From Times Sq'] >= -50]
    #RADIUS OF 50
    df_business = df_business[df_business['Distance From Times Sq'] < 50]
    success_around_times_sq = df_business[df_business['Success']== 1]
    sns.histplot(success_around_times_sq['Distance From Times Sq']).set_title('Successful Business and Distance From Times Sq. 50 km radius')
    #plt.ylabel('Count')
    #plt.xlabel('Distance from times square')
    plt.show()
    
    #RADIUS OF 5
    df_business = df_business[df_business['Distance From Times Sq'] < 100]
    success_around_times_sq = df_business[df_business['Success']== 1]
    sns.histplot(success_around_times_sq['Distance From Times Sq']).set_title('Successful Business and Distance from Times Sq. 100km radius')
    #plt.ylabel('Count')
    #plt.xlabel('Distance from times square')
    plt.show()
    
    
    success_around_station = df_business[df_business['Success']== 1]
    success_around_station = success_around_station[success_around_station['Distance_from_station']< 4]
    sns.histplot(success_around_station['Distance_from_station']).set_title('Successful and Distance from station')
    plt.show()
    
    unsuccess_around_station = df_business[df_business['Success'] == 0]
    unsuccess_around_station = unsuccess_around_station[unsuccess_around_station['Distance_from_station'] < 4]
    sns.histplot(unsuccess_around_station['Distance_from_station']).set_title('Unsuccessful and Distance from station')
    plt.show()
    #unsuccessful and 
    
    #Visual diagram of success of business and distance from times square
    # plt.scatter(df_business['Distance From Times Sq'],df_business['Success'])
    # plt.title('Business and their distance from Times sq')
    # plt.xlabel(' log Distance from times_sq')
    # plt.ylabel('Duration')
    # plt.show()
    df_success = df_business[df_business['Success'] == 1]
    df_unsuccess = df_business[df_business['Success'] == 0]
    
    mapObj = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    locations = list(zip(df_success.Latitude, df_success.Longitude))
    locations2 = list(zip(df_unsuccess.Latitude, df_unsuccess.Longitude))
    heat_map = plugins.HeatMap(locations, radius=3, blur=2, gradient={.65: 'green'})
    heat_map_2 = plugins.HeatMap(locations2, radius=3, blur =2, gradient={1: 'red'})
    mapObj.add_child(heat_map)
    mapObj.add_child(heat_map_2)
    mapObj.caption = "Successful and Unsuccessful Businesses"
    mapObj.save("successful_and_unsucessful.html")

    
    
    lreg = LinearRegression()
    X = df_business.loc[:,['Distance From Times Sq','Distance_from_station','Zip Codes',
                           'Industry_Laundries', 'Industry_Sidewalk Cafe', 'Industry_Stoop Line Stand', 'Industry_Tobacco Retail Dealer', 'Address Borough_brooklyn', 'Address Borough_manhattan', 'Address Borough_queens', 'Address Borough_staten island',]]
    x_train, x_test, y_train,y_test = split_test(X, df_business['Success'])
    #stratify equal number of success and failure. 
    #more accurate read on model performance.
    #FIND OUT THE RAION OF BELOW 5 YREAS AND ABOVE 5 YEAS. 
    #put y variable on stratify. 
    #help you break the 
    
    
    lreg.fit(x_train, y_train)
    y_pred = lreg.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    new_y_pred = [ ]
    for num in y_pred:
        if num >= 0.5:
            new_y_pred.append(1)
        else:
            new_y_pred.append(0)
    
    #print("y values")
    #print(y_test)
    #print("predict")
    #print(y_pred)
    #print(mse)
    print("Accuracy Linear Regression:", accuracy_score(y_test, new_y_pred))
    #try logistic regression
    
    print("I AM LOGISTIC REGRESSION ")
    #from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    mod = model.fit(x_train, y_train)
    y_pred = mod.predict(x_test)
    #print("true Y: values")
    #print(y_test)
    #print("predict y vlues")
    #print(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("mean square of Logistic regression: ", mse)
    print("Accuracy Logistic regression",accuracy_score(y_test, y_pred))
    
    #USING LOGIC
    # print(model.score(x_test,y_test))
    
    #lasso and ridge
    eps = 11
    deg = fit_poly(x_train,y_train,epsilon=eps, verbose=True)
    print(deg)
    poly_deg = 2
    
    print("x has 11 features")
    print(x_train)
    pkg_mod = fit_lasso_with_poly(x_train, y_train, 3)
    mse, r_2 = predict_using_trained_model(pkg_mod, x_train, y_test, x_test, 3)
    print(mse)
    print(r_2)
    
    print("Regular Lasso")
    pkg_mod = fit_lasso(x_train, y_train, 3)
    mse, r_2 = predict_mod(pkg_mod,x_test, y_test)
    print(mse)
    print(r_2)
    
    print("I AM RIDGE REGRESSION")
    pkg_mod = fit_ridge_with_poly(x_train, y_train, 3)
    mse, r_2 = predict_using_trained_model(pkg_mod, x_train, y_test, x_test, 3)
    print(mse)
    print(r_2)
    pkg_mod = fit_ridge(x_train, y_train,3)
    mse, r_2 = predict_mod(pkg_mod,x_test, y_test)
    print(mse)
    print(r_2)

    #Use differnt model
    #RANDOM FOREST
    accuracy = random_forest(x_train, y_train, x_test, y_test)
    #naive bayes
    gaussian_bayes(x_train, y_train, x_test, y_test)
    #SVM
    accuracy = svm_model(x_train, y_train, x_test, y_test)
    #COMPUTE GMM
    accuracy = gaussian_mixture(x_train, y_train, x_test, y_test)
    #GET KMEANS
    accuracy = k_means(x_train, y_train, x_test, y_test)
    #GET ALGOMERATIVE CLUSTERING
    accuracy = agglomerative_cluster(x_train, y_train, x_test, y_test)
    #ABOVE MIGHT CHANGE TAKE INTO ACCOUNT. 
    #IF new business will success?

    #Do you have the slope for the final regression?
    #BIG PART OF theta in text, or beta in stack or scipy.
    #what is the slope? 
    # -1000
    
    #COMPARE MSE TO POLYNOMIAL FEATURES, LINEAR WITH NO POLYNOMIAL Features, 
    # WHAT IS THE SLOPE? 
    #
    
    #COMPARISON BETWEEN NAIVA BAYES, LOGISTIC REGRESSION. 
    #USE ACCURACY
    
    
    #for binary not more than 0.5
    
    
    #how the model performs for each type of business.
    #CREATE  diff model.
    #APPLYING A LOT OF CULSTER ANALYSIS. 
    #if there are different ways to identify business are behave the same based on columns
    #that you choose.
    #Use clustering to justify the df.
    # and use mchine learning to maximuze accuracy.
    
    #use imputation if missing. use mean/ meadian.
    #which one is categorial or which one is
    
main()