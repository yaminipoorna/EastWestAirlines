########### All packages #######
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

########################################## airways dataset  #######################################
airways=pd.read_excel("C:/Users/yamini/Desktop/EastWestAirlines/EastWestAirlines.xlsx")
airways             #4000 rows

airways.isna().sum()              #no null values 
airways.duplicated().sum()        #no duplicate values

####################################### checking outliers  #########################################
airways.columns

plt.boxplot(airways.Balance)
plt.title("Balance boxplot")
plt.show()

######### checking exactly how many outliers are present in Balance variable
print("left side values: ",airways["Balance"].mean()-3*airways["Balance"].std())
print("right side values: ",airways["Balance"].mean()+3*airways["Balance"].std())
airways[(airways["Balance"]> 375928.3224566995)| (airways["Balance"]< -228725.6672929086)]     #74 outliers

######### Doing winsorization
airways["Balance"]=winsorize(airways.Balance, limits=[0.01,0.099])

plt.boxplot(airways.Balance)
plt.title("Balance boxplot")
plt.show()

#N0 need of winsorization for other variables because by doing winsorization we miss that exact raw data.
#And mostly variables had more than 50 outliers. So i dont think data is really needed winsorization.
#But i did winsorization for balance variable, to handle the data easily. So i wont get values with huge balance or very less balance.

################################# univariate and bivariate plot analysis ##############################
plt.hist(airways["Balance"])       #right skew..passengers are less who are having high balance 
plt.hist(airways["Qual_miles"])    #very less passengers are qualified for topflight status
plt.hist(airways["cc1_miles"])     #most passengers earned less miles using freq. flyer credit card
plt.hist(airways["cc2_miles"])     #most passengers earned less miles using Rewards credit card
plt.hist(airways["cc3_miles"])     #most passengers earned less miles using Small Business credit card
plt.hist(airways["Bonus_miles"])   #passengers are less who earned more bonus miles.
plt.hist(airways["Bonus_trans"])   #passengers who are having more bonus transactions are less
plt.hist(airways["Flight_miles_12mo"])   #passengers who travelled more miles was less
plt.hist(airways["Flight_trans_12"])     #passengers who did more flight transactions was less
plt.hist(airways["Days_since_enroll"])   #number of days since enrolled in flier program is normally distributed.
#At starting less passengers enrolled(above 200) and in the middle more passengers enrolled(above 500) and now passengers enrolling for
#freq. flier program was declining(below 200). So it is visible that mostly passengers are not showing 
#interest in the freq. flier program.

#scatter plot for balance and number of flight transactions for the last 12 months
plt.scatter(airways.Balance,airways.Flight_trans_12)
plt.xlabel("Balance")
plt.ylabel("Flight_trans_12")
plt.show()
#passengers who did more flight transactions had high balance.

################# Removing variables which are not needed.
airways
airways1=airways.drop(["ID#","Award?"],axis=1)
airways1

####################################### normalization ##################################
airways_norm=normalize(airways1)    
print(airways_norm)

####################################### finding distance ###############################
#calculating linkage. I took "complete linkage" to get the maximum distance between the clusters, as there is
#lot of data which had overlapping. I took "euclidean distance", as i want to grab as much as data possible, instead
#of neat and clear way because i want to offer freq. flier program to more passengers.
z=linkage(airways_norm,method="complete",metric="euclidean")

######################################## dendrogram ####################################
plt.figure(figsize=(15,8))
plt.title("airways")
plt.xlabel("index")
plt.ylabel("distance")
sch.dendrogram(z)
plt.show()                               
# i want to take 5 clusters, as it is more appropriate according to dendrogram.

##################################### clustering #######################################
airways_cluster=AgglomerativeClustering(n_clusters=5,linkage="complete",affinity="euclidean").fit(airways_norm)

#taking out clusters
airways_cluster.labels_
airways_clusters=pd.Series(airways_cluster.labels_)
airways_clusters.head(15)

airways1["clusters"]=airways_clusters
airways1.head(15)

#finding which clusters had more values using histogram.
plt.hist(airways1.clusters)
plt.show()                      #cluster 2 is having more values       

#grouped all the variables using clusters with mean
airways_cluster_groups=airways1.iloc[:,0:].groupby(airways1.clusters).mean()
airways_cluster_groups
#We can target to "cluster 1" passengers as they had high balance and quality miles for qualifiying top flight status. 
#They also had more cc1 miles(freq. flier credit card), it means that they are interested in the freq. flier program.
#Comparing to others, "cluster 1" passengers had high values in most of the variables.
#And in "cluster 1", passengers are not very less or not more. So we can offer freq. flier program to 
#more passengers.

#saving the data
airways1.to_csv("airways_with_clusters.csv")
