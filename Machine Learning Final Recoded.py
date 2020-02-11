#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:20:07 2019

@author: GuiReple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans


df = pd.read_excel('finalExam_preparation_data.xlsx')

survey_df = pd.DataFrame.copy(df)

survey_df.columns

########################
#Step 1: drop survey demographic exams. 
#######################

for col in enumerate(survey_df):
    print(col)

#Renamed columns to facilitate analysis
col_dictionary = [list(survey_df.columns), ['case_id',
                                'age_group',
                                'iphone',
                                'ipod',
                                'android',
                                'blackberry',
                                'nokia',
                                'windows_phone',
                                'hp',
                                'tablet',
                                'other',
                                'none',
                                'music_pps',
                                'tv_ch_pps',
                                'entertain_apps',
                                'tv_show_apps',
                                'game_apps',
                                'social_apps',
                                'general_news_apps',
                                'shopping_apps',
                                'pub_news_apps',
                                'other',
                                'none ',
                                'num_range_apps',
                                'free_apps%',
                                'facebook_freq',
                                'twitter_freq',
                                'myspace_freq',
                                'pandora_freq',
                                'vevo_freq',
                                'youtube_freq',
                                'aol_radio_freq',
                                'last_fm_freq',
                                'yahoo_media_freq',
                                'imdb_freq',
                                'linkedin_freq',
                                'netflix_freq',
                                'keep_up_tech',
                                'give_tech_advice',
                                'enjoy_purch_gadgets',
                                'thinks_tech_toomuch',
                                'tech=life_control',
                                'web_app_savetime',
                                'music_important',
                                'reads_tv_shows',
                                'privacy_issue',
                                'facebook_friend_stalker',
                                'internet_helps_keeptouch',
                                'online_social_only',
                                'self_proclaimed_leader',
                                'thinks_stands_out',
                                'likes_to_giveadvice',
                                'leads_decision_making',
                                'first_to_try_things',
                                'do_what_im_told',
                                'likes_control',
                                'risk_taker',
                                'is_creative',
                                'is_optmistic',
                                'active_gogo',
                                'busy_person',
                                'bargain_person',
                                'likes_shopping',
                                'bundle_package_shop',
                                'online_shopper_a',
                                'luxury_brand_oriented',
                                'designer_brands_pref',
                                'loves_all_apps',
                                'likes_cool_apps_notnum',
                                'app_showoff',
                                'children_influence_app',
                                'purchase_extra_featsapp',
                                'big_spender',
                                'follows_trends',
                                'brand=style',
                                'impulse_buyer',
                                'phone=entertainment',
                                'educ_group_lvl',
                                'marital_status',
                                'no_children',
                                'children<6',
                                'children6_12',
                                'children13_17',
                                'children18up',
                                'race_group',
                                'is_hispanic',
                                'income_group',
                                'gender_group']] 

#change df column names
survey_df.columns = col_dictionary[1]

surv_df = pd.DataFrame.copy(survey_df)   
surv_df = survey_df.drop(['age_group','educ_group_lvl','marital_status','no_children',
                'children<6','children6_12','children13_17','children18up',
                'race_group','is_hispanic','income_group','gender_group','case_id'],
                        axis = 1)

##################################
#Combining groups for analysis
##################################

###############################################################################
# Step 2: Scale to get equal variance
###############################################################################

scaler = StandardScaler()

scaler.fit(surv_df)

X_scaled_reduced = scaler.transform(surv_df)

###############################################################################
# Step 3: Run PCA without limiting the number of components
###############################################################################

surv_pca_reduced = PCA(n_components = None,
                           random_state = 508)


surv_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = surv_pca_reduced.transform(X_scaled_reduced)


###############################################################################
# Step 4: Analyze the scree plot to determine how many components to retain
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(surv_pca_reduced.n_components_)


plt.plot(features,
         surv_pca_reduced.explained_variance_ratio_,##This is the variance per group
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Survey Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()


###############################################################################
# Step 5: Run PCA again based on the desired number of components
###############################################################################

surv_pca_reduced = PCA(n_components = 5,
                           random_state = 508)


surv_pca_reduced.fit(X_scaled_reduced)

###############################################################################
# Step 6: Analyze factor loadings to understand principal components
###############################################################################

factor_loadings_df = pd.DataFrame(pd.np.transpose(surv_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(surv_df.columns)


print(factor_loadings_df.round(2))


factor_loadings_df.to_excel('practice_factor_loadings.xlsx')


###############################################################################
# Step 7: Analyze factor strengths per customer
###############################################################################

X_pca_reduced = surv_pca_reduced.transform(X_scaled_reduced)


X_pca_df = pd.DataFrame(X_pca_reduced)

###############################################################################
###############################################################################
# Step 8: Rename your principal components and reattach demographic information
###############################################################################
###############################################################################

#RENAME!
#X_pca_df.columns = ['Coffee_Shop_Essentials', 'Food_Items', 'Artistic_Pairings']


final_pca_df = pd.concat([survey_df.loc[ : , ['age_group','educ_group_lvl',
                                            'marital_status','no_children',
                                            'children<6','children6_12',
                                            'children13_17','children18up',
                                            'race_group','is_hispanic',
                                            'income_group','gender_group',
                                            ]] , X_pca_df], axis = 1)

final_pca_df.to_excel('pca_with_demographics.xlsx')

###############################################################################
###############################################################################
######################CLUSTERING AND PCA COMBINE###############################
###############################################################################



###############################################################################
# Step 1: Take your transformed dataframe
###############################################################################

print(X_pca_df.head(n = 5))


print(pd.np.var(X_pca_df))



###############################################################################
# Step 2: Scale to get equal variance
###############################################################################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns

###############################################################################
# Step 3: Experiment with different numbers of clusters
###############################################################################

survey_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)


survey_k_pca.fit(X_pca_clust_df)


survey_kmeans_pca = pd.DataFrame({'cluster': survey_k_pca.labels_})


print(survey_kmeans_pca.iloc[: , 0].value_counts())


###############################################################################
# Step 4: Analyze cluster centers
###############################################################################

centroids_pca = survey_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)

###############################################################################
###############################################################################
# Rename your principal components

#centroids_pca_df.columns = ['Coffee_Shop_Essentials', 'Food_Items', 'Artistic_Pairings']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('survey_pca_centriods.xlsx')

########################### CENTROID ANALYSIS #################################

ks = range(1, 50)
inertias = []


for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)


    # Fit model to samples
    model.fit(X_pca_clust_df)


    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)



# Plot ks vs inertias
fig, ax = plt.subplots(figsize = (12, 8))
plt.plot(ks, inertias, '-o')


plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)


plt.show()


###############################################################################
# Step 5: Analyze cluster memberships
###############################################################################

clst_pca_df = pd.concat([survey_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)

final_cltpca_df = pd.concat([survey_df.loc[ : , ['age_group','educ_group_lvl',
                                            'marital_status','no_children',
                                            'children<6','children6_12',
                                            'children13_17','children18up',
                                            'race_group','is_hispanic',
                                            'income_group','gender_group',
                                            ]] , clst_pca_df], axis = 1)

final_pca_df.to_excel('clst_pca_with_demographics.xlsx')

###############################################################################
# Step 6: Plotting
###############################################################################

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'income_group',
            y = 0,
            hue = 'cluster',
            data = final_cltpca_df)
plt.ylim(-2, 2)
plt.xlabel('Income Plot 0')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (12, 4))
sns.boxplot(x = 'age_group',
            y = 0,
            hue = 'cluster',
            data = final_cltpca_df)
plt.ylim(-3, 3)
plt.xlabel('Book Distribution')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'educ_group_lvl',
            y = 0,
            hue = 'cluster',
            data = final_cltpca_df)
plt.ylim(-2, 9)
plt.xlabel('Book Distribution')
plt.tight_layout()
plt.show()




fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'is_hispanic',
            y = 1,
            hue = 'cluster',
            data = final_cltpca_df)
plt.ylim(-2, 9)
plt.xlabel('Book Distribution')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'children18up',
            y = 1,
            hue = 'cluster',
            data = final_cltpca_df)
plt.ylim(-2, 9)
plt.xlabel('Book Distribution')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'children18up',
            y = 3,
            hue = 'cluster',
            data = final_cltpca_df)
plt.ylim(-2, 9)
plt.xlabel('Book Distribution')
plt.tight_layout()
plt.show()



















"""
###reversing question
survey_df['q12'].value_counts()

###Create new column to reverse
survey_df['rev_q12'] = -10

###Reversed our original column q12 to rev_q12 and dropped old column
survey_df['rev_q12'][survey_df['q12'] == 1] = 6
survey_df['rev_q12'][survey_df['q12'] == 2] = 5
survey_df['rev_q12'][survey_df['q12'] == 3] = 4
survey_df['rev_q12'][survey_df['q12'] == 4] = 3
survey_df['rev_q12'][survey_df['q12'] == 5] = 2
survey_df['rev_q12'][survey_df['q12'] == 6] = 1

survey_df['rev_q12'].value_counts()

survey_df = survey_df.drop(columns = ['q12'],
               axis = 1)

survey_df['q1'].value_counts()
"""

