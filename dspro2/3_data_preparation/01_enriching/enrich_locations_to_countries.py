#!/usr/bin/env python
# coding: utf-8

# # Enriching countries to locations

# ## Enrich the singleplayer json files from the coordinates

# In[1]:


from countryenricher import CountryEnricher

input_dir = '../../1_data_collection/.data'
output_dir = '../01_enriching/.data/'


# In[2]:


geo_enricher = CountryEnricher(input_dir, output_dir, allow_env=True)


# ## Enrich the multiplayer json files from the coordinates

# In[3]:


geo_enricher = CountryEnricher(input_dir, output_dir, from_country=True)

