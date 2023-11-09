#%%
import pickle



with open('data/datasets_25_29.pickle', 'rb') as handle:
    b = pickle.load(handle)



dias = [ '2023-09-25','2023-09-26','2023-09-27','2023-09-28','2023-09-29', ]

b['df_historico'][dias[0]]
