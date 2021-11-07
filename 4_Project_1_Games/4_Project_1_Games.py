#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Описание-данных" data-toc-modified-id="Описание-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Описание данных</a></span></li><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Замена-названий-столбцов" data-toc-modified-id="Замена-названий-столбцов-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Замена названий столбцов</a></span></li><li><span><a href="#Обработка-пропусков" data-toc-modified-id="Обработка-пропусков-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Обработка пропусков</a></span></li><li><span><a href="#Преобразование-данных-в-нужные-типы." data-toc-modified-id="Преобразование-данных-в-нужные-типы.-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Преобразование данных в нужные типы.</a></span></li><li><span><a href="#Считаю-суммарные-продажи-во-всех-регионах" data-toc-modified-id="Считаю-суммарные-продажи-во-всех-регионах-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Считаю суммарные продажи во всех регионах</a></span></li></ul></li><li><span><a href="#Исследовательский-анализ-данных" data-toc-modified-id="Исследовательский-анализ-данных-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Исследовательский анализ данных</a></span><ul class="toc-item"><li><span><a href="#Ранние-годы" data-toc-modified-id="Ранние-годы-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Ранние годы</a></span></li><li><span><a href="#Изменение-продаж-по-платформам" data-toc-modified-id="Изменение-продаж-по-платформам-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Изменение продаж по платформам</a></span></li><li><span><a href="#Характерный-срок-платформы" data-toc-modified-id="Характерный-срок-платформы-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Характерный срок платформы</a></span></li><li><span><a href="#Исследование-актального-периода" data-toc-modified-id="Исследование-актального-периода-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Исследование актального периода</a></span></li><li><span><a href="#Потенциально-прибыльные-платформы" data-toc-modified-id="Потенциально-прибыльные-платформы-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Потенциально прибыльные платформы</a></span></li><li><span><a href="#Влияние-отзывов" data-toc-modified-id="Влияние-отзывов-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Влияние отзывов</a></span><ul class="toc-item"><li><span><a href="#Отзывы-критиков" data-toc-modified-id="Отзывы-критиков-3.6.1"><span class="toc-item-num">3.6.1&nbsp;&nbsp;</span>Отзывы критиков</a></span></li><li><span><a href="#Отзывы-пользователей" data-toc-modified-id="Отзывы-пользователей-3.6.2"><span class="toc-item-num">3.6.2&nbsp;&nbsp;</span>Отзывы пользователей</a></span></li></ul></li><li><span><a href="#Распределение-игр-по-жанрам" data-toc-modified-id="Распределение-игр-по-жанрам-3.7"><span class="toc-item-num">3.7&nbsp;&nbsp;</span>Распределение игр по жанрам</a></span></li></ul></li><li><span><a href="#Портрет-пользователя-каждого-региона" data-toc-modified-id="Портрет-пользователя-каждого-региона-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Портрет пользователя каждого региона</a></span><ul class="toc-item"><li><span><a href="#Самые-популярные-платформы" data-toc-modified-id="Самые-популярные-платформы-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Самые популярные платформы</a></span></li><li><span><a href="#Самые-популярные-жанры" data-toc-modified-id="Самые-популярные-жанры-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Самые популярные жанры</a></span></li><li><span><a href="#Влияние-рейтинга-ESRB-на-продажи" data-toc-modified-id="Влияние-рейтинга-ESRB-на-продажи-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Влияние рейтинга ESRB на продажи</a></span></li></ul></li><li><span><a href="#Проверка-гипотез" data-toc-modified-id="Проверка-гипотез-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Проверка гипотез</a></span><ul class="toc-item"><li><span><a href="#Средние-пользовательские-рейтинги-платформ-Xbox-One-и-PC-одинаковые" data-toc-modified-id="Средние-пользовательские-рейтинги-платформ-Xbox-One-и-PC-одинаковые-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Средние пользовательские рейтинги платформ Xbox One и PC одинаковые</a></span></li><li><span><a href="#Средние-пользовательские-рейтинги-жанров-Action-и-Sports-равны" data-toc-modified-id="Средние-пользовательские-рейтинги-жанров-Action-и-Sports-равны-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Средние пользовательские рейтинги жанров Action и Sports равны</a></span></li></ul></li><li><span><a href="#Вывод" data-toc-modified-id="Вывод-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Вывод</a></span></li></ul></div>

# # Исследование успешности игр
# Вы работаете в интернет-магазине «Стримчик», который продаёт по всему миру компьютерные игры. Из открытых источников доступны исторические данные о продажах игр, оценки пользователей и экспертов, жанры и платформы (например, Xbox или PlayStation). Вам нужно выявить определяющие успешность игры закономерности. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.
# 

# ## Описание данных
#     Name — название игры. 
#     Platform — платформа. 
#     Year_of_Release — год выпуска.
#     Genre — жанр игры. 
#     NA_sales — продажи в Северной Америке (миллионы проданных копий). 
#     EU_sales — продажи в Европе (миллионы проданных копий). 
#     JP_sales — продажи в Японии (миллионы проданных копий). 
#     Other_sales — продажи в других странах (миллионы проданных копий). 
#     Critic_Score — оценка критиков (максимум 100). 
#     User_Score — оценка пользователей (максимум 10). 
#     Rating — рейтинг от организации ESRB (англ. Entertainment Software Rating Board). 
#     Эта ассоциация определяет рейтинг компьютерных игр и присваивает им подходящую возрастную категорию.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
import scipy as scipy
sns.set_style('whitegrid') #('darkgrid')


# In[2]:


games = pd.read_csv('/datasets/games.csv')


# In[3]:


games.info()


# ## Подготовка данных

# ### Замена названий столбцов

# In[4]:


games.columns = map(lambda x: x.lower(), games.columns)         # стоблцы в нижний регистр


#     Перевел названия столбцов к нижнему регистру

# ### Обработка пропусков

# In[5]:


display(games.isna().sum())
display((100 * games.isna().sum() / len(games)).round(2))


#     Есть пропуски в названии, годах  выпуска, оценках критиков и пользователей, и рейтинге. 
#     Т.к. пропусков в оценках и рейтинге много, нельзя их просто удалить

# In[6]:


games.dropna(subset=['name'], inplace = True)                   # 2 игры вообще без данных
games.dropna(subset=['year_of_release'], inplace = True)        # 269 или 1,6 %, найти можно, но не целесообразно


#     Убрал небольшую часть пропусков

# In[7]:


print(games['user_score'].isna().sum(), 'строк с NaN')


# In[8]:


#games[['user_score']] = games[['user_score']].fillna('0') 
games[['user_score']] = games[['user_score']].astype(str)       
print(games.loc[games['user_score'] == 'tbd', 'user_score'].count(),  'оценок со статусом tbd') 


#     Не стал менять NaN на 0.
#     TBD - аббревиатура от английского To Be Determined (будет определено) или To Be Decided (будет решено), поэтому пока игнорирую

# In[9]:


rating_info = games.pivot_table(index=['user_score', 'year_of_release'], 
                                values=['name'], 
                                aggfunc= 'count')
display(rating_info)                                             #tbd  в 2016 году 34


# Оценки tbd заменил на NaN

# In[10]:


games['user_score'] = np.where(games['user_score'] == 'tbd', np.nan, games['user_score']) #заменяю tbd на NaN
games[['user_score']] = games[['user_score']].astype(float) # всё ко float

print(games['user_score'].isna().sum(), 'строк с NaN')                        


# In[11]:


display(games.sort_values('critic_score', ascending=True).head())


#     Оценок критиков с 0 нет

# In[12]:


display(games.isna().sum())


#     Пропуски в рейтинге игры можно заполнить вручную, эти данные можно найти, но делать это нецелесообразно(слишком трудёмко)
#     Оставляю как есть

# ### Преобразование данных в нужные типы.

# In[13]:


games['year_of_release'] = games['year_of_release'].astype('int64')   #т.к. только год, можно в int так удобнее


#     Привел год выпуска к int, т.к. в исследовании используется только год, так удобнее 

# ### Считаю суммарные продажи во всех регионах

# In[14]:


def total_sales(df):                                            #общие продажи
    return (df['na_sales'] + df['eu_sales'] + df['jp_sales'] + df['other_sales'] )

games['total_sales'] = games.apply(total_sales, axis=1)


# ##  Исследовательский анализ данных

# ### Ранние годы

# In[15]:


early_times = games.pivot_table(index='year_of_release', 
                                values='total_sales', 
                                aggfunc='count'
                                )
early_times.plot(title = 'total sales',  figsize=(12, 7), kind='bar')
fig, ax = plt.subplots()                      #раньше было меньше

ax.plot(early_times)
ax.grid()

ax.set_xlabel('Год')
ax.set_ylabel('Количество продаж')

plt.show()


#     Снижение продаж связвнос кризисом 2009 года. Люди в первую очередь экономят на разлечениях, игры в том числе. Меньше продаж - меньше бюджет на разработку новой игры и т. д. Восстановление занимает долгое время

# In[16]:


platform_box_data = games.pivot_table(index=['platform', 'year_of_release'],              # для ящика с усами
                                      values=['total_sales'], 
                                      aggfunc='count'
                                      ).astype(int)
platform_box_data = platform_box_data.reset_index()

plt.figure(figsize=(30, 25), dpi= 100)
sns.boxplot(x='year_of_release', y='total_sales', data=platform_box_data, dodge=True)    # много маленьких ящиков

plt.show()


#     По графикам видно, что данных по играм мало, до второй половины 90х

# In[17]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(15, 7))

sns.distplot(games['year_of_release'],bins = 120, ax=ax_hist)
sns.boxplot(games['year_of_release'], ax=ax_box)

plt.show()


# In[18]:


max_sales = games.pivot_table(index=['platform' ], 
                                  values='total_sales', 
                                  aggfunc='sum'
                                  ).sort_values('total_sales', ascending=False)
# можно брать больше 150(1200 мах)

display(max_sales)


# In[19]:


selected_platforms = max_sales.query('total_sales > 150')
selected_platforms = selected_platforms.reset_index()


#     Данных по играм за ранние годы немного, поэтому отсекаю платформы с продажами меньше 150

# ### Изменение продаж по платформам

# In[20]:


platform_data_1 = games.pivot_table(index=['year_of_release'],  # для общего плота
                                  columns = 'platform', 
                                  values=['total_sales'], 
                                  aggfunc='count'
                                  )
platform_data_1 = platform_data_1.reset_index()


# In[21]:


platform_data_1.plot('year_of_release', 'total_sales', figsize=(20, 7))                   # много линий, не информативно
plt.show()


#     Очень много линий, неинформативно, поэтому делаю для каждого отдельно.

# In[22]:


platform_data = games.pivot_table(index=['platform', 'year_of_release'], # отдельных плотов
                                  #columns = 'platform', 
                                  values=['total_sales'], 
                                  aggfunc='count'
                                  )
platform_data = platform_data.reset_index()


# In[23]:


for i in [k for k in set(selected_platforms['platform'])]:                                   # больше плотов и ящиков
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True)
    sns.distplot(platform_data.loc[platform_data['platform'] == i]['total_sales'], 
                 ax=ax_hist, 
                 bins=10
                 )
    sns.boxplot(platform_data.loc[platform_data['platform'] == i]['total_sales'], 
                 ax=ax_box).set_title(i)
    plt.show()
    sns.distplot(platform_data.loc[platform_data['platform'] == i]['year_of_release'], 
                 bins=10,  
                 kde=True
                 ).set_title(i)
    plt.show()
   
    platform_data.loc[platform_data['platform'] == i].plot('year_of_release', 'total_sales', kind='bar').set_title(i)
    plt.show()


#     По графикам на 2015 и неполный 2016 года рост показывают Xone, PS4, PC

# ### Характерный срок платформы

# In[24]:


mean_year = 0
for i in [k for k in set(selected_platforms['platform'])]:
    years = int(platform_data.loc[platform_data['platform'] == i]['year_of_release'].describe().loc[['max']]) - int(platform_data.loc[platform_data['platform'] == i]['year_of_release'].describe().loc[['min']])
    print('Срок эксплуатации платформы {} {:.0f} лет'.format(i, years))
    mean_year += years 
mean_year = mean_year / len(selected_platforms)
print('По выбранным параметрам средний срок службы платформы {:.0f} лет'.format(mean_year))


#     По графиками видно, что с момента появления платформы до роста проходит примерно 2-3 года. поэтому есть смысл анализировать не характерный срок, а тот период, когда в среднем у платформы наблюдается значительный рост продаж.
#     Беру последние 3 года

# ### Исследование актального периода

# In[25]:


good_platforms = games.query('year_of_release > 2013')


# In[26]:


good_platforms_sales = good_platforms.pivot_table(index= 'platform', 
                                 values = 'total_sales', 
                                 aggfunc= 'sum').sort_values(by='total_sales', ascending=False)
display(good_platforms_sales)


#     Хоть PC и показывает небольшой уровень общих продаж, по графикам виден рост, как у PS4 и XOne. У остальных падение, что говорит о том, что они скоро исчезнут

# ### Потенциально прибыльные платформы

# In[27]:


good_platforms_sales_ps4 = good_platforms[good_platforms['platform'] == 'PS4']
good_platforms_sales_xone = good_platforms.loc[good_platforms['platform'] == 'XOne']
good_platforms_sales_pc = good_platforms.loc[good_platforms['platform'] == 'PC']


# In[28]:


good_platforms_sales_ps4.boxplot('total_sales').set_title('PS4')
plt.show()

good_platforms_sales_xone.boxplot('total_sales').set_title('XOne')
plt.show()

good_platforms_sales_pc.boxplot('total_sales').set_title('PC')
plt.show()


#     По графикам видно, что самый большой уровень продаж у PS4, затем XOne, PC - третий

# In[29]:


display('PS4', 
      good_platforms_sales_ps4[['total_sales']].describe().loc['50%'], 
      'XOne',  
      good_platforms_sales_xone[['total_sales']].describe().loc['50%'], 
      'PC', 
      good_platforms_sales_pc[['total_sales']].describe().loc['50%']
     )


#     Заметна значительная разница между консолями и ПК,скорее всего это из-за пиратства. На консоли практически невозможно 'скачать ' игры, если только не прошить саму консоль и лишиться некоторых функций. А на ПК это сделать проще.
#     Плюс эксклюзивные игры, которые сначала выходят на консоли, затем на ПК, или так и остаются эксклюзивом определенной консоли.

# ### Влияние отзывов

# #### Отзывы критиков
# 

# In[30]:


sales_ps4 = sns.lmplot(x='critic_score', y='total_sales', data=good_platforms_sales_ps4)
plt.show()
print(good_platforms_sales_ps4['critic_score'].corr(good_platforms_sales_ps4['total_sales']))
sales_xone = sns.lmplot(x='critic_score', y='total_sales', data=good_platforms_sales_xone)
plt.show()
print(good_platforms_sales_xone['critic_score'].corr(good_platforms_sales_xone['total_sales']))
sales_pc = sns.lmplot(x='critic_score', y='total_sales', data=good_platforms_sales_pc)
plt.show()
print(good_platforms_sales_pc['critic_score'].corr(good_platforms_sales_pc['total_sales']))


#      У все платформ по графикам и коэффициенту корреляции видна положительная взаимосвязь. У PS4 и XOne она сильнее, у PC слабая.
#      по графикам видно, что чем выше оценка критиков, тем выше уровень продаж

# #### Отзывы пользователей

# In[31]:


display(good_platforms.pivot_table(index= 'user_score', 
                                values= 'total_sales', 
                                aggfunc= 'count'))


# In[32]:


sales_ps4 = sns.lmplot(x='user_score', y='total_sales', data=good_platforms_sales_ps4)
plt.show()
print(good_platforms_sales_ps4['user_score'].corr(good_platforms_sales_ps4['total_sales']))

sales_xone = sns.lmplot(x='user_score', y='total_sales', data=good_platforms_sales_xone)
plt.show()
print(good_platforms_sales_xone['user_score'].corr(good_platforms_sales_xone['total_sales']))

sales_pc = sns.lmplot(x='user_score', y='total_sales', data=good_platforms_sales_pc)
plt.show()
print(good_platforms_sales_pc['user_score'].corr(good_platforms_sales_pc['total_sales']))


#     По коэффициенту корреляции видна слаба зависимость между отзывами пользователей и продажами вне зависимости от рассматриваемой платформы.
#     По графикам видно, что игры с оценками выше 4 имеют больший уровень продаж, но он не сильно меняется от 4 до 10 

# ### Распределение игр по жанрам

# In[33]:


user_char_genre = good_platforms.pivot_table(index='genre', 
                                                values='total_sales', 
                                                aggfunc='sum'
                                                ).sort_values(by= 'total_sales', ascending=False)
display(user_char_genre)


#     Самые прибыльные Action, Shooter, Sports, Role-Playing
#     Жанры с наименьшеи кровнем продаж Strategy, Puzzle

# ##  Портрет пользователя каждого региона

# In[34]:


good_platforms['rating'] = good_platforms['rating'].fillna('no_rating')

user_char_platform = good_platforms.pivot_table(index='platform', 
                                                values=['na_sales', 'eu_sales', 'jp_sales'], 
                                                aggfunc='sum'
                                                )

user_char_genre = good_platforms.pivot_table(index='genre', 
                                                values=['na_sales', 'eu_sales', 'jp_sales'], 
                                                aggfunc='sum'
                                                )

user_char_rating = good_platforms.pivot_table(index='rating', 
                                                values=['na_sales', 'eu_sales', 'jp_sales'], 
                                                aggfunc='sum'
                                                )
pd.set_option('chained_assignment', None)


# ### Самые популярные платформы

# In[35]:


for i in ['na_sales', 'eu_sales', 'jp_sales']:
    display(i, user_char_platform[i].sort_values(ascending=False).head())
    user_char_platform[i].sort_values(ascending=False).head(5).plot(kind='bar')
    plt.show()
    print()


#     В регионах NA и еEU первые места по продажам занимают PS4 и XOne, причем в Европе PS4 лидирует с большим отрывом 130\46 млн копий, а в Северной Америке отстование XOne незначительно 98\81
#     В Японии PS4 на втором месте с 15 млн, а XOne не попал в топ-5 вообще

# ### Самые популярные жанры

# In[36]:


for i in ['na_sales', 'eu_sales', 'jp_sales']:
    display(i, user_char_genre[i].sort_values(ascending=False).head())
    user_char_genre[i].sort_values(ascending=False).head().plot(kind='bar')
    plt.show()
    print()


#     Регионы NA и EU похожи и предпочтениях по жанрам Shooter, Action, Sports занимают первые три строчки(в Северной Америке Shooter на первом, в Европе - Action).
#     В Японии первые 3: Role-Playing, Action, Fighting(скорее всего культурные особенности)

# ### Влияние рейтинга ESRB на продажи

# In[37]:


for i in ['na_sales', 'eu_sales', 'jp_sales']:
    display(i, user_char_rating[i].sort_values(ascending=False).head())
    user_char_rating[i].sort_values(ascending=False).head().plot(kind='bar')
    #user_char_rating[user_char_rating[i] == np.nan].sum().plot(kind='bar')
    plt.show()
    print()


#     В Европе и Северной Америке рейтинги распределились M, E, T, E10+, причём примерно в одинаковом количестве.
#     В Японии T, E, M, E10+
#     В NA и EU большей  популярностью пользуются игры со взрослым рейтингом, в JP - с подростковым
#     Во всех региронах много игр вообще не имеют рейтинга

# In[38]:


rating_data = good_platforms.pivot_table(index='rating', 
                                        values = 'total_sales', 
                                        aggfunc = 'sum')
display(rating_data.sort_values(by='total_sales', ascending=False))


#     По продажам самые прибыльные игры со взрослым рейтингом, затем рейтинг для всех от 6, подростковый и всем старше 10 лет

# ## Проверка гипотез

#     Нулевая гипотеза всегда формулируется так, чтоб использовать знак равенства. Исходя из неё формулирую альтернативную гипотезу.
#     Для проверки гипотез использовал t-критерий Стьюдента, потому что этот критерий оценивает насколько различаются средние выборки.
#     А для проверки равенства дисперсий случайных выборок генеральной совокупности применил тест Левена

# ### Средние пользовательские рейтинги платформ Xbox One и PC одинаковые

# In[39]:


user_score_games = games.dropna(subset = ['user_score'], inplace=True)


#     Т.к. в user_score оставлял NaN, сейчас их надо убрать, помешают

# In[40]:


alpha = 0.05  # критический уровень статистической значимости

p1 = scipy.stats.levene(games.loc[games['platform'] == 'XOne']['user_score'], 
                        games.loc[games['platform'] == 'PC']['user_score']) # тест Левена
print('p-значение теста Левена: ', p1.pvalue)

# если p-value окажется меньше него - отвергнем гипотезу

if p1.pvalue < alpha:
    print('Отвергаем нулевую гипотезу о равных дисперсиях')
else:
    print('Не получилось отвергнуть нулевую гипотезу, считаю равными дисперсии выборок') 

print([np.var(x, ddof=1) for x in [games.loc[games['platform'] == 'XOne']['user_score'],
                                   games.loc[games['platform'] == 'PC']['user_score']]]
     ) #поэтому False в check_hypo

#  Если результирующее p-значение критерия Левена меньше некоторого уровня значимости (обычно 0,05), 
# полученные различия в дисперсиях выборки маловероятны на основе случайной выборки из генеральной 
# совокупности с равными дисперсиями. Таким образом, нулевая гипотеза о равных дисперсиях отклоняется 
# и делается вывод о различии дисперсий в генеральной совокупности

check_hypo = st.ttest_ind(games.loc[games['platform'] == 'XOne']['user_score'], 
                          games.loc[games['platform'] == 'PC']['user_score'], 
                          equal_var=True
                         )

print('p-значение: ', check_hypo.pvalue)

# если p-value окажется меньше него - отвергнем гипотезу

if check_hypo.pvalue < alpha:
    print('Отвергаем нулевую гипотезу')
else:
    print('Не получилось отвергнуть нулевую гипотезу') 


#     Т.к. при проверке опровергли нулевую гипотезу, считаю средние пользовательские рейтинги PC и Xbox One неравными
#     Т.е. среди пользователей популярность платформ отличается

# ### Средние пользовательские рейтинги жанров Action и Sports равны

# In[41]:


p2 = scipy.stats.levene(games.loc[games['genre'] == 'Action']['user_score'], 
                        games.loc[games['genre'] == 'Sports']['user_score']
                       )
print('p-значение теста Левена: ', p2.pvalue)

if p1.pvalue < alpha:
    print('Отвергаем нулевую гипотезу о равных дисперсиях')
else:
    print('Не получилось отвергнуть нулевую гипотезу, считаю равными дисперсии выборок') 

print([np.var(x, ddof=1) for x in [games.loc[games['genre'] == 'Action']['user_score'], games.loc[games['genre'] == 'Sports']['user_score']]]) #поэтому False в check_hypo

check_hypo = st.ttest_ind(games.loc[games['genre'] == 'Action']['user_score'], games.loc[games['genre'] == 'Sports']['user_score'], equal_var=True)
print('p-значение: ', check_hypo.pvalue)

alpha = 0.05     # критический уровень статистической значимости
                 # если p-value окажется меньше него - отвергнем гипотезу
if check_hypo.pvalue < alpha:
    print('Отвергаем нулевую гипотезу')
else:
    print('Не получилось отвергнуть нулевую гипотезу') 


#     Т.к. не удалось опровергнуть нулевую гипотезу, считаю средние пользовательские рейтинги жанров Action и Sports равны
#     Т.е. делать ставку на один из этих рейтингов не стоит, надо сравнить остальные

# ## Вывод

#      В исследуемых данных были пропуски, в частности в рейтинге игры, оценках критиков и пользователей. Часть пропусков удалены, часть проигнорированы.
#      В исследовании были проанализированы данные за продолжительный период, с середины 80-х до 2016 года. Были рассмотрены различные платформы, и рассчитан срок жизни платформы. Исходя из исследованных данных прогноз на 2017 год следует строить на основе продаж за последние 2-3 года.
#      Далее были выбраны перспективные платформы, по которым и будет строиться план рекламнной кампании. Это Xbox One, PS4 и PC.
#      Исследованы оценки критиков и пользователей. Оценки критиков имеют связь с продажами сильнее, чем оценки пользователей, поэтому следует опираться на них.
#      Исследование жанров показало, что самые продаваемые это Action, Shooter, Sports, Role-Playing.
#      Далее были рассмотрены  разные регионы: Северная Америка, Европа и Япония
#      В первых двух следует уделить внимание PS4 и Xbox One, там они самые распростроненные, в Японии - PS4 на втором месте по популярности, а Xbox One в Японии не признают.
#      При рассмотрении жанровых предпочтений по регионам оказалось, что Shooter и Action самые любимые у игроков Европы и Северной Америки, в Японии - Role-Playing.
#      Такой критерий как рейтинг ESRB показал, что в Японии очень распространены игры "без рейтинга", затем идут игры с подростковым рейтингом Т. В Северной Америке и Европе пользуется популярностью взрослый рейтинг М, на втором месте игры "без рейтинга".
#      Далее были рассмотрены гипотезы о том, что средние пользовательские оценки платформ Xbox One и PC одинаковые, и средние пользовательские рейтинги жанров Action и Sports одинаковые. Первая гипотеза была опровергнута, т.е. нельзя считать пользовательские рейтинги Xbox One и PC одинаковыми. А вторую опровергнуть не удалось, поэтому считаем, что пользователям одинаково интересны жанры Action и Sports.
#      На основе всех этих данных можно строить рекламную компанию
#      
#      
