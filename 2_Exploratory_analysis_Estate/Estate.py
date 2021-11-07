#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Изучение-данных-из-файла" data-toc-modified-id="Изучение-данных-из-файла-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Изучение данных из файла</a></span></li><li><span><a href="#Предобработка-данных" data-toc-modified-id="Предобработка-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Предобработка данных</a></span></li><li><span><a href="#Расчёты-и-добавление-результатов-в-таблицу" data-toc-modified-id="Расчёты-и-добавление-результатов-в-таблицу-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Расчёты и добавление результатов в таблицу</a></span></li><li><span><a href="#Исследовательский-анализ-данных" data-toc-modified-id="Исследовательский-анализ-данных-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Исследовательский анализ данных</a></span><ul class="toc-item"><li><span><a href="#Цена" data-toc-modified-id="Цена-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Цена</a></span></li><li><span><a href="#Площадь" data-toc-modified-id="Площадь-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Площадь</a></span></li><li><span><a href="#Комнаты" data-toc-modified-id="Комнаты-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Комнаты</a></span></li><li><span><a href="#Высота-потолков" data-toc-modified-id="Высота-потолков-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Высота потолков</a></span></li><li><span><a href="#Зависимость-цены-от-этажа" data-toc-modified-id="Зависимость-цены-от-этажа-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Зависимость цены от этажа</a></span></li><li><span><a href="#Зависимость-цены-от-количества-комнат" data-toc-modified-id="Зависимость-цены-от-количества-комнат-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Зависимость цены от количества комнат</a></span></li><li><span><a href="#Зависимость-цены-от-расстояния-до-центра" data-toc-modified-id="Зависимость-цены-от-расстояния-до-центра-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Зависимость цены от расстояния до центра</a></span></li><li><span><a href="#Зависимость-цены-от-дня-недели-публикации" data-toc-modified-id="Зависимость-цены-от-дня-недели-публикации-4.8"><span class="toc-item-num">4.8&nbsp;&nbsp;</span>Зависимость цены от дня недели публикации</a></span></li><li><span><a href="#Зависимость-цены-от-месяца-публикации" data-toc-modified-id="Зависимость-цены-от-месяца-публикации-4.9"><span class="toc-item-num">4.9&nbsp;&nbsp;</span>Зависимость цены от месяца публикации</a></span></li><li><span><a href="#Зависимость-цены-от-года-публикации" data-toc-modified-id="Зависимость-цены-от-года-публикации-4.10"><span class="toc-item-num">4.10&nbsp;&nbsp;</span>Зависимость цены от года публикации</a></span></li><li><span><a href="#Топ-10" data-toc-modified-id="Топ-10-4.11"><span class="toc-item-num">4.11&nbsp;&nbsp;</span>Топ-10</a></span></li><li><span><a href="#Расстояние-до-центра" data-toc-modified-id="Расстояние-до-центра-4.12"><span class="toc-item-num">4.12&nbsp;&nbsp;</span>Расстояние до центра</a></span></li><li><span><a href="#Анализ-центра-города" data-toc-modified-id="Анализ-центра-города-4.13"><span class="toc-item-num">4.13&nbsp;&nbsp;</span>Анализ центра города</a></span></li></ul></li><li><span><a href="#Общий-вывод" data-toc-modified-id="Общий-вывод-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Общий вывод</a></span></li></ul></div>

# # Исследование объявлений о продаже квартир
# 
# В вашем распоряжении данные сервиса Яндекс.Недвижимость — архив объявлений о продаже квартир в Санкт-Петербурге и соседних населённых пунктах за несколько лет. Нужно научиться определять рыночную стоимость объектов недвижимости. Ваша задача — установить параметры. Это позволит построить автоматизированную систему: она отследит аномалии и мошенническую деятельность. 
# 
# По каждой квартире на продажу доступны два вида данных. Первые вписаны пользователем, вторые получены автоматически на основе картографических данных. Например, расстояние до центра, аэропорта, ближайшего парка и водоёма. 

# Описание данных:
#  - airports_nearest — расстояние до ближайшего аэропорта в метрах (м)
#  - balcony — число балконов
#  - ceiling_height — высота потолков (м)
#  - cityCenters_nearest — расстояние до центра города (м)
#  - days_exposition — сколько дней было размещено объявление (от публикации до снятия)
#  - first_day_exposition — дата публикации
#  - floor — этаж
#  - floors_total — всего этажей в доме
#  - is_apartment — апартаменты (булев тип)
#  - kitchen_area — площадь кухни в квадратных метрах (м²)
#  - last_price — цена на момент снятия с публикации
#  - living_area — жилая площадь в квадратных метрах (м²)
#  - locality_name — название населённого пункта
#  - open_plan — свободная планировка (булев тип)
#  - parks_around3000 — число парков в радиусе 3 км
#  - parks_nearest — расстояние до ближайшего парка (м)
#  - ponds_around3000 — число водоёмов в радиусе 3 км
#  - ponds_nearest — расстояние до ближайшего водоёма (м)
#  - rooms — число комнат
#  - studio — квартира-студия (булев тип)
#  - total_area — площадь квартиры в квадратных метрах (м²)
#  - total_images — число фотографий квартиры в объявлении

# ## Изучение данных из файла

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('/datasets/real_estate_data.csv', sep='\t')


# In[3]:


data.info()


# In[4]:


display(data.isna().sum())


#     Вывод: В данных есть пропуски и заполненые NaN строки

# ## Предобработка данных

# In[5]:


data['last_price'] = data['last_price'].astype(int)     #цену в int


#     Перевел цену в int

# In[6]:


data['first_day_exposition'] = pd.to_datetime(
    data['first_day_exposition'], 
    format='%Y-%m-%dT%H:%M:%S'
)                                                       #дату к норм формату


#     Привел дату к удобному формату

# In[7]:


data['balcony'] = data['balcony'].fillna(0).astype(int) #замена NaN на 0, Г - Логика


#     Заменил NaN в столбце 'balcony' на 0

# In[8]:


#data.dropna(subset = ['floors_total'], inplace=True)    # убрал объявления без указания этажности
data['floors_total'] = data['floors_total'].astype('Int64') #этаж в int


#     Убрал обявления без указания этажа, эти данные вязть неоткуда.
#     Сделать обязательным поле при заполнении формы на сайте.

# In[9]:


data.dropna(subset = ['locality_name'], inplace=True)   # убрал без города


#     Убрал обявления без указания города.
#     Сделать обязательным поле при заполнении формы на сайте.

# In[10]:


data['parks_around3000'] = data['parks_around3000'].fillna(0).astype(int)#если парка нет, то это не NaN, но 0


#     В столбце 'parks_around3000' заменил NaN на 0, т.к. парков ближе 3км нет

# In[11]:


data['ponds_around3000'] = data['ponds_around3000'].fillna(0).astype(int) #пруды аналогично


#     Аналогично паркам

# In[12]:


data = data.drop_duplicates(
    [
    'floors_total', 
    'locality_name', 
    'kitchen_area', 
    'floor', 
    'rooms', 
    'living_area', 
    #'total_area'
    ]
) 


#     Убрал явные дубликаты

# In[13]:


data['city_center_km'] = data['cityCenters_nearest'] / 1000
data['city_center_km'] = (
    data['city_center_km'].fillna(0)
    .astype(int)
    .astype(object)
    .where(data['city_center_km'].notnull())# google в помощь
)


# In[14]:


def mean_locality_ceil(df):
    ceil_mean = df.groupby('locality_name')['ceiling_height'].mean() # средний потолок по городам
    unique_total_income = list(set(df['locality_name']))             # список городов
    dict_mean = dict(ceil_mean)
    for i in unique_total_income:                                    # заполняю пропуски средними значениями
        df.loc[df["locality_name"] == i,'ceiling_height']         = df.loc[df["locality_name"] == i,'ceiling_height'].fillna(dict_mean[i])


# In[15]:


mean_locality_ceil(data)


#     Заполнил пропуски в столбце 'ceiling_height' средними значениями по городу, где возможно

# In[16]:


display(data['ceiling_height'].isna().sum())#, sep='\n')#пропусков в полоках осталось меньше процента


# In[17]:


#data.dropna(subset = ['ceiling_height'], inplace=True)    # -120 объявлений без указания высоты потолков


#     Убрал эти пропуски

# In[18]:


null_airports = data[data['airports_nearest'].isnull()]
for i in dict(null_airports['locality_name'].value_counts()).items():
    display (i)  #('Санкт-Петербург', 84)


#     Из всех пропущенных строк в столбце 'airports_nearest', только 84 обявления из Санкт-Петербурга, а для остальных населеных пунктов эта информация неважна

# In[19]:


null_city_center = data[data['cityCenters_nearest'].isnull()]
for i in dict(null_city_center['locality_name'].value_counts()).items():
    display (i)  #('Санкт-Петербург', 60)


#     Аналогично расстоянию до аэропортов, 60 обявлений из Санкт-Петербугра.Оставляю как есть

# In[20]:


display(data['rooms'].value_counts())


# 192 квартиры с 0 комнат

# In[21]:


rooms_count = data.query('rooms == 0 and studio == False and open_plan == False')
display(rooms_count)


#     Пустой series, значит есть только 0-комнатные студий или открытой планировки
#     вывод: 0 комнат, этот или студия, или открытая планировка

# In[22]:


data.loc[data['rooms'] == 0, 'rooms'] = 1# 0 комнат, студия или открытая планировака, == 1


#     Меняю на 0 на 1

# In[23]:


data['is_apartment'] = data['is_apartment'].replace({np.nan: False})


#     Пропуски в 'is_apartment' считаю False, т.к. если в обявлении апартаменты, то это указывают, NaN говорит, что, скорее всего, в обявлении не апартаменты.
#     Пропуски в 'days_exposition' говорят, чтоб объявление на момент сбора данных не закрыто, и невозможно указать сколько дней оно было размещено (от публикации до снятия).
#     Пропуски в 'parks_nearest' и 'ponds_nearest' уазывают на то, что нет возможности указать расстояние до парка или пруда, если их нет в радиусе 3км

# In[24]:


data.info()


# In[25]:


display(data.isna().sum())


# In[26]:


display((100 * data.isna().sum() / len(data)).round(2))


# ## Расчёты и добавление результатов в таблицу

# In[27]:


def square_price(df):
    square_price =  df['last_price'] / df['total_area']
    return int(square_price)                                #цену в int


# In[28]:


data['square_price'] = data.apply(square_price, axis=1) #цена за кв.метр


#     Добавил столбец со ценой на кв.м

# In[29]:


data['week'] = data['first_day_exposition'].dt.day_name()
data['year'] = data['first_day_exposition'].dt.year
data['month'] = data['first_day_exposition'].dt.month


#     Добавил столбцы с годом, месяцем и днём недели

# In[30]:


def floor_info(df):                      #категория этажей
    if df['floor'] == df['floors_total']:
        return 'последний'
    elif df['floor'] == 1:
        return 'первый'
    else:
        return 'другой'


# In[31]:


data['floor_info'] = data.apply(floor_info, axis=1)


#     Добавил стобец категории этажей

# In[32]:


def dif_kitchen(df):                    #площадь кухни 9% пропуков, общая есть
    try:
        diff = df['kitchen_area'] / df['total_area']
        return diff
    except:
        return 'каких-то данных нет'


# In[33]:


data['kitchen_area_ratio'] = dif_kitchen(data)  #


#     Добавил соотношение площади кухни к общей.
#     В таблице отсутствуют данные о 9% площади кухни, но во всех есть данные об общей площади.
#     Поэтому решил оставить часть пропусков, т.к. заполнить их адекватно врядли получится    

# In[34]:


def dif_living(df):                     #жилая площать 7% пропусков, есть общая
    try:
        diff = df['living_area'] / df['total_area']
        return diff
    except:
        return 'каких-то данных нет'


# In[35]:


data['living_area_ratio'] = dif_living(data)    #кухни и жилой, сумма больше общей


#     Добавил соотношение жилой площади к общей.
#     Аналогично с соотношением площади кухни
#     В данных о площадях есть несоответствия, когда сумма площади кухни и жилой больше общей площади квартиры.
#     Поэтому существующие данные о площади кухни и жилой площади, рекомендую считать некорректными.
#     Желательно добавить проверку при заполнении объявления: Sобщая >= Sжилая + Sкухни

# ## Исследовательский анализ данных

# ### Цена

# In[36]:


data['last_price'].hist(bins = 1000)
plt.show()


#     График не является наглядным, цены почти от 0 до 800млн.

# In[37]:


display(data['last_price'].describe().astype(int))#50% - median


#     Отмечаю разницу между средним и медианой

# In[38]:


plt.ylim(0, 15000000)
data.boxplot(column='last_price')# первый квартиль от нуля-проверить, и max проверить
plt.show()


#     Судя по диаграмме разброса - максимум чуть меньше 12 млн.

# In[39]:


def price_cat(df):
    if df < 1000000:
        return 'до миллиона'
    elif 1000000 <= df < 5000000:
        return 'от 1 до 5 млн'
    elif 5000000 <= df < 10000000:
        return 'от 5 до 10 млн'
    elif 10000000 <= df < 20000000:
        return 'от 10 до 20'
    elif 20000000 <= df < 100000000:
        return 'от 20 до 100'
    elif 100000000 <= df < 300000000:
        return 'от 100 до 300'
    else:
        return 'более 300'


#     Дополнительные категории

# In[40]:


data['price_category'] = data['last_price'].apply(price_cat)


# In[41]:


depend_room_price = data.pivot_table(index='price_category', 
                                     values='last_price', 
                                     aggfunc=['median', 'count']
                                    )        
display(depend_room_price)# 1 12комнатная кв. зв 420млн


# In[42]:


good_data = data.query('100000 < last_price < 20000000') #разница ср и мед 1млн, убрал 3%, 713 обявлений


#     Убрал слишком низкие и слишком высокие цены, создал новый df с которым буду работать

# In[43]:


good_data['last_price'].hist(bins = 1000)
plt.show()


#     График более информативный

# In[44]:


display(good_data['last_price'].describe().astype(int))    #среднне стало ближе к медиане


#     Среднее значение цены стало ближе к медиане

# In[45]:


good_data.boxplot(column='last_price')
plt.show()


# ### Площадь

# In[46]:


data['total_area'].hist(bins = 1000)
plt.show()


#     По графику видно, что есть выбросы

# In[47]:


display(data['total_area'].describe().astype(int))


#     Отмечаю разницу между средним и медианой

# In[48]:


plt.ylim(0, 200)
data.boxplot(column='total_area')
plt.show()


#     Всё что больше 115 - выбросы

# In[49]:


def area_cat(df):
    if df < 25:
        return 'меньше 25м'
    elif 25 <= df < 50:
        return 'от 25 до 50м'
    elif 50 <= df < 75:
        return'от 50 до 75м'
    elif 75 <= df < 100:
        return 'от 75 до 100м'
    elif 100 <= df < 125:
        return 'от 100 до 125м'
    elif 125 <= df < 200:
        return 'от 125 до 200м'
    elif 200 <= df < 300:
        return 'от 200 до 300м'
    elif 300 <= df < 400:
        return 'от 300 до 400м'
    elif 400 <= df < 500:
        return 'от 400 до 500м'
    else:
        return'больше 500 м'


#     Дополнительные категории

# In[50]:


data['area_category'] = data['total_area'].apply(area_cat)


# In[51]:


depend_area_price = data.pivot_table(
    index='area_category', 
    values=['last_price', 'square_price'], 
    aggfunc={'last_price' : ['median', 'count'], 
              'square_price' : 'median'}
) 


# In[52]:


display(depend_area_price)# убираю больше 200м их 230, 


#     Считаю сколько обявлений можно отбросить чтобы не пострадали расчеты и стали ближе к реальным

# In[53]:


good_data = good_data.query('total_area < 200') #разница ср и мед 2, стд откл 35\26


#     Убрал квартиры с площадью более 200 кв.м

# In[54]:


good_data['total_area'].hist(bins = 1000)
plt.show()


#     Некоторые площади повторяются чаще других, скорее всего типовые дома, один застройщик

# In[55]:


display(good_data['total_area'].describe().astype(int))    #среднне стало ближе к медиане


#     Среднее стало ближе к медиане

# In[56]:


good_data.boxplot(column='total_area')
plt.show()


#     Часть выбросов осталось, без них никак

# ### Комнаты

# In[57]:


data['rooms'].hist(bins = 50)
plt.show()


#     Есть квартиры с большим количеством комнат

# In[58]:


display(data['rooms'].describe().astype(int))# median = mean


# In[59]:


data.boxplot(column='rooms')# 0 комнат, студия или открытая планировака, == 1
plt.show()


#     Больше 6 - выбросы

# In[60]:


depend_rooms_price = data.pivot_table(
      index='rooms', 
      values=['last_price', 'square_price'], 
      aggfunc={'last_price' : ['median', 'count'], 
              'square_price' : 'median'}
).astype(int)


# In[61]:


display(depend_rooms_price)#убираю больше 9 комнат


# In[62]:


good_data = good_data.query('rooms < 10') #беру срез от искомого df


#     Оставляю 9 и меньше комнат

# In[63]:


good_data['rooms'].hist(bins = 50)
plt.show()


# In[64]:


display(good_data['rooms'].describe().astype(int))    #среднне стало ближе к медиане


# In[65]:


good_data.boxplot(column='rooms')
plt.show()


#     Всё более менее в порядке

# ### Высота потолков

# In[66]:


data['ceiling_height'].hist(bins = 300)
plt.show()


#     График не является наглядным, высота до 100м.

# In[67]:


display(data['ceiling_height'].describe())


#     Разница между средним и медианой 0.05, минимум 1м, максимум 100м
# 

# In[68]:


data.boxplot(column='ceiling_height')
plt.ylim(1, 4)
plt.show()


#     Высота потолков чуть более 3м максимум, остальное выбросы

# In[69]:


depend_ceiling_locality = data.pivot_table(
      index='locality_name', 
      values='ceiling_height', 
      aggfunc=['median', 'count']
)


# In[70]:


display(depend_ceiling_locality)


#     Потолки сильно отличаются в разных городах

# In[71]:


def ceiling_cat(df):
    if df < 1:
        return 'меньше 15м'
    elif 1 <= df < 2:
        return 'от 1 до 2 м'
    elif 2 <= df < 3:
        return'от 2 до 3 м'
    elif 3 <= df < 4:
        return 'от 3 до 4 м'
    elif 4 <= df < 5:
        return 'от 4 до 5 м'
    elif 5 <= df < 6:
        return 'от 5 до 6 м'
    elif 6 <= df < 7:
        return 'от 6 до 7 м'
    elif 7 <= df < 8:
        return 'от 7 до 8 м'
    elif 8 <= df < 9:
        return 'от 8 до 9 м'
    elif 9 <= df < 10:
        return 'от 9 до 10 м'
    elif 10 <= df < 11:
        return 'от 10 до 11 м'
    else:
        return'больше 11 м'


# In[72]:


data['ceiling_category'] = data['ceiling_height'].apply(ceiling_cat)


#     Добавляю категории по высоте потолков

# In[73]:


depend_rooms_price = data.pivot_table(
      index='ceiling_category', 
      values=['last_price', 'square_price'], 
      aggfunc={'last_price' : ['median', 'count'], 
              'square_price' : 'median'}
)


# In[74]:


display(depend_rooms_price)


#     Убраю обявления со значениями высоты потолков меньше 2м и больше 10м, это не сильно повлияет на статистку

# In[75]:


good_data = good_data.query('2 < ceiling_height <= 6')


# In[76]:


display(good_data['ceiling_height'].describe())


#     Разница стала 0.03, и больше нетнеадекватно высоких и низких потолков

# In[77]:


good_data['ceiling_height'].hist(bins = 300)
plt.show()


# In[78]:


good_data.boxplot(column='ceiling_height')
plt.show()


# In[79]:


good_data[['last_price', 'total_area','rooms', 'ceiling_height' ]].describe().loc[['min', 'max']]


# # Время продажи квартиры

# In[80]:


data['days_exposition'].hist(bins = 200, range=(0, 100))
#пики на 45, 60, 90
plt.show()


#     На графике заметны явные пики на 7, 30, 45, 60, 90 дней. Скорее всего в эти дни объявления закрывались сервисом.
#     Также заметны полосы, с чем они связаны сказать сложно

# In[81]:


display(data['days_exposition'].sort_values().value_counts().head(20))


# In[82]:


data['days_exposition'].hist(bins = 500)

plt.show()


# In[83]:


display(data['days_exposition'].describe())# ср от мед отличается почти в 2 раза


#     Большая разница между средним и медианой, минимум 1 день, максимум 1580 дней - 4 с лишним года

# In[84]:


#plt.ylim(0, 20)
data.boxplot(column='days_exposition')
plt.show()


# In[85]:


def day_cat(df):
    if df < 15:
        return 'меньше 2 недель'
    if 15 < df < 30:
        return 'от 2х недель до месяца'
    if 30 < df < 60:
        return 'от 1 до 2х месяцев'
    if 60 < df < 120:
        return 'от 2х до 4х месяцев'
    if 120 < df < 180:
        return 'от 4 до 6 месяцев'
    if 180 < df < 365:
        return 'от полугода до года'
    if 365 < df < 730:
        return 'от года до двух лет'
    if 730 < df < 1095:
        return 'от двух до трех лет'
    if 1095 > df:
        return'больше 3х лет'    


#     Категории по сроку публикации объявления

# In[86]:


data['day_category'] = data['days_exposition'].apply(day_cat)


#     Чуть больше 500 и выше - выбросы

# In[87]:


depend_days_exposition_price = data.pivot_table(
      index='day_category', 
      values=['last_price', 'square_price'], 
      aggfunc={'last_price' : ['median', 'count'], 
              'square_price' : 'median'}
)


# In[88]:


display(depend_days_exposition_price)


# In[89]:


good_data = good_data.query('7 < days_exposition < 1095')
#быстрее 7 дней слишком быстро, дольше 3 лет - слишком долго


#     Убираю объявления меньше 7 дней, и больше 1095 дней(3 года)

# In[90]:


good_data['days_exposition'].hist(bins = 500)
plt.show()


#     Убрал максимум без потери значений

# In[91]:


#plt.ylim(0, 20)
good_data.boxplot(column='days_exposition')
plt.show()


#     Часть выбросов осталось

# In[92]:


display(good_data['days_exposition'].describe())


#     Разница стала чуть меньше

# ### Зависимость цены от этажа

# In[93]:


median_price_floor = good_data.pivot_table(
    index= 'floor_info',
    values= "last_price",
    aggfunc='median'
)
display(median_price_floor)# первый дешевле, потом последний, остальные дороже


#     В среднем стоимость квариры на первом этаже ниже всего, затем квартиры на последнем, после все остальные. Скорее всего связано с возможными коммунальными проблемами(протекание крыши, проблемы с подвалом) 

# ### Зависимость цены от количества комнат

# In[94]:


median_price_rooms = good_data.pivot_table(
    index= 'rooms',
    values= "last_price",
    aggfunc=['median', min, max, 'mean']
).astype(int)

median_price_rooms.set_axis(['median', 'min_price', 'max_price', 'mean_price'], 
                             axis='columns', 
                             inplace=True
)
display(median_price_rooms)# понятно больше - дороже, хотя мах +- одинаковый


#     Средняя стоимость квартиры меняется с количеством комнат, чем их больше - тем дороже

# In[95]:


median_price_rooms.plot(title = 'Комнаты\Цены',grid=True, legend=True)
plt.show()


# ### Зависимость цены от расстояния до центра

# In[96]:


median_price_center = good_data.pivot_table(
    index= 'city_center_km',
    values= "last_price",
    aggfunc=['median', min, max, 'mean']
).astype(int)
median_price_center.set_axis(['median', 'min_price', 'max_price', 'mean_price'], 
                             axis='columns', 
                             inplace=True
)
display(median_price_center) #ближе к центру - дороже


#     В общем средняя цена меняется от большего к меньшему, удаляясь от центра, хотя есть исключения

# In[97]:


median_price_center.plot(title = 'Комнаты\Цены',grid=True, legend=True)
plt.show()


# ### Зависимость цены от дня недели публикации

# In[98]:


median_price_day = good_data.pivot_table(
    index= 'week',
    values= "last_price",
    aggfunc=['median', min, max]
)

display(median_price_day) #более менее одинаково


#     Средняя цена почти не меняется от дня недели

# ### Зависимость цены от месяца публикации

# In[99]:


median_price_month = good_data.pivot_table(
    index= 'month',
    values= "last_price",
    aggfunc=['median', min, max]
)

display(median_price_month) # 10 и 12 самый высокий минимум


#     Средняя цена почти не зависит от месяца публикации

# ### Зависимость цены от года публикации

# In[100]:


median_price_year = good_data.pivot_table(
    index= 'year',
    values= "last_price",
    aggfunc=['median', min, max, 'count']
)

display(median_price_year) #медиана снизилась, появилось больше дешевых квартир, \
#минимум снизился, мах почти не изменился


#     Учитвая количество обявлений в 2014 году, нельзя делать какие-то выводы. Начиная с 2015 года средняя цена немного ниже 5 млн.

# ### Топ-10

# In[101]:


top_10_cities = good_data['locality_name'].value_counts().head(10)
cities = list(dict(top_10_cities).keys())
display(cities)

top_cities = good_data.query('locality_name in @cities')

zztop_10_cities = top_cities.pivot_table(
    index='locality_name', 
    values=['square_price', 'last_price'], 
    aggfunc={'last_price' : [min, max], 
              'square_price' : 'mean'}
)
display(zztop_10_cities.sort_values(by=('last_price', 'max'), ascending=False), 
     zztop_10_cities.sort_values(by=('last_price', 'min'), ascending=True)
     )


#     Город с самой дорогой средней стоимостью жилья - Санкт-Петербург, с самой низкой - Выборг

# ### Расстояние до центра

# In[102]:


my_city = ['Санкт-Петербург']
saintp = good_data.query('locality_name in @my_city')


#     Фильтр по городу

# In[103]:


saintp_price_center = saintp.pivot_table(
    index= 'city_center_km',
    values= "last_price",
    aggfunc=['mean', 'median', min, max]
).astype('int')
display(saintp_price_center)


#     Нашёл стоимость за каждый километр

# In[104]:


saintp_price_center.set_axis(['mean_price', 'median', 'min_price', 'max_price'], 
                             axis='columns', 
                             inplace=True
)
saintp_price_center.plot(style='o-', 
                         title  = 'Цена\Удаленность',
                         figsize = (12, 12)
)
plt.legend(loc='upper right')
plt.show()


#     После 3 км, цена выравнивается, считаю это границей центра города

# In[105]:


center_of_city = good_data.query('city_center_km < 3')


#     Отдельно выделяю центр

# ### Анализ центра города

# In[106]:


center_data_total_area = center_of_city.pivot_table(index='city_center_km', 
                           values= 'total_area', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_data_total_area)
#число комнат, этаж, удалённость от центра, дата размещения объявления


#     Площадь при удалении от центра уменьшается

# In[107]:


center_data_last_price = center_of_city.pivot_table(index='city_center_km', 
                           values= 'last_price', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_data_last_price)


#     Цена уменьшается при удалении от центра

# In[108]:


center_data_rooms = center_of_city.pivot_table(index='city_center_km', 
                           values= 'rooms', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_data_rooms)


#     Среднее кличество комнат почти не меняетя, максимальное количество меняется неравномерно 

# In[109]:


center_data_ceiling_height = center_of_city.pivot_table(index='city_center_km', 
                           values= 'ceiling_height', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_data_ceiling_height)


#     Средняя высота потолков почти не меняется, а максимальная меняется неравномерно 

# In[110]:


center_price_rooms = center_of_city.pivot_table(index='rooms', 
                           values= 'last_price', 
                           aggfunc=['mean', 'median', min, max]
).astype('int')


center_price_rooms.set_axis(['mean_price', 'median_price', 'min_price', 'max_price'], 
                             axis='columns', 
                             inplace=True
)
display(center_price_rooms)


#     Изменение цены - линейное, больше комнат - квартира дороже

# In[111]:


center_price_rooms.plot(title = 'Комнаты\Цены',grid=True, legend=True)
plt.show()


# In[112]:


center_price_floor = center_of_city.pivot_table(index='floor_info', 
                           values= 'last_price', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_price_floor)


#     Первый этаж самый дешевый, затем - последний и прочие

# In[113]:


center_price_km = center_of_city.pivot_table(index='city_center_km', 
                           values= 'last_price', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_price_km)


# In[114]:


center_price_date = center_of_city.pivot_table(index='first_day_exposition', 
                           values= 'last_price', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_price_date)


# In[115]:


center_price_year = center_of_city.pivot_table(index='year', 
                           values= 'last_price', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_price_year)


#     В среднем квартиры дешевели от года к году, но максимальные цены почти не менялись

# In[116]:


center_price_month = center_of_city.pivot_table(index='month', 
                           values= 'last_price', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_price_month)


#     В зависимости от месяца есть разница в среднем, хотя не значительно, максимальная почти не менялась

# In[117]:


center_price_day = center_of_city.pivot_table(index='week', 
                           values= 'last_price', 
                           aggfunc=['mean', 'median', min, max, 'count']
).astype('int')
display(center_price_day)


#     Зависимость от дней недели средняя цена почти одинаковая.
#     Стоимость кваартир в центре города и в городе в целом имеют примерно одинаковые показатели зависимости от изученных параметров, больше клмнат - дороже, хотя стоимость в центре выше, чем по городу

# ## Общий вывод

# В ходе исследования я выявил зависимость цен на квартиры от количества комнат(больше комнат - выше цена), расстояния от центра(в центре стоимлсть выше), от того на каком этаже распологается квартира(первый самый дешевый, самые дорогие - между первым и последним этажами). В среднем цена на квартиры от года к году снижается, из-за появления большего количества недорогого жилья, при этим максимальная цена на квартиры остается примерно на одном уровне, не зависимо от месторасположения.
# в данных есть выбросы и пропуски, часть из них можно было бы избежать при правильном заполнении.
# 

# In[118]:


center_of_city_final_price = center_of_city.pivot_table(index='rooms', 
                           values= 'last_price', 
                           aggfunc='mean'
)

good_data_final_price = good_data.pivot_table(index='rooms', 
                           values= 'last_price', 
                           aggfunc='mean'
)
center_of_city_final_price.set_axis(['center_price'], 
                             axis='columns', 
                             inplace=True
)
center_of_city_final_total_area = center_of_city.pivot_table(index='rooms', 
                           values= 'total_area', 
                           aggfunc='mean'
)

good_data_final_total_area = good_data.pivot_table(index='rooms', 
                           values= 'total_area', 
                           aggfunc='mean'
)

center_of_city_final_total_area.set_axis(['center_total_area'], 
                             axis='columns', 
                             inplace=True
)
center_of_city_final_ceil = center_of_city.pivot_table(index='rooms', 
                           values= 'ceiling_height', 
                           aggfunc='mean'
)

good_data_final_ceil = good_data.pivot_table(index='rooms', 
                           values= 'ceiling_height', 
                           aggfunc='mean'
)

center_of_city_final_ceil.set_axis(['center_ceiling_height'], 
                             axis='columns', 
                             inplace=True
)

ax = center_of_city_final_price.plot(title = 'Комнаты\Цены',grid=True, legend=True)
good_data_final_price.plot(ax=ax, title = 'Комнаты\Цены',grid=True, legend=True)

ax1= center_of_city_final_total_area.plot(title = 'Комнаты\Площадь',grid=True, legend=True)
good_data_final_total_area.plot(ax=ax1, title = 'Комнаты\Площадь',grid=True, legend=True)

ax2 = center_of_city_final_ceil.plot(title = 'Комнаты\Потолки',grid=True, legend=True)
good_data_final_ceil.plot(ax=ax2, title = 'Комнаты\Потолки',grid=True, legend=True)

center_of_city_final = center_of_city.pivot_table(index='rooms', 
                           values= ['last_price','total_area','ceiling_height'],
                           aggfunc='mean'
)
good_data_final = good_data.pivot_table(index='rooms', 
                           values= ['last_price','total_area', 'ceiling_height'], 
                           aggfunc='mean'
)
display(center_of_city_final.round(2), good_data_final.round(2))#, sep='\n')


# In[119]:


display('Вся выборка:')
data[['last_price', 'total_area','rooms', 'ceiling_height' ]].describe().loc['50%']


# In[120]:


display('Центр СПб:')
center_of_city[['last_price', 'total_area','rooms', 'ceiling_height' ]].describe().loc['50%']

