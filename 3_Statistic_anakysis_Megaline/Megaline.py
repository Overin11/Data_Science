#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Изучение-общей-информации" data-toc-modified-id="Изучение-общей-информации-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Изучение общей информации</a></span></li><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Подготовка данных</a></span></li><li><span><a href="#Посчитаю-для-каждого-пользователя:" data-toc-modified-id="Посчитаю-для-каждого-пользователя:-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Посчитаю для каждого пользователя:</a></span><ul class="toc-item"><li><span><a href="#Количество-сделанных-звонков-и-израсходованных-минут-разговора-по-месяцам:" data-toc-modified-id="Количество-сделанных-звонков-и-израсходованных-минут-разговора-по-месяцам:-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Количество сделанных звонков и израсходованных минут разговора по месяцам:</a></span></li><li><span><a href="#Количество-отправленных-сообщений-по-месяцам:" data-toc-modified-id="Количество-отправленных-сообщений-по-месяцам:-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Количество отправленных сообщений по месяцам:</a></span></li><li><span><a href="#Объем-израсходованного-интернет-трафика-по-месяцам:" data-toc-modified-id="Объем-израсходованного-интернет-трафика-по-месяцам:-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Объем израсходованного интернет-трафика по месяцам:</a></span></li><li><span><a href="#Помесячную-выручку-с-каждого-пользователя:" data-toc-modified-id="Помесячную-выручку-с-каждого-пользователя:-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Помесячную выручку с каждого пользователя:</a></span></li></ul></li><li><span><a href="#Анализ-данных" data-toc-modified-id="Анализ-данных-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Анализ данных</a></span></li><li><span><a href="#Проверка-гипотез" data-toc-modified-id="Проверка-гипотез-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Проверка гипотез</a></span><ul class="toc-item"><li><span><a href="#Cредняя-выручка-пользователей-тарифов-«Ультра»-и-«Смарт»-различаются" data-toc-modified-id="Cредняя-выручка-пользователей-тарифов-«Ультра»-и-«Смарт»-различаются-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Cредняя выручка пользователей тарифов «Ультра» и «Смарт» различаются</a></span></li><li><span><a href="#Cредняя-выручка-пользователи-из-Москвы-отличается-от-выручки-пользователей-из-других-регионов." data-toc-modified-id="Cредняя-выручка-пользователи-из-Москвы-отличается-от-выручки-пользователей-из-других-регионов.-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Cредняя выручка пользователи из Москвы отличается от выручки пользователей из других регионов.</a></span></li></ul></li><li><span><a href="#Вывод" data-toc-modified-id="Вывод-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Вывод</a></span></li></ul></div>

# # Определение перспективного тарифа для телеком-компании
# Вы аналитик компании «Мегалайн» — федерального оператора сотовой связи. Клиентам предлагают два тарифных плана: «Смарт» и «Ультра». Чтобы скорректировать рекламный бюджет, коммерческий департамент хочет понять, какой тариф приносит больше денег.
# Вам предстоит сделать предварительный анализ тарифов на небольшой выборке клиентов. В вашем распоряжении данные 500 пользователей «Мегалайна»: кто они, откуда, каким тарифом пользуются, сколько звонков и сообщений каждый отправил за 2018 год. Нужно проанализировать поведение клиентов и сделать вывод — какой тариф лучше.

# Описание тарифов:
# 
# # Тариф «Смарт»
# Ежемесячная плата: 550 рублей
# Включено 500 минут разговора, 50 сообщений и 15 Гб интернет-трафика
# 
# Стоимость услуг сверх тарифного пакета: 
# - минута разговора: 3 рубля («Мегалайн» всегда округляет вверх значения минут и мегабайтов. Если пользователь проговорил всего 1 секунду, в тарифе засчитывается целая минута); 
# - сообщение: 3 рубля; 
# - 1 Гб интернет-трафика: 200 рублей.
# 
# # Тариф «Ультра»
# Ежемесячная плата: 1950 рублей
# Включено 3000 минут разговора, 1000 сообщений и 30 Гб интернет-трафика
# 
# Стоимость услуг сверх тарифного пакета: 
# - минута разговора: 1 рубль; 
# - сообщение: 1 рубль; 
# - 1 Гб интернет-трафика: 150 рублей.
# 
#     Примечание:
# «Мегалайн» всегда округляет секунды до минут, а мегабайты — до гигабайт. Каждый звонок округляется отдельно: даже если он длился всего 1 секунду, будет засчитан как 1 минута.
# Для веб-трафика отдельные сессии не считаются. Вместо этого общая сумма за месяц округляется в бо́льшую сторону. Если абонент использует 1025 мегабайт в этом месяце, с него возьмут плату за 2 гигабайта.

# ##   Изучение общей информации

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st


# In[2]:


calls = pd.read_csv('/datasets/calls.csv')
internet = pd.read_csv('/datasets/internet.csv')
messages = pd.read_csv('/datasets/messages.csv')
users = pd.read_csv('/datasets/users.csv')
tariffs = pd.read_csv('/datasets/tariffs.csv')


# <div class="alert alert-success">
# <b>Комментарий ревьюера v1:</b>
# 
# 
# Здорово, что загрузка библиотек и данных сгруппированы в начале проекта. 
# 
# </div>
# 

# ## Подготовка данных

# In[3]:


calls.info()


# In[4]:


print(calls[calls['duration'] == 0]['duration'].count())
calls['duration'].hist(bins= 100)
plt.show()


#     Всего 200 тыс. звонков, из них 39 тыс. продолжительностью 0 минут, скорее всего это пропущенные входящие звонки (звонок есть, продолжительность 0)
#     Убрав их потеряю много данных, оставляю

# In[5]:


print(internet[internet['mb_used'] == 0]['mb_used'].count(), internet.count(), sep='\n')
internet['mb_used'].hist(bins=100)
plt.show()


#     Всего зафиксировано 150 тыс. сессий, из них почти в 20 тыс. объём 0 мб. Либо сессия была очень короткой и округлилась в меньшую сторону, либо это связвно с техническими особенностями оператора
#     Убирать эти данные не буду

# In[6]:


calls['call_date'] = pd.to_datetime(  #даты к формату дат
    calls['call_date'], 
    format='%Y-%m-%d'
)
users['reg_date'] = pd.to_datetime(
    users['reg_date'], 
    format='%Y-%m-%d'
)
internet['session_date'] = pd.to_datetime(
    internet['session_date'], 
    format='%Y-%m-%d'
)
messages['message_date'] = pd.to_datetime(
    messages['message_date'], 
    format='%Y-%m-%d'
)


#     Привел даты к удобному формату

# In[7]:


def duble(smthng):
    print(smthng.duplicated().sum())


# In[8]:


datum = [calls, users, internet, messages]
for i in datum:
    duble(i)


#     Повторов не было

# ## Посчитаю для каждого пользователя: 

# ### Количество сделанных звонков и израсходованных минут разговора по месяцам:
# 

# In[9]:


calls['duration'] = np.ceil(calls['duration'])


#     Округлил в большую сторону, как указано в тарифе

# In[10]:


calls['month'] = calls['call_date'].dt.month_name()


#     Добавил столбец с месяцами

# ### Количество отправленных сообщений по месяцам:
# 

# In[11]:


messages['month'] = messages['message_date'].dt.month_name()


# ### Объем израсходованного интернет-трафика по месяцам:

# In[12]:


internet['month'] = internet['session_date'].dt.month_name()


# ### Помесячную выручку с каждого пользователя:

# In[13]:


calls_monthly = calls.pivot_table(index=['user_id', 'month'], 
                                  values=['id','duration'], 
                                  aggfunc= {'id':'count',
                                            'duration':'sum'
                                            }  
)
calls_monthly.columns = ['duration', 'number_of_calls']


#     Посчитал количество звонков и общую длительность разговоров по месячно для каждого пользователя

# In[14]:


msg_monthly = internet.pivot_table(index=['user_id', 'month'], 
                                  values=['id'], 
                                  aggfunc='count'
)
msg_monthly.columns = ['number_of_messages']


#     Посчитал количество сообщений в месяц для каждого пользователя

# In[15]:


int_monthly = internet.pivot_table(index=['user_id', 'month'], 
                                  values= ['id', 'mb_used'], 
                                  aggfunc= {'id':'count',
                                            'mb_used':'sum'
                                            }                                    
)
int_monthly.columns = ['number_of_sessions', 'mb_used']


#     Аналогично для интернет трафика

# In[16]:


result = calls_monthly.merge(msg_monthly, 
                             on=['user_id', 'month'], 
                             how='outer'
)
result = result.merge(int_monthly,
                      on=['user_id', 'month'], 
                      how='outer'
)


#     Объединил в одну таблицу

# In[17]:


result = result.fillna(0)


#     Заполнил пропуски 0

# In[18]:


def full_gigabytes(df):                 
    return np.ceil(df / 1024)

result['full_gigabytes'] = result['mb_used'].apply(full_gigabytes)


#     Округляю трафик до Гб как по тарифу

# In[19]:


result = result.join(users.set_index('user_id'), on='user_id')

tariffs.set_axis(['messages_included', 'mb_per_month_included', 'minutes_included',
       'rub_monthly_fee', 'rub_per_gb', 'rub_per_message', 'rub_per_minute',
       'tariff'], axis= 'columns', inplace=True)
print(tariffs.columns)                                              #поменял имя стобца для join`а
result = result.join(tariffs.set_index('tariff'), on='tariff')


#     Добавил данные о пользователях, в т. ч. данные о тарифе

# In[20]:


def income(df):
    payment = 0
    if df['tariff'] == 'smart':
        if df['duration'] > df['minutes_included']:
            payment = (df['duration'] - df['minutes_included']) * df['rub_per_minute']
        if df['number_of_messages'] > df['messages_included']:
            payment += (df['number_of_messages'] - df['messages_included']) * df['rub_per_message']
        if df['full_gigabytes'] > np.ceil(df['mb_per_month_included'] / 1024):
            payment += (df['full_gigabytes'] - np.ceil(df['mb_per_month_included'] / 1024)) * df['rub_per_gb']
        return (df['rub_monthly_fee'] + payment)
    if df['tariff'] == 'ultra':
        if df['duration'] > df['minutes_included']:
            payment = (df['duration'] - df['minutes_included']) * df['rub_per_minute']
        if df['number_of_messages'] > df['messages_included']:
            payment += (df['number_of_messages'] - df['messages_included']) * df['rub_per_message']
        if df['full_gigabytes'] > np.ceil(df['mb_per_month_included'] / 1024):
            payment += (df['full_gigabytes'] - np.ceil(df['mb_per_month_included'] / 1024)) * df['rub_per_gb']
        return (df['rub_monthly_fee'] + payment)


# In[21]:


result.columns


# In[22]:


result['income'] = result.apply(income, axis=1)


#     Посчитал помесячную выручку с каждого пользователя

# In[23]:


result = result.reset_index()


# ## Анализ данных

# In[24]:


result_smart = result[result['tariff'] == 'smart']
result_ultra = result[result['tariff'] == 'ultra']


# In[25]:


smart_data = result_smart.pivot_table(index = 'tariff', 
                         values = ['duration', 'number_of_messages', 'full_gigabytes'], 
                         aggfunc = ['mean', 'var', 'std']
                         ).astype(int)

ultra_data = result_ultra.pivot_table(index = 'tariff', 
                         values = ['duration', 'number_of_messages', 'full_gigabytes'], 
                         aggfunc = ['mean', 'var', 'std']
                         ).astype(int)
print('Тариф smart')
for i in range(9):
    display(*smart_data.columns[i], smart_data.iloc[0, i])
    print('\n')
print('Тариф ultra')
for i in range(9):
    display(*ultra_data.columns[i], ultra_data.iloc[0, i])
    print('\n')


#     Посчитал среднее количество, дисперсию и стандартное отклонение

# In[26]:


ax = result.loc[result['tariff'] == 'smart']['duration'].plot(   #ось
    kind='hist',                      #тип графика
    y='smart_duration',               #ось У
    histtype='step',                  #тип гистограммы
    bins=100,                         #корзина
    linewidth=2,                      #ширина линии
    alpha=0.6,                        #прозрачность, чтобы видеть пересечение
    label='smart',                    #подпись
    legend=True
)
result.loc[result['tariff'] == 'ultra']['duration'].plot(
    kind='hist',
    y='ultra_duration',
    histtype='step',
    bins=100,
    linewidth=2,
    alpha=0.6,
    label='ultra',
    ax=ax,                            #ось равно оси первого, чтобы были на одном
    grid=True,                        #сетка
    legend=True,                      #легенда
    title  = 'Длительность разговора', 
)
plt.show()


# In[27]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(16, 5))
sns.distplot(result.loc[result['tariff'] == 'smart']['duration'],ax=ax_hist)
sns.boxplot(result.loc[result['tariff'] == 'smart']['duration'], ax=ax_box)
plt.show()
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(16, 5))
sns.distplot(result.loc[result['tariff'] == 'ultra']['duration'],ax=ax_hist)
sns.boxplot(result.loc[result['tariff'] == 'ultra']['duration'], ax=ax_box)
plt.show()


#     Гистограмма длительности разгоровов

# In[28]:


ax = result.loc[result['tariff'] == 'smart']['mb_used'].plot(   #ось
    kind='hist',                    #тип графика
    y='smart_mb_used',              #ось У
    histtype='step',                #тип гистограммы
    bins=100,                       #корзина
    linewidth=2,                    #ширина линии
    alpha=0.6,                      #прозрачность, чтобы видеть пересечение
    label='smart',                  #подпись
    legend=True, 
    title  = 'Интернет трафик'
)
result.loc[result['tariff'] == 'ultra']['mb_used'].plot(
    kind='hist',
    y='ultra_mb_used',
    histtype='step',
    bins=100,
    linewidth=2,
    alpha=0.6,
    label='ultra',
    ax=ax,                          #ось равно оси первого, чтобы были на одном
    grid=True,                      #сетка
    legend=True,                    #легенда
)
plt.show()


# In[29]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(16, 5))
sns.distplot(result.loc[result['tariff'] == 'smart']['mb_used'],ax=ax_hist)
sns.boxplot(result.loc[result['tariff'] == 'smart']['mb_used'], ax=ax_box)
plt.show()
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(16, 5))
sns.distplot(result.loc[result['tariff'] == 'ultra']['mb_used'],ax=ax_hist)
sns.boxplot(result.loc[result['tariff'] == 'ultra']['mb_used'], ax=ax_box)
plt.show()


#     Гистограмма израсходованного интернет траффика

# In[30]:


ax = result.loc[result['tariff'] == 'smart']['number_of_messages'].plot(   #ось
    kind='hist',                       #тип графика
    y='smart_messages',                #ось У
    histtype='barstacked',             #тип гистограммы
    bins=100,                          #корзина
    linewidth=2,                       #ширина линии
    alpha=0.6,                         #прозрачность, чтобы видеть пересечение
    label='smart',                     #подпись
    legend=True
)
result.loc[result['tariff'] == 'ultra']['number_of_messages'].plot(
    kind='hist',
    y='ultra_messages',
    histtype='barstacked',
    bins=100,
    linewidth=2,
    alpha=0.6,
    label='ultra',
    ax=ax,                             #ось равно оси первого, чтобы были на одном
    grid=True,                         #сетка
    legend=True,                       #легенда
    title  = 'Количество сообщений', 
)
plt.show()


# In[31]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(16, 5))
sns.distplot(result.loc[result['tariff'] == 'smart']['number_of_messages'],ax=ax_hist)
sns.boxplot(result.loc[result['tariff'] == 'smart']['number_of_messages'], ax=ax_box)
plt.show()
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(16, 5))
sns.distplot(result.loc[result['tariff'] == 'ultra']['number_of_messages'],ax=ax_hist)
sns.boxplot(result.loc[result['tariff'] == 'ultra']['number_of_messages'], ax=ax_box)
plt.show()


#     Гистограмма сообщений

# In[32]:


print('Средние значения тарифа Ultra')
result.loc[result['tariff'] == 'ultra'][['duration', 'number_of_messages','mb_used']].describe().loc['50%']


# In[33]:


print('Средние значения тарифа Smart')
result.loc[result['tariff'] == 'smart'][['duration', 'number_of_messages','mb_used']].describe().loc['50%']


#     1. Звонки:
#     Пользователи тарифа смарт чаще тратят на разговоры около 400-550 минут в месяц, пользователи тарифа ультра одинаково часто говорят примерно от 250 до 800 минут в месяц, хотя у них много звонков 0 минут(пропущенных).
#     2. Интернет:
#     На тарифе смарт у пользователей чаще уходит на интернер трафик от 13 до 20 Гб(лимит по тарифу 15, самая частая величина)
#     У пользователей ультра пик примерно между 13 и 23 Гб, хотя количество пользователей потративших меньше 13 или больше 23 Гб отличается незначительно.
#     3. Сообщения:
#     У пользователей смарт пик - 50(лимит тарифа), в общем они тратят от 40 до 70 сообщения в месяц чаще всего пользователи ультра отправляли гораздо меньше сообщений, чаще всего от 30 до 45, примерно

# ## Проверка гипотез

#     Нулевая гипотеза всегда формулируется так, чтоб использовать знак равенства. Исходя из неё формулирую альтернативную гипотезу 
#     Для проверки гипотез использовал t-критерий Стьюдента, потому что этот критерий оценивает насколько различаются средние выборки, и я считаю, что в данных нет выбросов, т.к. они сильно влияют на t-критерий

# ### Cредняя выручка пользователей тарифов «Ультра» и «Смарт» различаются
# 

#     Нулевая гипотеза: средняя выручка пользователей тарифов «Ультра» и «Смарт» равны
#     Альтернативная гипотеза: средняя выручка пользователей тарифов «Ультра» и «Смарт» различаются

# In[34]:


ultra_income_mean = np.random.choice(result.loc[result['tariff'] == 'ultra']['income'], 40)
smart_income_mean = np.random.choice(result.loc[result['tariff'] == 'smart']['income'], 40)


#     Cлучайная выборка

# In[35]:


check_hypo = st.ttest_ind(ultra_income_mean, smart_income_mean, equal_var=True)
print('p-значение: ', check_hypo.pvalue)
alpha = 0.05  # критический уровень статистической значимости
# если p-value окажется меньше него - отвергнем гипотезу
if check_hypo.pvalue < alpha:
    print('Отвергаем нулевую гипотезу')
else:
    print('Не получилось отвергнуть нулевую гипотезу')


#     Вывод: По данной выборке можно сказать, что выручка по тарифам отличается, у пользователей тарифа 'Ultra' = чуть больше 2000, у 'Smart' - почти 1300

# ### Cредняя выручка пользователи из Москвы отличается от выручки пользователей из других регионов.

#     Нулевая гипотеза: средняя выручка пользователей из Москвы и из регионов равны
#     Альтернативная гипотеза: средняя выручка пользователей из Москвы и из регионов различаются

# In[36]:


moscow_income_mean = np.random.choice(result.loc[result['city'] == 'Москва']['income'], 40)
region_income_mean = np.random.choice(result.loc[result['city'] != 'Москва']['income'], 40)


#     Случайная выборка

# In[37]:


check_hypo_city = st.ttest_ind(moscow_income_mean, region_income_mean, equal_var=True)
print('p-значение: ', check_hypo_city.pvalue)
alpha = 0.05  # критический уровень статистической значимости
# если p-value окажется меньше него - отвергнем гипотезу
if check_hypo_city.pvalue < alpha:
    print('Отвергаем нулевую гипотезу')
else:
    print('Не получилось отвергнуть нулевую гипотезу')   


# In[38]:


income_mean = result.pivot_table(index='tariff', 
                                values='income', 
                                aggfunc='mean')
print('Средняя выручка по каждому тарифу')
print(income_mean)


#     Вывод: Гипотезу о том, что Пользователи из Москвы тратят на связь больше, чем пользователи регионов не удалось опровергнуть. Поэтому считаем, что их расходы равны

# ## Вывод
# 

#     Исходя из полученных данных средняя прибыль больше на тарифе 'Ultra'. На тарифе 'Smart' пользователи чаще выходят за лимит по собощениям, многие тратят больше минут, чем включено в тариф, и много пользователей расходуют трафик сверх лимита. На тарифе 'Ultra', только по инетернет трафику есть превышения лимита.
# 
#         Средние значения тарифа Ultra:
#     
# - duration                518.00
# - number_of_messages       38.00
# - mb_used               19308.01
# 
# 
#         Средние значения тарифа Smart:
#     
# - duration                422.00
# - number_of_messages       51.00
# - mb_used               16506.84
# 
# 
# 
#       В ходе исследования выяснилось, что пользователи из Москвы и других регионов, обоих тарифов тратят в среднем одинаково

# In[39]:


ax = result.loc[result['tariff'] == 'smart']['income'].plot(   #ось
    kind='hist',                       #тип графика
    y='smart_incomes',                 #ось У
    histtype='barstacked',             #тип гистограммы
    bins=80,                           #корзина
    linewidth=2,                       #ширина линии
    alpha=0.6,                         #прозрачность, чтобы видеть пересечение
    label='smart',                     #подпись
    legend=True
)
result.loc[result['tariff'] == 'ultra']['income'].plot(
    kind='hist',
    y='ultra_income',
    histtype='barstacked',
    bins=80,
    linewidth=2,
    alpha=0.6,
    label='ultra',
    ax=ax,                             #ось равно оси первого, чтобы были на одном
    grid=True,                         #сетка
    legend=True,                       #легенда
    title  = 'Прибыль', 
)
plt.show()


# In[40]:


result.loc[result['tariff'] == 'smart']['income'].describe().loc[['min', 'mean', 'max']]


# In[41]:


result.loc[result['tariff'] == 'ultra']['income'].describe().loc[['min', 'mean', 'max']]


# 1. В данных есть звонки длительностью 0 минут, их много и это, скорее всего, пропущенные.
# 2. Так же присутствуют сессии с 0 потраченных мб, их тоже много. Они связаны и техническими особеностями или минимальным траффиком.
# 3. Длительность разговора у пользователей тарифа 'Ultra' больше, чем у 'Smart'. 518.00 минут в среднем, против 422.00
# 
#     А количество сообщений больше у пользователей 'Smart'. 51 и 38 
#     
#     
#     Пользователи тарифа 'Ultra' траятят 19308 мб траффика в среднем, а 'Smart' - 16506 мб.
# 4. Средняя прибыль по кажому тарифу 'Ultra' - 2070, 'Smart' - 1292
# 5. Пользователи из Москвы и рагионов тратят в среднем одинаково, не зависимо от тарифа

#     Исходя из данных и исследования, я делаю вывод, что лучше продвигать тариф 'Smart', независимо от региона. Пользователи тарифа 'Ultra' при абонентской плате 1950, в среднем тратят 2070, т.е. почти не выходят за месячный лимит по звонкам\сообщениям\траффику. У пользователей 'Smart', при абонентской плате 550, в среднем уходит 1292, т.е. им не хватает месячного лимита, и они готовы заплатить больше за дополнительные минуты\сообщения\трафиик
