#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-info">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Привет еще раз. Спасибо, что доделал работу. Оформление комментариев по работе сохраняется. Только обозначим, что это вторая итерация. 
# 
# </div>

# <div class="alert alert-info">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Привет, Евгений! Спасибо, что прислал задание:) Поздравляю с приближением к концу первого модуля! Меня зовут Слепцов Артем и я буду проверять твой проект) Ты проделал большую работу над проектом. Он выполнен уже на достойном уровне. Однако есть моменты, которые еще можно улучшить. Будет здорово, если ты, надеюсь, не против, если я буду на ты, будешь отвечать на комментарии и участвовать в диалоге. Если обращение на ты неприемлемо, то прошу сообщить. 
# 
# Мои комментарии обозначены пометкой **Комментарий ревьюера**. Далее в файле ты сможешь найти их в похожих ячейках:
#     
# <div class="alert alert-success">Если фон комментария зелёный - всё сделано правильно. Рекомендации укажу таким же цветом;</div>
#         
# <div class="alert alert-warning">Оранжевый - некритичные замечания. Если таких замечаний меньше трех - проект может быть принят без их отработки;</div>
#         
# <div class="alert alert-danger">Красный - нужно переделать. </div>
#         
# Не удаляй эти комментарии и постарайся учесть их в ходе выполнения данного проекта. Свои же комментарии ты можешь обозначать любым заметным способом. 
# 
# </div>

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

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Вступление в работу очень важно. Так ты поясняешь то, чему она посвящена. Здорово, что каждому пункту вводной информации ты уделил внимание. 
# 
# </div>

# <div class="alert alert-warning">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Я заметил, что ячейки в твоей тетрадке начинаются не с 1. Перед отправкой работы рекомендую перезапускать ноутбук, чтобы убедиться, что все ячейки выполняются корректно.
# 
# </div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# В полном перезапуске проекта тебе поможет Kernel => Restart & Run All.
# 
# </div>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
import scipy as scipy
sns.set_style('whitegrid') #('darkgrid')

games = pd.read_csv('/datasets/games.csv')


# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Правильно, что весь импорт ты проводишь в первой ячейке работы. Так твой коллега, запускающий работу, будет в курсе используемых в ней библиотек и сможет при необходимости быстро настроить окружение. 
# 
# </div>

# In[2]:


games.info()


# ## Подготовка данных

# ### Замена названий столбцов

# In[3]:


games.columns = map(lambda x: x.lower(), games.columns)         # стоблцы в нижний регистр


# Перевел названия столбцов к нижнему регистру

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Данный метод позволяет нам автоматизировать процесс приведения названий столбцов к нижнему регистру. Так исключается вероятность опечаток. 
# 
# </div>

# ### Обработка пропусков

# In[4]:


print(games.isna().sum())


# Есть пропуски в названии, годах  выпуска, оценках критиков и пользователей, и рейтинге. 
# 
# Т.к. пропусков в оценках и рейтинге много, нельзя их просто удалить

# In[5]:


games.dropna(subset=['name'], inplace = True)                   # 2 игры вообще без данных
games.dropna(subset=['year_of_release'], inplace = True)        # 269 или 1,6 %, найти можно, но не целесообразно


# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Соглашусь, такую долю строк мы можем убрать из данных.
# 
# </div>

# Убрал небольшую часть пропусков

# In[6]:


print(games['user_score'].isna().sum(), 'строк с NaN')


# In[7]:


#games[['user_score']] = games[['user_score']].fillna('0') 
games[['user_score']] = games[['user_score']].astype(str)       
print(games.loc[games['user_score'] == 'tbd', 'user_score'].count(),  'оценок со статусом tbd') 


# Не стал менять NaN на 0.
# TBD - аббревиатура от английского To Be Determined (будет определено) или To Be Decided (будет решено), поэтому пока игнорирую

# In[8]:


rating_info = games.pivot_table(index=['user_score', 'year_of_release'], 
                                values=['name'], 
                                aggfunc= 'count')
display(rating_info)                                             #tbd  в 2016 году 34


# Оценки tbd заменил на NaN

# In[9]:


games['user_score'] = np.where(games['user_score'] == 'tbd', np.nan, games['user_score']) #заменяю tbd на 0
games[['user_score']] = games[['user_score']].astype(float) # всё ко float

print(games['user_score'].isna().sum(), 'строк с NaN')                        


# <div class="alert alert-danger">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# По своей сути tbd и является Nan. Заменять данные значения на нули не стоит. 
# </div>

# <div class="alert alert-info"> Согласен, заменил на NaN</div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Верное решение. Так нереальные значения не попадут в анализ.
# 
# </div>

# In[10]:


#print(games.sort_values('critic_score', ascending=True).head())
display(games.sort_values('critic_score', ascending=True).head())


# <div class="alert alert-warning">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Таблицы на печать выводи с помощью метода display. print переводит данные в строку, а нам этого не надо. 
# 
# </div>

# <div class="alert alert-info"> Так действительно лучше)</div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Отлично!
# 
# </div>

# Оценок критиков с 0 нет

# In[11]:


#games['critic_score'] = games['critic_score'].fillna(0)


# In[12]:


print(games.isna().sum())


# Пропуски в рейтинге игры можно заполнить вручную, эти данные можно найти, но делать это нецелесообразно(слишком трудёмко)
# 
# Оставляю как есть

# ### Преобразование данных в нужные типы.

# In[13]:


games['year_of_release'] = games['year_of_release'].astype('int64')   #т.к. только год, можно в int так удобнее


# Привел год выпуска к int, т.к. в исследовании используется только год, так удобнее 

# ### Считаю суммарные продажи во всех регионах

# In[14]:


def total_sales(df):                                            #общие продажи
    return (df['na_sales'] + df['eu_sales'] + df['jp_sales'] + df['other_sales'] )

games['total_sales'] = games.apply(total_sales, axis=1)


# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Да, восстановить пропущенные значения мы не можем. Данных для этого недостаточно. Лучше работать с меньшим количеством данных хорошего качества.
#     
# Ошибки в данных устранены. Данные подготовлены к дальнейшему анализу. 
# 
# </div>

# ##  Исследовательский анализ данных

# ### Ранние годы

# In[15]:


early_times = games.pivot_table(index='year_of_release', 
                                values='total_sales', 
                                aggfunc='count'
                                )
early_times.plot(title = 'total sales',  figsize=(12, 7), kind='bar')
fig, ax = plt.subplots()#раньше было меньше

ax.plot(early_times)
ax.grid()

ax.set_xlabel('Год')
ax.set_ylabel('Количество продаж')

plt.show()


# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Как думаешь, с чем связан спад в индустрии последних лет?
# 
# </div>

# <div class="alert alert-info"> Кризис 2009 года, люди в первую очередь экономят на разлечениях, игры в том числе. Меньше продаж - меньше бюджет на разработку новой игры и т. д. Восстановление занимает долгое время</div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Соглашусь с твоим обоснованием.
# 
# </div>

# In[16]:


platform_box_data = games.pivot_table(index=['platform', 'year_of_release'],              # для ящика с усами
                                      values=['total_sales'], 
                                      aggfunc='count'
                                      ).astype(int)
platform_box_data = platform_box_data.reset_index()

plt.figure(figsize=(30, 25), dpi= 90)
sns.boxplot(x='year_of_release', y='total_sales', data=platform_box_data, dodge=True)    # много маленьких ящиков

plt.show()


# По графикам видно, что данных по играм мало, до второй половины 90х

# In[17]:


#games['year_of_release'].hist(bins=90)
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


# Данных по играм за ранние годы немного, поэтому отсекаю платформы с продажами меньше 150

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


# Очень много линий, неинформативно, поэтому делаю для каждого отдельно.

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


# По графикам на 2015 и неполный 2016 года рост показывают Xone, PS4, PC

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Да, именно эти платформы мы и можем назвать перспективными на 2017 год. 
# 
# </div>

# ### Характерный срок платформы

# In[24]:


mean_year = 0
for i in [k for k in set(selected_platforms['platform'])]:
    years = int(platform_data.loc[platform_data['platform'] == i]['year_of_release'].describe().loc[['max']]) - int(platform_data.loc[platform_data['platform'] == i]['year_of_release'].describe().loc[['min']])
    print('Срок эксплуатации платформы {} {:.0f} лет'.format(i, years))
    mean_year += years 
mean_year = mean_year / len(selected_platforms)
print('По выбранным параметрам средний срок службы платформы {:.0f} лет'.format(mean_year))


# По графиками видно, что с момента появления платформы до роста проходит примерно 2-3 года. поэтому есть смысл анализировать не характерный срок, а тот период, когда в среднем у платформы наблюдается значительный рост продаж.
# 
# Беру последние 3 года

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Значение средней продолжительности существования платформы приведено. Круто, что ты привел расчет периода жизни платформ. Однако учитывать актуальные на настоящий момент платформы не стоит. Период их жизни еще продолжается. 
# 
# </div>

# ### Исследование актального периода

# In[25]:


good_platforms = games.query('year_of_release > 2013')


# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Актуальный период назван. Так в рассмотрение попадут только последние поколения платформ, а также будем рассматривать только конечный на данный момент интервал развития игровой индустрии. 
# 
# </div>

# In[43]:


good_platforms_sales = good_platforms.pivot_table(index= 'platform', 
                                 values = 'total_sales', 
                                 aggfunc= 'sum').sort_values(by='total_sales', ascending=False)
display(good_platforms_sales)


# Хоть PC и показывает небольшой уровень общих продаж, по графикам виден рост, как у PS4 и XOne. У остальных падение, что говорит о том, что они скоро исчезнут

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


# По графикам видно, что самый большой уровень продаж у PS4, затем XOne, PC - третий

# In[46]:


display('PS4', 
      good_platforms_sales_ps4[['total_sales']].describe().loc['50%'], 
      'XOne',  
      good_platforms_sales_xone[['total_sales']].describe().loc['50%'], 
      'PC', 
      good_platforms_sales_pc[['total_sales']].describe().loc['50%']
     )


# Медианы

# <div class="alert alert-warning">
# <font size="5"><b>Комментарий ревьюера</b></font>
#  
# Подумай, из-за чего формируется разница между платформами. Старайся не только описывать результат, но и трактовать его. 
# 
# </div>

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

# In[47]:


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

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Результат получен. Подумай, чем он вызван. Здорово, что рассмотрены несколько платформ. 
# 
# </div>

# ### Распределение игр по жанрам

# In[48]:


user_char_genre = good_platforms.pivot_table(index='genre', 
                                                values='total_sales', 
                                                aggfunc='sum'
                                                ).sort_values(by= 'total_sales', ascending=False)
display(user_char_genre)


#     Самые прибыльные Action, Shooter, Sports, Role-Playing
#     Жанры с наименьшеи кровнем продаж Strategy, Puzzle

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Анализ популярности жанров проведен. Однако не стоит забывать, что производство игр в жанрах Action или Shooter обходится сильно дороже, чем производство Puzzle-игр. 
# 
# </div>

# ##  Портрет пользователя каждого региона

# In[53]:


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


# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Попробуй использовать команду pd.set_option('chained_assignment', None) для скрытия данного предупреждения.
# 
# </div>

# ### Самые популярные платформы

# In[55]:


for i in ['na_sales', 'eu_sales', 'jp_sales']:
    display(i, user_char_platform[i].sort_values(ascending=False).head())
    user_char_platform[i].sort_values(ascending=False).head(5).plot(kind='bar')
    plt.show()
    print()


#     В регионах NA и еEU первые места по продажам занимают PS4 и XOne, причем в Европе PS4 лидирует с большим отрывом 130\46 млн копий, а в Северной Америке отстование XOne незначительно 98\81
#     В Японии PS4 на втором месте с 15 млн, а XOne не попал в топ-5 вообще

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Лучше все же использовать столбчатую диаграмму при построении графиков. Все-таки мы работаем с категориальными данными. 
# 
# </div>

# <div class="alert alert-info"> согласен</div>

# ### Самые популярные жанры

# In[59]:


for i in ['na_sales', 'eu_sales', 'jp_sales']:
    display(i, user_char_genre[i].sort_values(ascending=False).head())
    user_char_genre[i].sort_values(ascending=False).head().plot(kind='bar')
    plt.show()
    print()


#     Регионы NA и EU похожи и предпочтениях по жанрам Shooter, Action, Sports занимают первые три строчки(в Северной Америке Shooter на первом, в Европе - Action).
#     В Японии первые 3: Role-Playing, Action, Fighting

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
#  
# Из-за чего японский рынок игр так сильно отличается от других? 
# 
# </div>

# <div class="alert alert-info"> Возможно, культурные особенности</div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Да, вполне возможное обоснование. 
# 
# </div>

# ### Влияние рейтинга ESRB на продажи

# In[61]:


for i in ['na_sales', 'eu_sales', 'jp_sales']:
    display(i, user_char_rating[i].sort_values(ascending=False).head())
    user_char_rating[i].sort_values(ascending=False).head().plot(kind='bar')
    #user_char_rating[user_char_rating[i] == np.nan].sum().plot(kind='bar')
    plt.show()
    print()


# <div class="alert alert-danger">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Обрати внимание - ты не учитываешь игры без рейтинга. В результате огромная часть игр просто выпадает из анализа. Найди способ учесть их при группировке значений по рейтингу.
#     
# </div>

# <div class="alert alert-info"> Добавил рейтинг 'без рейтинга', их действительно много</div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Как думаешь, почему большая часть игр японского региона не имеет рейтинга?
# 
# </div>

#     В Европе и Северной Америке рейтинги распределились M, E, T, E10+, причём примерно в одинаковом количестве.
#     В Японии T, E, M, E10+
#     В NA и EU большей  популярностью пользуются игры со взрослым рейтингом, в JP - с подростковым
#     Во всех региронах много игр вообще не имеют рейтинга

# In[62]:


rating_data = good_platforms.pivot_table(index='rating', 
                                        values = 'total_sales', 
                                        aggfunc = 'sum')
display(rating_data.sort_values(by='total_sales', ascending=False))


#     По продажам самые прибыльные игры со взрослым рейтингом, затем рейтинг для всех от 6, подростковый и всем старше 10 лет

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Портрет типичного пользователя каждого из регионов получен. Приведены все необходимые графики. Здорово, что отмечены индивидуальные особенности каждого региона. 
# 
# </div>

# ## Проверка гипотез

#     Нулевая гипотеза всегда формулируется так, чтоб использовать знак равенства. Исходя из неё формулирую альтернативную гипотезу.
#     Для проверки гипотез использовал t-критерий Стьюдента, потому что этот критерий оценивает насколько различаются средние выборки.
#     А для проверки равенства дисперсий случайных выборок генеральной совокупности применил тест Левена

# ### Средние пользовательские рейтинги платформ Xbox One и PC одинаковые

# In[39]:


user_score_games = games.dropna(subset = ['user_score'], inplace=True)


# Т.к. в user_score оставлял NaN, сейчас их надо убрать, помешают

# In[40]:


#xone_rating_mean = np.random.choice(games.loc[games['platform'] == 'XOne']['user_score'], 40)
#pc_rating_mean = np.random.choice(games.loc[games['platform'] == 'PC']['user_score'], 40)

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


#     Т.к. при проверке ривергли нулевую гипотезу, считаю средние пользовательские рейтинги PC и Xbox One неравными
#     Т.е. среди пользователей популярность платформ отличается

# <div class="alert alert-danger">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Гипотезы сформулированы верно. Не стоит брать только 40 значений в анализ. Используй все данные в тесте. В твоем варианте можно получить абсолютно противоположные результаты. Используй все данные при проверке гипотез.  
# 
# </div>

# <div class="alert alert-info"> Так и есть, спасибо)</div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Так мы используем все данные в анализе, поэтому результат будет надежнее. 
# 
# </div>

# ### Средние пользовательские рейтинги жанров Action и Sports равны

# In[41]:


#action_rating_mean = np.random.choice(games.loc[games['genre'] == 'Sports']['user_score'], 40)
#sport_rating_mean = np.random.choice(games.loc[games['genre'] == 'Action']['user_score'], 40)

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
alpha = 0.05  # критический уровень статистической значимости
# если p-value окажется меньше него - отвергнем гипотезу
if check_hypo.pvalue < alpha:
    print('Отвергаем нулевую гипотезу')
else:
    print('Не получилось отвергнуть нулевую гипотезу') 


#     Т.к. не удалось опровергнуть нулевую гипотезу, считаю средние пользовательские рейтинги жанров Action и Sports равны
#     Т.е. делать ставку на один из этих рейтингов не стоит, надо сравнить остальные

# <div class="alert alert-danger">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Исправь аналогичную помарку, используй все данные при проверке гипотез. Второе, ты используешь данные первой гипотезы о платформах при проведении тестирования, а не данные по жанрам. Исправь данную неточность. 
# 
# </div>

# <div class="alert alert-info"> Здесь без изменений</div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Помарки исправлены. Тесты стоит проводить максимально внимательно. 
# 
# </div>

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

# <div class="alert alert-danger">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Финальный вывод и есть главный результат твоей работы. Стоит писать его подробно по результатам проведенной работы. В нем можно приводить полученные в ходе работы значения. Также можно расписать все, что было сделано в работе.
# 
# </div>

# <div class="alert alert-info"> К сожалению, первый вывод не сохранился, проглядел</div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Выводы описаны грамотно. Приведены ответы на главные вопросы проекта. В выводах можно приводить полученные ранее значения, правильно. Здорово, что по каждому пункту работы приведен вывод. Еще лучше будет, если приведешь рекомендации для компании по дальнейшим действиям. 
# 
# </div>

# <div class="alert alert-info">
# <font size="5"><b>Комментарий ревьюера</b></font>
# 
# Если тебе нравится тема визуализации, то можешь изучить другие методы библиотеки seaborn. Она позволяет строить довольно презентабельные графики.
# 
# Ты выполнил все пункты работы, молодец! Критических замечаний немного. Однако с ними важно поработать. Также есть достаточное число желтых комментариев, которые стоит исправить. Думаю, ты справишься с исправлениями быстро. Жду твою работу :)
# 
# </div>

# <div class="alert alert-success">
# <font size="5"><b>Комментарий ревьюера 2</b></font>
# 
# Помарки исправлены, и теперь работа выполнена хорошо. Классно, что ты. спользуешь так много визуализации в работе. Отличный проект вышел, молодец. Поздравляю со сданным проектом. Надеюсь, он был интересен и познавателен. Спасибо за оставленные комментарии по исправлениям. Успехов в дальнейшем пути :)
# 
# </div>

# <div class="alert alert-info">
# Спасибо
# </div>
