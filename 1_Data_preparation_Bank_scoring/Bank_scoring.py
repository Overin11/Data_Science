#!/usr/bin/env python
# coding: utf-8

# # Исследование надёжности заёмщиков
# 
# Заказчик — кредитный отдел банка. Нужно разобраться, влияет ли семейное положение и количество детей клиента на факт погашения кредита в срок. Входные данные от банка — статистика о платёжеспособности клиентов.
# 
# Результаты исследования будут учтены при построении модели **кредитного скоринга** — специальной системы, которая оценивает способность потенциального заёмщика вернуть кредит банку.

# ## Шаг 1. Откройте файл с данными и изучите общую информацию

# In[11]:


import pandas as pd


# In[12]:


data = pd.read_csv('/datasets/data.csv')
data.info()


# In[13]:


display(data.groupby('children')['children'].count())                                      # кол-во клиентов по кол-ву детей)
display(data.groupby('dob_years')['dob_years'].count())                                    # кол-во клиентов оп возрастам)
display(data.groupby(['education_id', 'education'])['education_id'].count())               # образование
display(data.groupby('gender')['gender'].count())                                          # пол
display(data.groupby('purpose')['purpose'].count())                                        # цели кредитов
display(data.groupby(['family_status',  'family_status_id'])['family_status_id'].count())  # семейное положение / кол-во


# **Вывод**

# В стобцах days_employed, total_income есть значения NaN
# В days_employed данные представлены в float64
# В столбце children есть значения -1 и 20, их сумма составляет меньше 1%, можно убрать
# В столбце dob_years есть 0 в 101 строке, аналогично
# В education названия уровня образования написан в разных регистрах, нужно привести к одному виду
# В gender один клиент не определился с полом, убрать
# В purpose много одинаковых целей записаны в разной форме, привести к общему виду(лемматизировать)
# В family_status/family_status_id проблем нет
# 

# ## Шаг 2. Предобработка данных

# ### Обработка пропусков

# In[14]:


def mean_income(df):
    income_mean = df.groupby('income_type')['total_income'].mean()         # средний доход по типу знятости
    unique_total_income = list(set(df['income_type']))                     # список источников дохода
    dict_mean = dict(income_mean)
    for i in unique_total_income:                                          # заполнение пропусков средними значениями
        df.loc[df["income_type"] == i,'total_income'] = df.loc[df["income_type"] == i,'total_income'].fillna(dict_mean[i])
mean_income(data)

data['days_employed'] = data['days_employed'].apply(abs)                   #убираю "-" в стаже


# In[15]:


def mean_days_employed(df):
    employed_mean = df.groupby('dob_years')['days_employed'].mean()      # ср. стаж по возрасту
    unique_employed_mean = list(set(df['dob_years']))                    # список возрастов
    for i in unique_employed_mean:                                       # заполнение пропусков средними значениями
        df.loc[df["dob_years"] == i,'days_employed'] = df.loc[df["dob_years"] == i,'days_employed'].fillna(dict(employed_mean)[i])
mean_days_employed(data)


# In[16]:


data.loc[data["children"] == 20,'children'] = 2 #сичтаю, что 20 детей опечатка, меняю на 2
data['children'] = data['children'].apply(abs)  #аналогично с -1, убираю знак минус
data = data[data['gender'] != 'XNA']            #убираю клиента без пола
data = data[data['dob_years'] != 0]             #убираю клиентов с возрастом 0


# **Вывод**

# Пропуски в столбце days_employed заполнены средними значениями стажа для каждого возраста.
# В total_income - средними значениями доходов для каждого типа занятости
# Пропуски, возможно, появились из-за невнимательности менеждера банка или клиента, при запонении/проверке анкеты

# ### Замена типа данных

# In[17]:


def experience(df):                                        #перевод стажа из дней в года
    if df['income_type'] != 'пенсионер':
        return df['days_employed']  / 365
    else:
        return df['days_employed']  / 24 / 365
data['years_employed'] =data.apply(experience, axis=1)     #применяю ко всей таблице


# In[18]:


data['days_employed'] = data['days_employed'].astype(int)  #замена типа данных на целочисленныый
data['years_employed'] = data['years_employed'].astype(int)
data['total_income'] = data['total_income'].astype(int)
data.info()
data.head()


# **Вывод**

# Стаж пенсионеров записан неправдоподобно большими числами. При переводе в года получаются сотни лет, если считать, что он записан в днях, как у остальных категорий. Делаю вывод, что стаж записан в часах.
# Столбец со стажем и общим доходом перевел в целые числа

# ### Обработка дубликатов

# In[19]:


display(data.groupby(['education_id', 'education'])['education_id'].count())   #образование / кол-во
print()
data['education'] = data['education'].str.lower()                            # образвание - в нижний регистр
display(data.groupby(['education_id', 'education'])['education_id'].count())


# **Вывод**

# Уровень образования клиента бы записан в разных регистрах

# ### Лемматизация

# In[20]:


from pymystem3 import Mystem
from collections import Counter
m = Mystem()
lemmas_out = ''
for purpose in data['purpose'].unique(): # перебор уникальных значений списка целей кредита
    lemmas = m.lemmatize(purpose)        # выделение лемм уникальных значений
    lemmas_out += ''.join(lemmas)
print(lemmas_out)    
print(Counter(lemmas_out.split()))


# **Вывод**

# Выделил леммы, посчитал самые частоповторяющиеся, на их основе выделил основные категории

# ### Категоризация данных

# In[21]:


data['parents'] = 0                                               # столбец статус родителей
data.loc[data['children'] > 0, 'parents'] = 1


# In[22]:


from pymystem3 import Mystem                                      # категоризация по цели кредитования
m = Mystem()
def replace(purpose):
    lemmas = m.lemmatize(purpose)
    if 'автомобиль' in lemmas:
        return 'Автомобиль'
    if 'образование' in lemmas:
        return 'Образование'
    if 'недвижимость' in lemmas or 'жилье' in lemmas:
        return 'Недвижимость'
    if 'свадьба' in lemmas:
        return 'Свадьба' 

data['main_purpose']=data['purpose'].apply(replace)              #новый столбец с исправленнымим целями
print(data['main_purpose'].value_counts())


# In[23]:


def child_status(df):
    if df < 1:
        return 'без детей'
    elif df > 2:
        return 'многодетный'
    else:
        return '1-2 ребенка'
     
data['child_status'] = data['children'].apply(child_status)
print(data.groupby(['children', 'child_status'])['child_status'].count())


# **Вывод**

# Исходя из целей кредита добавил категории каждому клиенту, также - категорию по количеству детей
# 

# ## Шаг 3. Ответьте на вопросы

# - Есть ли зависимость между наличием детей и возвратом кредита в срок?

# In[24]:


debt_child_depend = data.groupby(['parents', 'debt'])['debt'].count() #зависимость дети\просрочка
debt_child_depend_short = data.groupby('parents')['debt'].count()     # кол-во родителей с просрочкой и без
print('Просрочили кредит без детей {:.2%} от общего числа заемщиков без детей или {} человек'.format(debt_child_depend[0][1] / debt_child_depend_short[0], debt_child_depend[0][1]))
print('Просрочили кредит с детьми {:.2%} от общего числа заемщиков с детьми или {} человек'.format(debt_child_depend[1][1] / debt_child_depend_short[1], debt_child_depend[1][1]))


# **Вывод**

# Есть небольшая разница между количеством клиентов просрочивших кредит среди бездетных и клиентов с детьми, но она незначительная

# - Есть ли зависимость между семейным положением и возвратом кредита в срок?

# In[25]:


family_status_debt = data.groupby(['family_status_id',  'debt'])['debt'].count()     #зависимость семейное положение\просрочка
family_status_debt_short = data.groupby(['family_status_id'])['debt'].count()

print('Просрочили кредит не женат / не замужем {: >13.2%}'.format(family_status_debt[4][1] / family_status_debt_short[4]))
print('Просрочили кредит в разводе {: >25.2%}'.format(family_status_debt[3][1] / family_status_debt_short[3]))
print('Просрочили кредит вдовец / вдова {: >20.2%}'.format(family_status_debt[2][1] / family_status_debt_short[2]))
print('Просрочили кредит гражданский брак {: >18.2%}'.format(family_status_debt[1][1] / family_status_debt_short[1]))
print('Просрочили кредит женат / замужем {: >19.2%}'.format(family_status_debt[0][1] / family_status_debt_short[0]))


# **Вывод**

# Чаще всего, если клиент не женат / не замужем, то он допускает просточку по платежу.
# Вдовец / вдова реже всех допускают такое

# - Есть ли зависимость между уровнем дохода и возвратом кредита в срок?

# In[26]:


low_income = data['total_income'].quantile(.33)            #уровень дохода между средним и низким среди клиентов
high_income = data['total_income'].quantile(.66)           #уровень дохода между средним и высоким среди клиентов

data['income_level'] = 'Средний'
data.loc[data['total_income'] > high_income, 'income_level'] = 'Выше среднего'
data.loc[data['total_income'] < low_income, 'income_level'] = 'Ниже среднего'

income_level_debt = data.groupby(['income_level',  'debt'])['debt'].count()
income_level_debt_short = data.groupby('income_level')['debt'].count()
print('Просрочили кредит с доходом выше среднего {: >14.2%}'.format(income_level_debt['Выше среднего'][1] / income_level_debt_short['Выше среднего']))
print('Просрочили кредит с доходом ниже среднего {: >14.2%}'.format(income_level_debt['Ниже среднего'][1] / income_level_debt_short['Ниже среднего']))
print('Просрочили кредит со средним доходом {: >19.2%}'.format(income_level_debt['Средний'][1] / income_level_debt_short['Средний']))


# **Вывод**

# Хуже всего дела с возвратом кредита всрок обстоят у клиентов со средним доходом, у клиентов с доходом выше среднего - наоборот

# - Как разные цели кредита влияют на его возврат в срок?

# In[27]:


main_purpose_debt = data.groupby(['main_purpose',  'debt'])['debt'].count()     #зависимость семейное цуль кредита\просрочка
main_purpose_debt_short = data.groupby('main_purpose')['debt'].count()

print('Просрочили кредит на автомобиль {: >22.2%}'.format(main_purpose_debt['Автомобиль'][1] / main_purpose_debt_short['Автомобиль']))
print('Просрочили кредит на недвижимость {: >20.2%}'.format(main_purpose_debt['Недвижимость'][1] / main_purpose_debt_short['Недвижимость']))
print('Просрочили кредит на оборазование {: >20.2%}'.format(main_purpose_debt['Образование'][1] / main_purpose_debt_short['Образование']))
print('Просрочили кредит на свадьбу {: >25.2%}'.format(main_purpose_debt['Свадьба'][1] / main_purpose_debt_short['Свадьба']))


# **Вывод**

# Кредит на автомобиль возвращеют хуже всего, почти как на образование. Клиенты с кредииом на недвидимость самые надёжные

# ## Шаг 4. Общий вывод

# Самый ответственный клиент - это бездетный вдовец / вдова с доходом выше среднего с целью кредита - на недвижимость. И , если клиент не женат / не замужем со средним доходом и детьми, вероятность просрочка кредит на автомобиль будет выше всех остальных 
